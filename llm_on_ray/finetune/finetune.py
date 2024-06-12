#
# Copyright 2023 The LLM-on-Ray Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import copy

#!/usr/bin/env python

import os
import argparse
import sys
import re

from typing import Any, Dict, Union, Optional

from itertools import chain

import torch
import datasets
import transformers
import wandb

from peft import get_peft_model, LoraConfig

import ray
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from ray.air import RunConfig, FailureConfig

from pydantic_yaml import parse_yaml_raw_as

from llm_on_ray import common
from llm_on_ray.finetune import template
from llm_on_ray.finetune.finetune_config import FinetuneConfig
from importlib import util

IGNORE_INDEX = -100


def adapt_transformers_to_device(config: Dict):
    device = config["Training"]["device"]
    if device in ["hpu"]:
        from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

        # adapt transformers to gaudi
        adapt_transformers_to_gaudi()


def set_seed(config: Dict):
    seed = config["Training"].get("seed", None)
    if seed is None:
        return
    device = config["Training"]["device"]
    if device in ["cpu", "gpu"]:
        from accelerate.utils import set_seed as _set_seed

        _set_seed(seed)
    elif device in ["hpu"]:
        from optimum.habana.utils import set_seed as _set_seed

        _set_seed(seed)


def convert_to_training_args(cls, config: Dict):
    device = config["Training"]["device"]
    accelerate_mode = config["Training"]["accelerate_mode"]
    save_strategy = config["General"]["save_strategy"]

    args = {
        "output_dir": config["General"]["output_dir"],
        "report_to": config["General"]["report_to"],
        "resume_from_checkpoint": config["General"]["resume_from_checkpoint"],
        "gradient_checkpointing": config["General"]["enable_gradient_checkpointing"],
        "save_strategy": save_strategy if save_strategy != "False" else "no",
        "bf16": config["Training"]["mixed_precision"] == "bf16",
        "num_train_epochs": config["Training"]["epochs"],
        "per_device_train_batch_size": config["Training"]["batch_size"],
        "per_device_eval_batch_size": config["Training"]["batch_size"],
        "optim": config["Training"]["optimizer"],
        "learning_rate": config["Training"]["learning_rate"],
        "logging_steps": config["Training"]["logging_steps"],
        "lr_scheduler_type": config["Training"]["lr_scheduler"],
        "weight_decay": config["Training"]["weight_decay"],
        "gradient_accumulation_steps": config["Training"]["gradient_accumulation_steps"],
        "do_train": True,
        "warmup_ratio": 0.03,
        "log_level": "info",
        "save_total_limit": 10,
    }

    # set attr do_eval
    vf = config["Dataset"].get("validation_file", None)
    vsp = config["Dataset"].get("validation_split_percentage", 0)
    if vf is not None or (vsp / 100 > 0.0 and vsp / 100 < 1.0):
        args.update({"do_eval": True})

    # set attr max_steps
    if config["Training"]["max_train_steps"] is not None:
        args.update({"max_steps": config["Training"]["max_train_steps"]})

    # set attr for device cpu
    if device == "cpu":
        if hasattr(cls, "use_cpu"):
            args.update({"use_cpu": True})
        if hasattr(cls, "no_cuda"):
            args.update({"no_cuda": True})
        args.update({"use_ipex": True})

    # set attr 'deepspeed'
    if accelerate_mode == "DEEPSPEED":
        args.update({"deepspeed": config["Training"]["deepspeed_config_file"]})

    # set attr for FSDP
    # if accelerate_mode == "FSDP":
    #     args.updatwe({})

    # set attr for Intel Gaudi
    if device == "hpu":
        args.update({"use_habana": True})
        args.update({"use_lazy_mode": config["Training"]["hpu_execution_mode"] == "lazy"})
        args.update({"pipelining_fwd_bwd": True})

    return cls(**args)


def convert_dtype(dtype: str) -> Optional[torch.dtype]:
    supported_dtypes = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "no": None,
    }
    return supported_dtypes[dtype]


def load_tokenizer(config: Dict):
    if config["General"].get("tokenizer_name") is not None:
        tokenizer_name = config["General"].get("tokenizer_name")
    else:
        tokenizer_name = config["General"]["base_model"]
    load_config = config["General"].get("config", {})
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, **load_config)
    return tokenizer


def load_dataset(config: Dict):
    dataset_file = config["Dataset"].get("train_file", None)
    if dataset_file is None:
        return

    if os.path.exists(dataset_file):
        # load from local file
        def local_load(name, **load_config):
            if os.path.isfile(name):
                file = os.path.basename(os.path.abspath(name))
                path = os.path.dirname(os.path.abspath(name))
                dataset = datasets.load_dataset(path, data_files=file, **load_config)
            else:
                dataset = datasets.load_dataset(name, **load_config)
            return dataset["train"]

        train_dataset = local_load(dataset_file)
        validation_file = config["Dataset"].get("validation_file", None)
        if validation_file is not None:
            validation_dataset = local_load(validation_file)
            return datasets.DatasetDict({"train": train_dataset, "validation": validation_dataset})

        validation_split_percentage = config["Dataset"].get("validation_split_percentage", 0)
        if validation_split_percentage / 100 > 0.0 and validation_split_percentage / 100 < 1.0:
            dataset_dict = train_dataset.train_test_split(
                test_size=validation_split_percentage / 100
            )
            dataset_dict["validation"] = dataset_dict["test"]
            return dataset_dict

        return datasets.DatasetDict({"train": train_dataset})
    else:
        # try to download and load dataset from huggingface.co
        load_config = config["General"].get("config", {})
        use_auth_token = load_config.get("use_auth_token", None)
        raw_dataset = datasets.load_dataset(dataset_file, use_auth_token=use_auth_token)

        validation_split_percentage = config["Dataset"].get("validation_split_percentage", 0)
        if "validation" not in raw_dataset.keys() and (
            validation_split_percentage / 100 > 0.0 and validation_split_percentage / 100 < 1.0
        ):
            dataset_dict = raw_dataset["train"].train_test_split(
                test_size=validation_split_percentage / 100
            )
            dataset_dict["validation"] = dataset_dict["test"]
            return dataset_dict

        return raw_dataset


def tokenize_dataset(config: Dict, tokenizer, dataset):
    max_length = config["Dataset"].get("max_length", 512)
    group = config["Dataset"].get("group", True)
    config["Dataset"].get("block_size", 512)
    tokenizer.pad_token = tokenizer.eos_token

    if isinstance(dataset, datasets.Dataset):
        column_names = dataset.column_names

    if isinstance(dataset, datasets.DatasetDict):
        column_names = dataset["train"].column_names

    print("before")
    print(dataset)

    # if column_names and template.TEXT_COLUMN_NAME not in column_names:
    #
    #     def prompt_SlimOrca_to_alpaca(rec):
    #         default_system = "You are a helpful, respectful and honest assistant."
    #         examples = {}
    #         conv = rec["conversations"]
    #         # system
    #         if conv[0]["from"] != "system":
    #             examples["system"] = default_system
    #             start = 0
    #         elif conv[0]["from"] == "system" and conv[0]["value"] == "":
    #             examples[conv[0]["from"]] = default_system
    #             start = 1
    #         else:
    #             examples[conv[0]["from"]] = conv[0]["value"]
    #             start = 1
    #
    #         for j in range(start, len(conv) - 1, 2):
    #             examples[conv[j]["from"]] = conv[j]["value"]
    #             examples[conv[j + 1]["from"]] = conv[j + 1]["value"]
    #         instruction = (examples["system"],)
    #         response = (examples["gpt"],)
    #         input = (examples["human"],)
    #         if not instruction:
    #             raise ValueError(f"Expected an instruction in: {rec}")
    #         if not response:
    #             raise ValueError(f"Expected a response in: {rec}")
    #
    #         if input:
    #             rec["text"] = template.PROMPT_WITH_INPUT_FORMAT.format(
    #                 instruction=instruction, response=response, input=input
    #             )
    #         else:
    #             rec["text"] = template.PROMPT_NO_INPUT_FORMAT.format(
    #                 instruction=instruction, response=response
    #             )
    #         return rec
    #
    #     def prompt(rec):
    #         instruction = rec["instruction"]
    #         response = rec["response"]
    #         context = rec.get("context")
    #         if not instruction:
    #             raise ValueError(f"Expected an instruction in: {rec}")
    #         if not response:
    #             raise ValueError(f"Expected a response in: {rec}")
    #         if context:
    #             rec["text"] = template.PROMPT_WITH_INPUT_FORMAT.format(
    #                 instruction=instruction, response=response, input=context
    #             )
    #         else:
    #             rec["text"] = template.PROMPT_NO_INPUT_FORMAT.format(
    #                 instruction=instruction, response=response
    #             )
    #         return rec

        # dataset = dataset.map(
        #     prompt_SlimOrca_to_alpaca,
        #     load_from_cache_file=False,
        #     desc="Prompt",
        # )
        # column_names += [template.TEXT_COLUMN_NAME]


    def prompt_slim_orca(examples, tokenizer):
        INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        default_system = "You are a helpful, respectful and honest assistant."
        user = "### Instruction:\n"
        assistant = "### Response:\n"
        # end = "### End:\n"
        end = tokenizer.eos_token
        prompts = {}
        prompts["prompt_sources"] = []
        prompts["prompt_targets"] = []

        for conv in examples:
            conv = conv["conversations"]
            # system
            if conv[0]["from"] != "system":
                prompt = INTRO_BLURB + end + "\n" + user + default_system
                start = 0
            elif conv[0]["from"] == "system" and conv[0]["value"] == "":
                prompt = INTRO_BLURB + end + "\n" + user + default_system
                start = 1
            else:
                prompt = INTRO_BLURB + end + "\n" + user + conv[0]["value"]
                start = 1

            for j in range(start, len(conv) - 1, 2):
                u = conv[j]["value"]
                ass = conv[j + 1]["value"]
                prompt = prompt + " Input:" + u + end + "\n" + assistant
                response = ass + end
                prompts["prompt_sources"].append(prompt)
                prompts["prompt_targets"].append(response)

                prompt += response + "\n"

        return prompts

    def prompt_SlimOrca(examples, tokenizer):
        system = "### System:\n"
        default_system = "You are a helpful, respectful and honest assistant."
        user = "### User:\n"
        assistant = "### Assistant:\n"
        end = tokenizer.eos_token
        prompts = {}
        prompts["prompt_sources"] = []
        prompts["prompt_targets"] = []
        for conv in examples:
            conv = conv["conversations"]

            # system
            if conv[0]["from"] != "system":
                prompt = system + default_system + end + "\n"
                start = 0
            elif conv[0]["from"] == "system" and conv[0]["value"] == "":
                prompt = system + default_system + end + "\n"
                start = 1
            else:
                prompt = system + conv[0]["value"] + end + "\n"
                start = 1

            for j in range(start, len(conv) - 1, 2):
                u = conv[j]["value"]
                ass = conv[j + 1]["value"]
                prompt = prompt + user + u + end + "\n" + assistant
                response = ass + end
                prompts["prompt_sources"].append(prompt)
                prompts["prompt_targets"].append(response)

                prompt += response + "\n"

        return prompts

    for key in dataset:
        prompts = prompt_slim_orca(dataset[key], tokenizer)
        dataset[key] = datasets.Dataset.from_dict(prompts)

    print("after")
    print(dataset)

    def tokenize_function(examples):
        keys = list(examples.data.keys())
        if len(keys) != 2:
            raise ValueError("Unsupported dataset format")

        st = [s + t for s, t in zip(examples[keys[0]], examples[keys[1]])]
        return tokenizer(
            st,
            padding=False,
            truncation=True,
            return_tensors=None,
            max_length=max_length,
        )

    def truncate_sequences(sequences, max_length):
        words_to_cut = sum(list(map(len, sequences))) - max_length
        if words_to_cut <= 0:
            return sequences

        while words_to_cut > 0 and len(sequences) > 0:
            words_to_cut -= len(sequences[0])
            sequences = sequences[1:]

        return sequences

    def preprocess_slim_orca_function(examples):
        max_seq_length = 512
        max_source_length = 384
        assistant = "### Response:\n"
        end = tokenizer.eos_token
        assistant_tokens = tokenizer.tokenize(assistant)
        header = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            + end
            + "\n"
        )

        instructions = [q.strip() for q in examples["prompt_sources"]]
        responses = [q.strip() for q in examples["prompt_targets"]]

        examples["input_ids"] = []
        examples["labels"] = []
        examples["attention_mask"] = []

        for instruction, response in zip(instructions, responses):
            convs = re.findall(
                r"### Instruction.*?{0}|### Response.*?{0}".format(end), instruction, re.DOTALL
            )
            convs_tokens = [tokenizer.tokenize(conv) + tokenizer.tokenize("\n") for conv in convs]
            header_tokens = tokenizer.tokenize(header) + tokenizer.tokenize("\n")

            max_input = max_source_length - len(header_tokens) - len(assistant_tokens)

            truncated_convs = truncate_sequences(convs_tokens, max_input)

            if len(truncated_convs) == 0:
                truncated_convs = [convs_tokens[-1][: max_input - 3] + convs_tokens[-1][-3:]]

            prompt_tokens = [header_tokens] + truncated_convs + [assistant_tokens]
            prompt_ids = [
                tokenizer.convert_tokens_to_ids(prompt_token) for prompt_token in prompt_tokens
            ]
            prompt_ids = list(chain(*prompt_ids))

            resp_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response.strip()))
            # keep last and eos_id
            max_resp = max_seq_length - len(prompt_ids) - 1
            if len(resp_ids) > max_resp:
                resp_ids = resp_ids[: max_resp - 1] + resp_ids[-1:]

            input_ids = prompt_ids + resp_ids + [tokenizer.eos_token_id]
            labels = [IGNORE_INDEX] * len(prompt_ids) + resp_ids + [tokenizer.eos_token_id]

            # padding
            input_len = len(input_ids)
            pad_len = max_seq_length - input_len
            input_ids = input_ids + [tokenizer.eos_token_id] * pad_len
            labels = labels + [IGNORE_INDEX] * pad_len
            attention_mask = [1] * input_len + [0] * pad_len

            assert len(input_ids) == max_seq_length
            assert len(prompt_ids) <= max_source_length
            assert len(labels) == len(input_ids) == len(attention_mask)

            examples["input_ids"].append(torch.tensor(input_ids))
            examples["labels"].append(labels)
            examples["attention_mask"].append(attention_mask)

        return examples

    def tokenize(prompt, add_eos_token=True):
        results = tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors=None,
        )
        for i in range(len(results["input_ids"])):
            if (
                results["input_ids"][i][-1] != tokenizer.eos_token_id
                and len(results["input_ids"][i]) < 512
                and add_eos_token
            ):
                results["input_ids"][i].append(tokenizer.eos_token_id)
                results["attention_mask"][i].append(1)

        results["labels"] = copy.deepcopy(results["input_ids"])
        results["input_id_len"] = [len(result) for result in results["input_ids"]]
        return results

    def preprocess_function(examples):
        keys = list(examples.data.keys())
        if len(keys) != 2:
            raise ValueError("Unsupported dataset format")

        st = [s + t for s, t in zip(examples[keys[0]], examples[keys[1]])]

        examples_tokenized = tokenize(st)
        input_ids = examples_tokenized["input_ids"]
        labels = examples_tokenized["labels"]
        sources_tokenized = tokenize(examples[keys[0]], add_eos_token=False)
        for label, source_len in zip(labels, sources_tokenized["input_id_len"]):
            label[:source_len] = [IGNORE_INDEX] * source_len
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": examples_tokenized["attention_mask"],
        }

    def preprocess_slimorca_function(examples):
        max_seq_length = 512
        max_source_length = 384
        assistant = "### Assistant:\n"
        end = tokenizer.eos_token
        assistant_tokens = tokenizer.tokenize(assistant)

        instructions = [q.strip() for q in examples["prompt_sources"]]
        responses = [q.strip() for q in examples["prompt_targets"]]

        examples["input_ids"] = []
        examples["labels"] = []
        examples["attention_mask"] = []

        for instruction, response in zip(instructions, responses):
            header = re.findall(r"### System.*?{}".format(end), instruction, re.DOTALL)[0]
            convs = re.findall(
                r"### User.*?{0}|### Assistant.*?{0}".format(end), instruction, re.DOTALL
            )
            convs_tokens = [tokenizer.tokenize(conv) + tokenizer.tokenize("\n") for conv in convs]
            header_tokens = tokenizer.tokenize(header) + tokenizer.tokenize("\n")

            max_input = max_source_length - len(header_tokens) - len(assistant_tokens)

            truncated_convs = truncate_sequences(convs_tokens, max_input)

            if len(truncated_convs) == 0:
                truncated_convs = [convs_tokens[-1][: max_input - 3] + convs_tokens[-1][-3:]]

            prompt_tokens = [header_tokens] + truncated_convs + [assistant_tokens]
            prompt_ids = [
                tokenizer.convert_tokens_to_ids(prompt_token) for prompt_token in prompt_tokens
            ]
            prompt_ids = list(chain(*prompt_ids))

            resp_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response.strip()))
            # keep last and eos_id
            max_resp = max_seq_length - len(prompt_ids) - 1
            if len(resp_ids) > max_resp:
                resp_ids = resp_ids[: max_resp - 1] + resp_ids[-1:]

            input_ids = prompt_ids + resp_ids + [tokenizer.eos_token_id]
            labels = [IGNORE_INDEX] * len(prompt_ids) + resp_ids + [tokenizer.eos_token_id]

            # padding
            input_len = len(input_ids)
            pad_len = max_seq_length - input_len
            input_ids = input_ids + [tokenizer.eos_token_id] * pad_len
            labels = labels + [IGNORE_INDEX] * pad_len
            attention_mask = [1] * input_len + [0] * pad_len

            assert len(input_ids) == max_seq_length
            assert len(prompt_ids) <= max_source_length
            assert len(labels) == len(input_ids) == len(attention_mask)

            examples["input_ids"].append(torch.tensor(input_ids))
            examples["labels"].append(labels)
            examples["attention_mask"].append(attention_mask)

        return examples

    # column_names = list(dataset["train"].features)
    print("remove_columns")
    print(column_names)
    tokenized_dataset = dataset.map(
        tokenize_function,
        load_from_cache_file=False,
        batched=True,
        remove_columns=column_names,
        desc="Tokenize dataset",
    )

    if group:

        def concatenate_data(dataset, max_seq_length):
            concatenated_dataset = {}
            for column in dataset.features:
                concatenated_data = [item for sample in dataset[column] for item in sample]
                reshaped_data = [
                    concatenated_data[i * max_seq_length : (i + 1) * max_seq_length]
                    for i in range(len(concatenated_data) // max_seq_length)
                ]
                concatenated_dataset[column] = reshaped_data
            concatenated_dataset["labels"] = copy.deepcopy(concatenated_dataset["input_ids"])
            return datasets.Dataset.from_dict(concatenated_dataset)

        tokenized_dataset["train"] = concatenate_data(tokenized_dataset["train"], 512)
    return tokenized_dataset


def prepare_data_collator(config: Dict, tokenizer):
    return transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
    )


def load_model(config: Dict):
    model_name = config["General"]["base_model"]
    model_dtype = convert_dtype(config["Training"].get("mixed_precision", "no"))
    model_config = config["General"].get("config", {})
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=model_dtype, **model_config
    )

    lora_config = config["General"].get("lora_config", None)
    if lora_config:
        peft_config = LoraConfig(**lora_config)
        model = get_peft_model(model, peft_config)

    egc = config["General"].get("enable_gradient_checkpointing", False)
    if egc:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    model.to(dtype=model_dtype, device=torch.device(config["Training"]["device"]))

    return model


def get_trainer(config: Dict, model, tokenizer, tokenized_dataset, data_collator):
    device = config["Training"]["device"]
    if device in ["cpu", "gpu"]:
        from transformers import Trainer, TrainingArguments

        training_args = convert_to_training_args(TrainingArguments, config)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"]
            if tokenized_dataset.get("validation") is not None
            else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        return training_args, trainer
    elif device in ["hpu"]:
        from optimum.habana.transformers import GaudiTrainer
        from optimum.habana.transformers import GaudiTrainingArguments
        from optimum.habana import GaudiConfig

        # If gaudi_config_name is provided, load gaudi_config from huggingface model hub(https://huggingface.co/Habana), otherwise use default gaudi_config
        gaudi_config_name = config["General"].get("gaudi_config_name", None)
        if gaudi_config_name is not None:
            gaudi_config = GaudiConfig.from_pretrained(gaudi_config_name)
        else:
            gaudi_config = GaudiConfig()
            gaudi_config.use_fused_adam = True
            gaudi_config.use_fused_clip_norm = True

        training_args = convert_to_training_args(GaudiTrainingArguments, config)
        trainer = GaudiTrainer(
            model=model,
            args=training_args,
            gaudi_config=gaudi_config,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"]
            if tokenized_dataset.get("validation") is not None
            else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        return training_args, trainer
    return None


def train_func(config: Dict[str, Any]):
    os.chdir(config["cwd"])
    adapt_transformers_to_device(config)

    set_seed(config)

    tokenizer = load_tokenizer(config)

    dataset = load_dataset(config)

    tokenized_dataset = tokenize_dataset(config, tokenizer, dataset)
    print("train_func tokenized_dataset")
    print(tokenized_dataset)
    if "train" not in tokenized_dataset:
        raise ValueError("--do_train requires a train dataset")
    print(tokenized_dataset["train"])
    data_collator = prepare_data_collator(config, tokenizer)

    model = load_model(config)

    training_args, trainer = get_trainer(config, model, tokenizer, tokenized_dataset, data_collator)

    common.logger.info("train start")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    common.logger.info("train finish")


def get_finetune_config():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )

    # Print help if no arguments were provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file) as f:
        finetune_config = parse_yaml_raw_as(FinetuneConfig, f)
    return finetune_config.dict()


def main(external_config=None):
    if not external_config:
        config = get_finetune_config()
    else:
        config = external_config

    config["cwd"] = os.getcwd()

    num_training_workers = config["Training"].get("num_training_workers")
    resources_per_worker = config["Training"].get("resources_per_worker")

    if config["Training"].get("accelerate_mode", None) is None:
        config["Training"][
            "accelerate_mode"
        ] = "DDP"  # will use DDP to accelerate if no method specified

    ccl_worker_count = 1
    device = config["Training"]["device"]
    if device != "cpu":
        ccl_worker_count = num_training_workers

    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "OMP_NUM_THREADS": str(resources_per_worker["CPU"]),
                "CCL_ZE_IPC_EXCHANGE": "sockets",
                "CCL_WORKER_COUNT": str(ccl_worker_count),
                "CCL_LOG_LEVEL": "info",
                "FI_TCP_IFACE": "lo",
                "FI_PROVIDER": "tcp",
            }
        }

        if config["General"]["gpt_base_model"] is True:
            runtime_env["pip"] = ["transformers==4.26.0"]

        if device == "gpu":
            num_cpus = (
                resources_per_worker["CPU"] * num_training_workers + 1
            )  # additional 1 for head worker
            ray.init(num_cpus=num_cpus, runtime_env=runtime_env)
        else:
            ray.init(runtime_env=runtime_env)

    common.logger.info(f"ray available resources = {ray.available_resources()}")
    use_gpu = True if device == "gpu" else False
    scaling_config = ScalingConfig(
        num_workers=num_training_workers,
        use_gpu=use_gpu,
        resources_per_worker=resources_per_worker,
        placement_strategy="SPREAD",
    )

    # if try to use Intel GPU, convert device to 'xpu'
    # due to accelerate internal use 'xpu' represent Intel GPU
    if device == "gpu":
        from accelerate.utils import is_xpu_available

        if is_xpu_available():
            device = "xpu"

    if config.get("torch_config", None) is None:
        backend = None
        if device == "cpu" or device == "xpu" or device == "gpu":
            backend = "ccl"
        elif device == "hpu":
            backend = "hccl"
        torch_config = common.TorchConfig(backend=backend, device=device)
    else:
        customer_torch_config = config.get("torch_config")
        torch_config = common.TorchConfig(**customer_torch_config, device=device)

    if config.get("failure_config", None) is None:
        failure_config = FailureConfig()
    else:
        customer_failure_config = config.get("failure_config")
        failure_config = FailureConfig(**customer_failure_config)

    if config.get("run_config", None) is None:
        run_config = RunConfig(failure_config=failure_config)
    else:
        customer_run_config = config.get("run_config")
        if customer_run_config.get("failure_config", None) is None:
            customer_run_config["failure_config"] = failure_config
        run_config = RunConfig(**customer_run_config)

    trainer = TorchTrainer(
        train_func,
        train_loop_config=config,
        scaling_config=scaling_config,
        torch_config=torch_config,
        run_config=run_config,
    )
    results = trainer.fit()
    if external_config is not None:
        return results


if __name__ == "__main__":
    main()
