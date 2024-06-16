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
import os
import argparse
import sys
from typing import Any, Dict, Union, Optional

from itertools import chain

import torch
import datasets
import transformers

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


class Finetuning:
    def adapt_transformers_to_device(self, config: Dict):
        device = config["Training"]["device"]
        if device in ["hpu"]:
            from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

            # adapt transformers to gaudi
            adapt_transformers_to_gaudi()

    def set_seed(self, config: Dict):
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

    def convert_to_training_args(self, cls, config: Dict):
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

    def convert_dtype(self, dtype: str) -> Optional[torch.dtype]:
        supported_dtypes = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "no": None,
        }
        return supported_dtypes[dtype]

    def load_tokenizer(self, config: Dict):
        if config["General"].get("tokenizer_name") is not None:
            tokenizer_name = config["General"].get("tokenizer_name")
        else:
            tokenizer_name = config["General"]["base_model"]
        load_config = config["General"].get("config", {})
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, **load_config)
        return tokenizer

    def load_dataset(self, config: Dict):
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
                return datasets.DatasetDict(
                    {"train": train_dataset, "validation": validation_dataset}
                )

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

    def tokenize_dataset(self, config: Dict, tokenizer, dataset):
        max_length = config["Dataset"].get("max_length", 512)
        group = config["Dataset"].get("group", True)
        block_size = config["Dataset"].get("block_size", 512)
        tokenizer.pad_token = tokenizer.eos_token

        if isinstance(dataset, datasets.Dataset):
            column_names = dataset.column_names

        if isinstance(dataset, datasets.DatasetDict):
            column_names = dataset["train"].column_names

        if column_names and template.TEXT_COLUMN_NAME not in column_names:

            def prompt(rec):
                instruction = rec["instruction"]
                response = rec["response"]
                context = rec.get("context")
                if not instruction:
                    raise ValueError(f"Expected an instruction in: {rec}")
                if not response:
                    raise ValueError(f"Expected a response in: {rec}")
                if context:
                    rec["text"] = template.PROMPT_WITH_INPUT_FORMAT.format(
                        instruction=instruction, response=response, input=context
                    )
                else:
                    rec["text"] = template.PROMPT_NO_INPUT_FORMAT.format(
                        instruction=instruction, response=response
                    )
                return rec

            dataset = dataset.map(
                prompt,
                load_from_cache_file=False,
                desc="Prompt",
            )
            column_names += [template.TEXT_COLUMN_NAME]

        def tokenize_function(examples):
            return tokenizer(examples[template.TEXT_COLUMN_NAME], max_length=max_length)

        tokenized_dataset = dataset.map(
            tokenize_function,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Tokenize dataset",
        )

        if group:

            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                if total_length >= block_size:
                    total_length = (total_length // block_size) * block_size
                # Split by chunks of max_len.
                result = {
                    k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                    for k, t in concatenated_examples.items()
                }
                result["labels"] = result["input_ids"].copy()
                return result

            tokenized_dataset = tokenized_dataset.map(
                group_texts,
                batched=True,
                load_from_cache_file=False,
                desc=f"Grouping texts in chunks of {block_size}",
            )

        return tokenized_dataset

    def prepare_data_collator(self, config: Dict, tokenizer):
        return transformers.DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
        )

    def load_model(self, config: Dict):
        model_name = config["General"]["base_model"]
        model_dtype = self.convert_dtype(config["Training"].get("mixed_precision", "no"))
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

    def get_trainer(self, config: Dict, model, tokenizer, tokenized_dataset, data_collator):
        device = config["Training"]["device"]
        if device in ["cpu", "gpu"]:
            from transformers import Trainer, TrainingArguments

            training_args = self.convert_to_training_args(TrainingArguments, config)
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

            training_args = self.convert_to_training_args(GaudiTrainingArguments, config)
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