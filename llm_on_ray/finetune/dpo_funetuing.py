from peft import LoraConfig
from transformers import AutoModelForCausalLM


class DPOFuneTuning:
    def __init__(self, config):
        self.config = config

    def get_model(self):
        # load policy model
        model = AutoModelForCausalLM.from_pretrained(
            self.config["General"]["base_model"],
            config=self.config,
            low_cpu_mem_usage=True,
            use_auth_token=True if self.config["General"]["config"]["use_auth_token"] else None,
        )
        model.config.use_cache = False
        return model

    def get_model_ref(self):
        # load reference model
        model_ref = AutoModelForCausalLM.from_pretrained(
            self.config["General"]["base_model"],
            config=self.config,
            low_cpu_mem_usage=True,
            use_auth_token=True if self.config["General"]["config"]["use_auth_token"] else None,
        )
        model_ref.config.use_cache = False
        return model_ref

    def dpo_train(self, training_args, train_datasets, validation_datasets, tokenizer):
        from trl import DPOTrainer

        lora_config = self.config["General"].get("lora_config", None)

        return DPOTrainer(
            self.get_model(),
            self.get_model_ref() if lora_config is not None else None,
            args=training_args,
            beta=self.config["Training"].get("beta"),
            train_dataset=train_datasets,
            eval_dataset=validation_datasets,
            tokenizer=tokenizer,
            peft_config=LoraConfig(**lora_config) if lora_config is not None else None,
            max_length=self.config["Dataset"].get("max_length"),
            force_use_ref_model=True if lora_config is not None else False,
        )


class GaudiDPOFuneTuning(DPOFuneTuning):
    def dpo_train(self, training_args, train_datasets, validation_datasets, tokenizer):
        from optimum.habana.trl import GaudiDPOTrainer as DPOTrainer

        lora_config = self.config["General"].get("lora_config", None)
        return DPOTrainer(
            self.get_model(),
            self.get_model_ref() if lora_config is not None else None,
            args=training_args,
            beta=self.config["Training"].get("beta"),
            train_dataset=train_datasets,
            eval_dataset=validation_datasets,
            tokenizer=tokenizer,
            peft_config=LoraConfig(**lora_config) if lora_config is not None else None,
            max_length=self.config["Dataset"].get("max_length"),
        )
