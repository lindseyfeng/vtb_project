
# 0. imports
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union, List, Literal
import torch.nn as nn
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from peft import PeftModel
import json
from trl import DPOTrainer, DPOConfig
import wandb
import torch.nn.functional as F
from trl import create_reference_model
from contextlib import contextmanager, nullcontext
import warnings
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
wandb.login(key="11308deb4bc611b19b149922fcd5c1406b496347")

from typing import List, Dict, Any, Optional
import torch
from torch.nn.utils.rnn import pad_sequence

class DPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.
    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
    """

    def __init__(self, pad_token_id: int = 0, label_pad_token_id: int = -100, is_encoder_decoder: Optional[bool] = False):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.is_encoder_decoder = is_encoder_decoder

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Filter out any features that have None values
        features = [f for f in features if all(value is not None for value in f.values())]

        if not features:
            raise ValueError("No valid features provided after filtering out None values.")

        # First, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in features]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.startswith(("chosen", "rejected", "completion")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in features]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in features]
                    if k.endswith("_input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k.endswith("_logps"):
                # the cached reference model logprobs
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch



    
# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="./qwen2-vtb/checkpoint-500",
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-5, metadata={"help": "optimizer learning rate"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="rmsprop", metadata={"help": "the optimizer type"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "num epoch"})
    per_device_train_batch_size: Optional[int] = field(default=32, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=32, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=64, metadata={"help": "the lora r parameter"})
    max_target_length: Optional[int] = field(default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"})
    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./dpo_qwen2-0.5b", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    train_path: Optional[str] = field(
        default="train_data.json",
    )
    val_path: Optional[str] = field(
        default="val_data.json",
    )

def get_train_data() -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    # return dataset.map(split_prompt_and_responses)
    df = pd.read_csv('train_set.csv')
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        return {
            "prompt": sample["Prompt"],
            "chosen": sample["Chosen"],
            "rejected": sample["Reject"],
        }
    return dataset.map(split_prompt_and_responses)


def get_test_data() -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    # return dataset.map(split_prompt_and_responses)
    df = pd.read_csv('test_set.csv')
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        return {
            "prompt": sample["Prompt"],
            "chosen": sample["Chosen"],
            "rejected": sample["Reject"],
        }
    return dataset.map(split_prompt_and_responses)

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path)
    model_ref = create_reference_model(model)


    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the Stack-exchange paired dataset
    train_dataset = get_train_data()
    train_dataset = train_dataset.filter(lambda x: x['chosen'] is not None and x['rejected'] is not None)
    print(train_dataset)
    first_entry = train_dataset[0]
    print("Prompt:", first_entry['prompt'])
    print("Chosen Response:", first_entry['chosen'])
    print("Rejected Response:", first_entry['rejected'])

    # 3. Load evaluation dataset
    eval_dataset = get_test_data()
    eval_dataset = eval_dataset.filter(lambda x: x['chosen'] is not None and x['rejected'] is not None)
    print(eval_dataset)

    # train_dataset = load_json(script_args.train_path)
    # eval_dataset = load_json(script_args.val_path)

    # 4. initialize training arguments:
    training_args = DPOConfig(
        do_eval = True,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        logging_steps=script_args.logging_steps,
        logging_first_step=True,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        save_strategy="steps",
        max_steps=script_args.max_steps,
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        remove_unused_columns=False,
        run_name="dpo_qwen2",
        num_train_epochs=script_args.num_train_epochs,
        save_total_limit=4,
        bf16=True,
    )



    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        max_target_length=script_args.max_target_length,
        force_use_ref_model = True,
        data_collator = DPODataCollatorWithPadding(tokenizer.pad_token_id, -100, model.config.is_encoder_decod)
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.evaluate()

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)