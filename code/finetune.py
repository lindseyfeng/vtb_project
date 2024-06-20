import pandas as pd
from datasets import Dataset
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


def formatting_prompts_func(example):
    print(example)
    for i in range(len(example['Prompt'])):
        text = f"### Question: {example['Prompt'][i]}\n ### Answer: {example['Chosen']}"

    return output_texts


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")

# Load your dataset
df = pd.read_csv('../train_set.csv')

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)
print(dataset)

sft_config = SFTConfig(
    output_dir="./qwen2-vtb-dpo-sft",
)
trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=SFTConfig(output_dir="/tmp"),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)
trainer.train()