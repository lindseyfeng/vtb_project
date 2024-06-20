import pandas as pd
from datasets import Dataset
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['Prompt'][i]}\n ### Answer: {example['Chosen']}"
        output_texts.append(text)
    return output_texts


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# Load your dataset
df = pd.read_csv('../train_set.csv')

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)
print(dataset)

sft_config = SFTConfig(
    dataset_text_field="Content",
    max_seq_length=512,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8, 
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