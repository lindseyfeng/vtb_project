import pandas as pd
from datasets import Dataset
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


def formatting_prompts_func(example):
    print("len", len(example['Prompt']))
    output_texts = []
    for i in range(len(example['Prompt'])):
        text = f"### Question: {example['Prompt'][i]}\n ### Answer: {example['Chosen'][i]}"
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

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=SFTConfig(output_dir="./qwen2-vtb-dpo-sft"),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)
trainer.train()