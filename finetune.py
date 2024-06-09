import pandas as pd
from datasets import Dataset
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")

# Load your dataset
df = pd.read_csv('train_set.csv')

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)
print(dataset)

sft_config = SFTConfig(
    dataset_text_field="Content",
    max_seq_length=512,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8, 
    output_dir="./qwen2-vtb",
)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset=dataset,
    args=sft_config,
)
trainer.train()
