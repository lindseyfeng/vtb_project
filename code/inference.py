from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch
import random

# Load the model and tokenizer
model_name = "../qwen2-vtb-dpo-sft/final_checkpoint"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the CSV file
df = pd.read_csv('../test_set.csv')
random.seed(42)
# Assuming the CSV has a column named 'Content'
prompts = df['Prompt'].tolist()
sampled_prompts = random.sample(prompts, 10)
model.generation_config.pad_token_id = tokenizer.pad_token_id
# Function to generate a response with a length of 100 tokens
def generate_response(prompt, max_new_tokens=50):
    print("prompt:", prompt)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move tensors to the appropriate device
    generation_config = {
        'max_new_tokens': max_new_tokens,  # Adjust for input length + inputs['input_ids'].shape[1]
        'do_sample': True,
        # Uncomment and adjust these parameters as needed:
        'top_k': 50,
        'top_p': 0.8,
        'temperature': 0.9,
        'no_repeat_ngram_size': 5,  # No repetition of 2-grams
        'repetition_penalty':1, 
    }
    outputs = model.generate(**inputs, **generation_config)
    generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print("response", response)
    print()
    return response

# Generate responses for all prompts in the CSV
responses = [generate_response(prompt) for prompt in sampled_prompts]


# # Save the dataframe with responses to a new CSV file
# df.to_csv('output_with_responses.csv', index=False)

# print("Responses generated and saved to 'output_with_responses.csv'")
