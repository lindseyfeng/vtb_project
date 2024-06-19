from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch

# Load the model and tokenizer
model_name = "../dpo_qwen2-0.5b/final_checkpoint"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the CSV file
df = pd.read_csv('../test_set.csv')

# Assuming the CSV has a column named 'Content'
prompts = df['Content'].tolist()[:5]

# Function to generate a response with a length of 100 tokens
def generate_response(prompt, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move tensors to the appropriate device
    generation_config = {
        'max_new_tokens': max_new_tokens,  # Adjust for input length + inputs['input_ids'].shape[1]
        'do_sample': True,
        # Uncomment and adjust these parameters as needed:
        'top_k': 50,
        'top_p': 0.8,
        'temperature': 0.7,
    }
    outputs = model.generate(**inputs, **generation_config)
    generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print(response)
    return response

# Generate responses for all prompts in the CSV
responses = [generate_response(prompt) for prompt in prompts]


# # Save the dataframe with responses to a new CSV file
# df.to_csv('output_with_responses.csv', index=False)

# print("Responses generated and saved to 'output_with_responses.csv'")
