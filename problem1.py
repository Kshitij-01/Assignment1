from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set random seed for reproducibility
torch.manual_seed(42)

# Step 1: Load model and tokenizer
print("Loading distilgpt2 model and tokenizer...")
tok_gpt2 = AutoTokenizer.from_pretrained("distilgpt2")
model_gpt2 = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Add padding token if it doesn't exist
if tok_gpt2.pad_token is None:
    tok_gpt2.pad_token = tok_gpt2.eos_token

print("Model loaded successfully!\n")

# Step 2: Base prompt
base_prompt = """You are given a purchase request. Extract a JSON object with fields item and quantity.

Text: "Order three boxes of blue markers for the design team."

JSON:"""

print("Base prompt:")
print(base_prompt)
print("\n" + "="*80 + "\n")

# Tokenize the prompt
inputs = tok_gpt2.encode(base_prompt, return_tensors="pt")

# Generate a few test samples
print("Generating test samples...")
print("-" * 80)

for i in range(3):
    outputs = model_gpt2.generate(
        inputs,
        max_new_tokens=60,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tok_gpt2.eos_token_id
    )
    
    # Decode the generated text
    generated_text = tok_gpt2.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the generated part (after the prompt)
    generated_part = generated_text[len(base_prompt):].strip()
    
    print(f"\nSample {i+1}:")
    print(generated_part)
    print("-" * 80)

