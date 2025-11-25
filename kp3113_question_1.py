"""
Problem 1: Decoding Controls and Structured Output
Author: kp3113
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# Load distilgpt2
print("Loading distilgpt2 model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model = model.to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully!\n")

# Prompts
base_prompt = """You are given a purchase request. Extract a JSON object with fields item and quantity.

Text: "Order three boxes of blue markers for the design team."

JSON:"""

schema_prompt = """You are given a purchase request. Extract a JSON object with fields item and quantity.

Text: "Order three boxes of blue markers for the design team."

Output must be valid JSON exactly: {"item": "<string>", "quantity": <integer>}. No comments.

JSON:"""

print("Base prompt:")
print(base_prompt)
print("\n" + "="*80 + "\n")


def compute_distinct_n(text, n):
    """Ratio of unique n-grams to total n-grams - measures lexical diversity"""
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def compute_repetition_rate(text):
    """Fraction of adjacent identical tokens - high values indicate degenerate output"""
    tokens = text.split()
    if len(tokens) < 2:
        return 0.0
    adjacent_repeats = sum(1 for i in range(len(tokens) - 1) if tokens[i] == tokens[i+1])
    return adjacent_repeats / (len(tokens) - 1)


def compute_json_validity(text):
    """Check if output contains valid JSON with item and quantity fields"""
    try:
        start_idx = text.find('{')
        if start_idx == -1:
            return False
        
        # Match braces to extract JSON substring
        brace_count = 0
        end_idx = start_idx
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if brace_count != 0:
            return False
        
        parsed = json.loads(text[start_idx:end_idx])
        if isinstance(parsed, dict) and 'item' in parsed and 'quantity' in parsed:
            int(parsed['quantity'])  # verify quantity is numeric
            return True
        return False
    except:
        return False


def generate_samples(prompt, config_name, num_samples=10):
    """Generate samples using specified decoding strategy"""
    samples = []
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    for _ in range(num_samples):
        gen_kwargs = {
            "max_new_tokens": 60,
            "pad_token_id": tokenizer.eos_token_id,
        }
        
        if config_name == "Greedy":
            gen_kwargs["do_sample"] = False
        elif config_name.startswith("Temperature"):
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(config_name.split("=")[1])
        elif config_name.startswith("Top-k"):
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = 0.7
            gen_kwargs["top_k"] = int(config_name.split("=")[1])
        elif config_name.startswith("Top-p"):
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = 0.7
            gen_kwargs["top_p"] = float(config_name.split("=")[1])
        
        outputs = model.generate(inputs, **gen_kwargs)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        samples.append(generated_text[len(prompt):].strip())
    
    return samples


def compute_metrics(samples):
    """Compute averaged metrics for a batch of samples"""
    results = {"distinct_1": [], "distinct_2": [], "lengths": [], "repetition_rates": [], "json_valid": []}
    
    for sample in samples:
        results["distinct_1"].append(compute_distinct_n(sample, 1))
        results["distinct_2"].append(compute_distinct_n(sample, 2))
        results["lengths"].append(len(sample.split()))
        results["repetition_rates"].append(compute_repetition_rate(sample))
        results["json_valid"].append(compute_json_validity(sample))
    
    n = len(samples)
    return {
        "distinct_1": sum(results["distinct_1"]) / n,
        "distinct_2": sum(results["distinct_2"]) / n,
        "mean_length": sum(results["lengths"]) / n,
        "repetition_rate": sum(results["repetition_rates"]) / n,
        "json_validity_rate": sum(results["json_valid"]) / n
    }


# Decoding configurations to test
configurations = [
    "Greedy",
    "Temperature=0.7",
    "Temperature=1.0",
    "Top-k=40",
    "Top-k=200",
    "Top-p=0.8",
    "Top-p=0.95"
]

# Run experiments with base prompt
results_base = {}
print("Generating samples and computing metrics (BASE PROMPT)...")
print("="*80)

for config in configurations:
    print(f"\nConfiguration: {config}")
    print("-" * 80)
    samples = generate_samples(base_prompt, config)
    metrics = compute_metrics(samples)
    results_base[config] = {"samples": samples, "metrics": metrics}
    
    print(f"Distinct-1: {metrics['distinct_1']:.4f}")
    print(f"Distinct-2: {metrics['distinct_2']:.4f}")
    print(f"Mean Length: {metrics['mean_length']:.2f}")
    print(f"Repetition Rate: {metrics['repetition_rate']:.4f}")
    print(f"JSON Validity Rate: {metrics['json_validity_rate']:.2%}")

# Run experiments with schema prompt
print("\n" + "="*80)
print("Generating samples and computing metrics (SCHEMA PROMPT)...")
print("="*80)
print("\nSchema prompt:")
print(schema_prompt)
print("\n" + "="*80)

results_schema = {}

for config in configurations:
    print(f"\nConfiguration: {config}")
    print("-" * 80)
    samples = generate_samples(schema_prompt, config)
    metrics = compute_metrics(samples)
    results_schema[config] = {"samples": samples, "metrics": metrics}
    
    print(f"Distinct-1: {metrics['distinct_1']:.4f}")
    print(f"Distinct-2: {metrics['distinct_2']:.4f}")
    print(f"Mean Length: {metrics['mean_length']:.2f}")
    print(f"Repetition Rate: {metrics['repetition_rate']:.4f}")
    print(f"JSON Validity Rate: {metrics['json_validity_rate']:.2%}")

# Summary tables
print("\n" + "="*80)
print("SUMMARY TABLE - BASE PROMPT")
print("="*80)
print(f"{'Configuration':<20} {'Dist-1':<10} {'Dist-2':<10} {'Mean Len':<10} {'Rep Rate':<12} {'JSON Valid':<12}")
print("-" * 80)

for config in configurations:
    m = results_base[config]["metrics"]
    print(f"{config:<20} {m['distinct_1']:<10.4f} {m['distinct_2']:<10.4f} {m['mean_length']:<10.2f} {m['repetition_rate']:<12.4f} {m['json_validity_rate']:<12.2%}")

print("\n" + "="*80)
print("SUMMARY TABLE - SCHEMA PROMPT")
print("="*80)
print(f"{'Configuration':<20} {'Dist-1':<10} {'Dist-2':<10} {'Mean Len':<10} {'Rep Rate':<12} {'JSON Valid':<12}")
print("-" * 80)

for config in configurations:
    m = results_schema[config]["metrics"]
    print(f"{config:<20} {m['distinct_1']:<10.4f} {m['distinct_2']:<10.4f} {m['mean_length']:<10.2f} {m['repetition_rate']:<12.4f} {m['json_validity_rate']:<12.2%}")

print("\n" + "="*80)
print("JSON VALIDITY COMPARISON")
print("="*80)
print(f"{'Configuration':<20} {'Base Prompt':<15} {'Schema Prompt':<15} {'Improvement':<15}")
print("-" * 80)

for config in configurations:
    base_valid = results_base[config]["metrics"]["json_validity_rate"]
    schema_valid = results_schema[config]["metrics"]["json_validity_rate"]
    improvement = schema_valid - base_valid
    print(f"{config:<20} {base_valid:<15.2%} {schema_valid:<15.2%} {improvement:+.2%}")

print("="*80)

"""
ANALYSIS

Basically what I found is that Temperature=1.0 gave the highest diversity - around 0.61 
distinct-1 with the base prompt and 0.76 with the schema prompt. Logically this makes sense 
because higher temperature flattens the probability distribution, so the model becomes more 
willing to pick less likely tokens instead of always going for the safe choice. Greedy 
decoding had the lowest diversity (0.25 distinct-1) since it deterministically picks the 
most probable token every time.

The interesting part was JSON validity. With the base prompt, I got 0% JSON validity across 
all configurations - the model just rambled on without producing any valid JSON. When I 
added the schema prompt that explicitly says "Output must be valid JSON exactly: {...}", 
only Temperature=1.0 managed to produce valid JSON (10% validity). The other configurations 
still got 0%.

The main issue here is that distilgpt2 was not trained to follow instructions. It's just a 
causal language model that predicts the next token based on what it's seen before. So even 
when I tell it "output valid JSON", it doesn't really understand that as a command. The only 
reason Temperature=1.0 worked at all is because the high randomness let it occasionally 
stumble onto a valid JSON structure by chance.

If I wanted to actually get reliable JSON output, I'd need to either use constrained decoding 
(like the outlines library) to force the model to only generate valid JSON tokens, or switch 
to an instruction-tuned model like Flan-T5 that was actually trained to follow format 
instructions.
"""
