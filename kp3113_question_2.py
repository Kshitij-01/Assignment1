"""
Problem 2: Supervised Fine-Tuning (Instruction Tuning)
Author: kp3113
"""

import json
import random
import time
import torch
import numpy as np
import os

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)

MODEL_NAME = "google/flan-t5-small"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "sft_dataset.json")

random.seed(42)
torch.manual_seed(42)


def load_dataset():
    """Load training and evaluation data from JSON"""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    train_data = data["train"]
    eval_data = data["eval"]
    random.shuffle(train_data)
    random.shuffle(eval_data)
    return train_data, eval_data


train_data, eval_data = load_dataset()
print(f"Training: {len(train_data)} examples | Evaluation: {len(eval_data)} examples")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# T5 tokenizer maps curly braces to <unk>, so we add them as regular tokens
tokenizer.add_tokens(["{", "}"])

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model.to(device)


def exact_match(pred, label):
    return 1.0 if pred.strip() == label.strip() else 0.0


def compute_extraction_metrics(pred, label):
    """Check JSON validity and field accuracy"""
    res = {"json_valid": 0.0, "field_match": 0.0}
    pred_clean = pred.strip()
    
    try:
        obj_pred = json.loads(pred_clean)
        if isinstance(obj_pred, dict) and "item" in obj_pred and "quantity" in obj_pred:
            res["json_valid"] = 1.0
            obj_label = json.loads(label)
            item_match = str(obj_pred["item"]).strip().lower() == str(obj_label["item"]).strip().lower()
            qty_match = int(obj_pred["quantity"]) == int(obj_label["quantity"])
            if item_match and qty_match:
                res["field_match"] = 1.0
    except:
        pass
    
    return res


def evaluate_model(model, tokenizer, eval_data, return_examples=False, example_offset=0, json_examples_indices=None, sentiment_examples_indices=None):
    """Evaluate on sentiment classification and JSON extraction"""
    model.eval()
    device = next(model.parameters()).device
    
    sent_total, sent_correct = 0, 0
    ext_total, json_valid, field_match = 0, 0.0, 0.0
    sent_examples = []
    ext_examples = []
    sent_count = 0
    ext_count = 0
    
    # Collect all examples with their eval_data indices
    all_sent_examples = []
    all_ext_examples = []
    
    for idx, ex in enumerate(eval_data):
        enc = tokenizer(ex["input"], return_tensors="pt", truncation=True, max_length=128)
        enc = {k: v.to(device) for k, v in enc.items()}
        
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=48, num_beams=3, early_stopping=True)
        pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        
        if ex["task"] == "sentiment":
            sent_total += 1
            sent_correct += exact_match(pred, ex["target"])
            if return_examples:
                all_sent_examples.append({
                    "eval_idx": idx,
                    "input": ex["input"],
                    "target": ex["target"],
                    "pred": pred
                })
            sent_count += 1
        else:
            ext_total += 1
            m = compute_extraction_metrics(pred, ex["target"])
            json_valid += m["json_valid"]
            field_match += m["field_match"]
            if return_examples:
                all_ext_examples.append({
                    "eval_idx": idx,  # Index in eval_data
                    "input": ex["input"], 
                    "target": ex["target"], 
                    "pred": pred,
                    "complexity": len(ex["input"]) + len(ex["target"])  # Simple complexity metric
                })
            ext_count += 1
    
    # Select sentiment examples
    if return_examples and all_sent_examples:
        if sentiment_examples_indices is not None:
            # Use provided indices (indices into eval_data)
            for target_idx in sentiment_examples_indices:
                for sent_ex in all_sent_examples:
                    if sent_ex["eval_idx"] == target_idx:
                        sent_examples.append({"input": sent_ex["input"], "target": sent_ex["target"], "pred": sent_ex["pred"]})
                        break
        else:
            # Use offset-based selection
            count = 0
            for sent_ex in all_sent_examples:
                if len(sent_examples) < 2 and count >= example_offset:
                    sent_examples.append({"input": sent_ex["input"], "target": sent_ex["target"], "pred": sent_ex["pred"]})
                count += 1
    
    # Select complex JSON examples if indices provided, otherwise use offset
    if return_examples and all_ext_examples:
        if json_examples_indices is not None:
            # Use provided indices (indices into eval_data)
            for target_idx in json_examples_indices:
                for ext_ex in all_ext_examples:
                    if ext_ex["eval_idx"] == target_idx:
                        ext_examples.append({"input": ext_ex["input"], "target": ext_ex["target"], "pred": ext_ex["pred"]})
                        break
        else:
            # Select most complex ones
            sorted_ext = sorted(all_ext_examples, key=lambda x: x["complexity"], reverse=True)
            ext_examples = [{"input": ex["input"], "target": ex["target"], "pred": ex["pred"]} 
                           for ex in sorted_ext[:2]]
    
    result = {
        "sentiment_accuracy": sent_correct / sent_total if sent_total > 0 else 0.0,
        "json_validity": json_valid / ext_total if ext_total > 0 else 0.0,
        "json_field_match": field_match / ext_total if ext_total > 0 else 0.0
    }
    
    if return_examples:
        result["sentiment_examples"] = sent_examples
        result["extraction_examples"] = ext_examples
    
    return result


# Find most complex JSON examples to show (same for both evaluations)
extraction_examples_with_complexity = []
for i, ex in enumerate(eval_data):
    if ex["task"] == "extraction":
        complexity = len(ex["input"]) + len(ex["target"]) + len(ex["target"].split())
        extraction_examples_with_complexity.append((i, ex, complexity))
# Sort by complexity and get indices of top 2
extraction_examples_with_complexity.sort(key=lambda x: x[2], reverse=True)
complex_json_indices = [x[0] for x in extraction_examples_with_complexity[:2]]

# Find sentiment examples with different sentiment levels - one from each level
sentiment_by_level = {"very_negative": [], "negative": [], "neutral": [], "positive": [], "very_positive": []}
for i, ex in enumerate(eval_data):
    if ex["task"] == "sentiment":
        target = ex["target"]
        if target in sentiment_by_level:
            sentiment_by_level[target].append(i)

# Select one example from each of the 5 sentiment levels for both baseline and after SFT
baseline_sent_indices = []
after_sft_sent_indices = []
sentiment_order = ["very_negative", "negative", "neutral", "positive", "very_positive"]

for level in sentiment_order:
    if sentiment_by_level[level]:
        # Baseline: use first example from each level
        baseline_sent_indices.append(sentiment_by_level[level][0])
        # After SFT: use a different example if available, otherwise use the same
        if len(sentiment_by_level[level]) > 1:
            after_sft_sent_indices.append(sentiment_by_level[level][1])
        else:
            after_sft_sent_indices.append(sentiment_by_level[level][0])

# Baseline evaluation (before fine-tuning)
print("\nBaseline evaluation...")
baseline = evaluate_model(model, tokenizer, eval_data, return_examples=True, 
                         json_examples_indices=complex_json_indices, 
                         sentiment_examples_indices=baseline_sent_indices)
print(f"Task A Accuracy: {baseline['sentiment_accuracy']:.3f}")
print(f"Task B JSON Validity: {baseline['json_validity']:.3f}")

# Baseline example outputs
print("\n" + "="*60)
print("BASELINE EXAMPLE OUTPUTS (Before SFT)")
print("="*60)
print("\nTask A (Sentiment Classification):")
sentiment_labels = ["very_negative", "negative", "neutral", "positive", "very_positive"]
for i, ex in enumerate(baseline.get("sentiment_examples", [])):
    label = sentiment_labels[i] if i < len(sentiment_labels) else f"Example {i+1}"
    print(f"\n{label.capitalize().replace('_', ' ')}:")
    print(f"  Input:  {ex['input']}")
    print(f"  Target: {ex['target']}")
    print(f"  Pred:   {ex['pred']}")

print("\nTask B (JSON Extraction):")
for i, ex in enumerate(baseline.get("extraction_examples", []), 1):
    print(f"\nExample {i}:")
    print(f"  Input:  {ex['input']}")
    print(f"  Target: {ex['target']}")
    print(f"  Pred:   {ex['pred']}")
print("="*60)


# Dataset class for training
class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_input=128, max_target=48):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_target = max_target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        inp = self.tokenizer(ex["input"], max_length=self.max_input, truncation=True, padding=False)
        tgt = self.tokenizer(text_target=ex["target"], max_length=self.max_target, truncation=True, padding=False)
        return {"input_ids": inp["input_ids"], "attention_mask": inp["attention_mask"], "labels": tgt["input_ids"]}


train_dataset = SFTDataset(train_data, tokenizer)
eval_dataset = SFTDataset(eval_data, tokenizer)


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.where(preds == -100, tokenizer.pad_token_id, preds)
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    preds_dec = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels_dec = tokenizer.batch_decode(labels, skip_special_tokens=True)
    correct = sum(1 for p, l in zip(preds_dec, labels_dec) if p.strip() == l.strip())
    return {"eval_accuracy": correct / len(preds_dec) if preds_dec else 0.0}


print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./sft_checkpoints",
    learning_rate=5e-4,
    num_train_epochs=40,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    logging_steps=25,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=48,
    save_total_limit=3,
    report_to=[],
    dataloader_num_workers=0,
    remove_unused_columns=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

print("\nFine-tuning...")
t_start = time.time()
trainer.train()
train_time = time.time() - t_start

trainer.model.save_pretrained("./sft_model")
tokenizer.save_pretrained("./sft_model")
print(f"Training time: {train_time:.2f}s")

# Final evaluation
print("\nEvaluating fine-tuned model...")
sft_metrics = evaluate_model(trainer.model, tokenizer, eval_data, return_examples=True, 
                             json_examples_indices=complex_json_indices, 
                             sentiment_examples_indices=after_sft_sent_indices)

# Results
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"{'Task':<6}{'Metric':<18}{'Before SFT':<14}{'After SFT':<14}")
print("-"*60)
print(f"{'A':<6}{'Accuracy':<18}{baseline['sentiment_accuracy']:<14.3f}{sft_metrics['sentiment_accuracy']:<14.3f}")
print(f"{'B':<6}{'JSON Validity':<18}{baseline['json_validity']:<14.3f}{sft_metrics['json_validity']:<14.3f}")
print(f"{'B':<6}{'Field Match':<18}{baseline['json_field_match']:<14.3f}{sft_metrics['json_field_match']:<14.3f}")
print("="*60)
print(f"Training time: {train_time:.2f}s | Parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")

# Example outputs
print("\n" + "="*60)
print("EXAMPLE OUTPUTS (After SFT)")
print("="*60)
print("\nTask A (Sentiment Classification):")
sentiment_labels = ["very_negative", "negative", "neutral", "positive", "very_positive"]
for i, ex in enumerate(sft_metrics.get("sentiment_examples", [])):
    label = sentiment_labels[i] if i < len(sentiment_labels) else f"Example {i+1}"
    print(f"\n{label.capitalize().replace('_', ' ')}:")
    print(f"  Input:  {ex['input']}")
    print(f"  Target: {ex['target']}")
    print(f"  Pred:   {ex['pred']}")

print("\nTask B (JSON Extraction):")
for i, ex in enumerate(sft_metrics.get("extraction_examples", []), 1):
    print(f"\nExample {i}:")
    print(f"  Input:  {ex['input']}")
    print(f"  Target: {ex['target']}")
    print(f"  Pred:   {ex['pred']}")
print("="*60)

"""
ANALYSIS

Basically what happened with Task A (Sentiment) is that before fine-tuning, the model was 
getting around 43% accuracy which is not great for a 5-class problem. Looking at the examples, 
it was confusing similar classes - predicting "negative" instead of "very_negative", or 
"positive" instead of "very_positive". After SFT, accuracy jumped to about 87%. The model 
learned to distinguish between the intensity levels properly. Logically this makes sense 
because instruction tuning teaches the model the exact output format I expect.

For Task B (JSON Extraction), the situation was more interesting. Before SFT, I got 0% JSON 
validity - the model just output plain text like "15 whiteboard erasers" instead of proper 
JSON. After training, I hit 100% JSON validity. Field match is around 77% - the model 
sometimes shortens item names (like "printer cartridges" instead of "printer ink cartridges") 
but the structure is always correct.

The main blocker I ran into was that FLAN-T5's tokenizer maps curly braces {} to <unk> tokens, 
which then get stripped during decoding. So the model literally could not output braces no 
matter how much I trained it. The trick was to add them as regular tokens using 
tokenizer.add_tokens(["{", "}"]) and then resize the model embeddings. Once I did that, 
JSON validity went from 0% to 100% immediately. This taught me that SFT is really about 
format alignment - the pretrained model already understands the concepts, it just needs to 
learn the exact output format. And always check if special characters are in your tokenizer's 
vocabulary first.
"""
