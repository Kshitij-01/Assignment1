# Homework 1: Language Modeling and Supervised Fine-Tuning

Author: kp3113

**Note:** This README file was written with AI assistance. However, all analysis, results interpretation, and conclusions were done and written by me.

## Files

- `kp3113_question_1.py` - Problem 1: Decoding Controls and Structured Output
- `kp3113_question_2.py` - Problem 2: Supervised Fine-Tuning
- `sft_dataset.json` - Training/evaluation data for Problem 2
- `requirements.txt` - Python dependencies

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python kp3113_question_1.py
python kp3113_question_2.py
```

## Notes

### Problem 1: Results Summary

| Configuration | Prompt Type | Distinct-1 | Distinct-2 | Mean Length | JSON Validity |
|---------------|-------------|------------|------------|-------------|---------------|
| Greedy | Base | 0.2500 | 0.2558 | 44.00 | 0.00% |
| Greedy | Schema | 0.3500 | 0.3684 | 20.00 | 0.00% |
| Temperature=0.7 | Base | 0.3537 | 0.4058 | 42.80 | 0.00% |
| Temperature=0.7 | Schema | 0.5898 | 0.6125 | 29.50 | 0.00% |
| Temperature=1.0 | Base | 0.6148 | 0.7218 | 34.70 | 0.00% |
| Temperature=1.0 | Schema | 0.7629 | 0.8557 | 31.70 | 10.00% |
| Top-k=40 | Base | 0.3562 | 0.3827 | 41.20 | 0.00% |
| Top-k=40 | Schema | 0.5242 | 0.5669 | 27.20 | 0.00% |
| Top-k=200 | Base | 0.3123 | 0.3400 | 44.70 | 0.00% |
| Top-k=200 | Schema | 0.4657 | 0.5084 | 27.50 | 0.00% |
| Top-p=0.8 | Base | 0.2523 | 0.2605 | 44.00 | 0.00% |
| Top-p=0.8 | Schema | 0.4592 | 0.4800 | 26.90 | 0.00% |
| Top-p=0.95 | Base | 0.3249 | 0.3512 | 43.10 | 0.00% |
| Top-p=0.95 | Schema | 0.3763 | 0.3947 | 21.40 | 0.00% |

**Key Findings:**
- Temperature=1.0 with Schema Prompt achieved the highest diversity (Distinct-1: 0.7629) and the only non-zero JSON validity (10.00%)
- Base prompt configurations produced 0% JSON validity across all decoding strategies
- Schema prompt reduced output length but only improved JSON validity for Temperature=1.0
- Higher temperature increases diversity by flattening the probability distribution

### Problem 2: Tokenizer Fix for JSON Output

The FLAN-T5 tokenizer maps curly braces `{` and `}` to the `<unk>` (unknown) token, 
which gets stripped during decoding. This caused 0% JSON validity initially because 
the model never saw braces during training.

**Fix applied:**
```python
tokenizer.add_tokens(["{", "}"])
model.resize_token_embeddings(len(tokenizer))
```

This adds braces as regular vocabulary tokens (not special tokens, which would also 
get stripped). After this fix, the model achieved 100% JSON validity.

### Problem 2: Results Summary

| Task | Metric | Before SFT | After SFT |
|------|--------|------------|-----------|
| A | Sentiment Accuracy | 0.433 | 0.867 |
| B | JSON Validity | 0.000 | 1.000 |
| B | Field Match | 0.000 | 0.767 |

**Training Details:**
- Training examples: 331
- Evaluation examples: 60 (30 sentiment, 30 extraction)
- Training time: ~36 seconds
- Model parameters: 76,934,528

**Example Outputs (Before SFT):**

Task A (Sentiment Classification):
- Input: "Terrible quality and the company refused to help. Never buying from them again."
  - Target: `very_negative`
  - Pred: `negative` ❌
- Input: "Not impressed. The quality is subpar and doesn't match the description."
  - Target: `negative`
  - Pred: `negative` ✅
- Input: "The product is functional. It does what it's supposed to do, nothing more, nothing less."
  - Target: `neutral`
  - Pred: `very_negative` ❌

Task B (JSON Extraction):
- Input: "We need 15 whiteboard erasers for the conference room."
  - Target: `{"item": "whiteboard erasers", "quantity": 15}`
  - Pred: `15 whiteboard erasers` ❌ (not JSON format)
- Input: "We require 50 units of printer ink cartridges."
  - Target: `{"item": "printer ink cartridges", "quantity": 50}`
  - Pred: `printer ink cartridges` ❌ (not JSON format)

**Example Outputs (After SFT):**

Task A (Sentiment Classification):
- Input: "This is the worst product I've ever bought. Complete waste of money and time. I want a full refund immediately."
  - Target: `very_negative`
  - Pred: `very_negative` ✅
- Input: "The item arrived damaged and the customer service was unhelpful. Very disappointed."
  - Target: `negative`
  - Pred: `negative` ✅
- Input: "Standard product. Gets the job done without any surprises."
  - Target: `neutral`
  - Pred: `neutral` ✅
- Input: "Very nice quality. Glad I bought it and would recommend."
  - Target: `positive`
  - Pred: `positive` ✅
- Input: "This is absolutely phenomenal! I'm ecstatic and it's the best thing I've ever bought!"
  - Target: `very_positive`
  - Pred: `very_positive` ✅

Task B (JSON Extraction):
- Input: "We need 15 whiteboard erasers for the conference room."
  - Target: `{"item": "whiteboard erasers", "quantity": 15}`
  - Pred: `{ "item": "whiteboard erasers", "quantity": 15}` ✅ (valid JSON)
- Input: "We require 50 units of printer ink cartridges."
  - Target: `{"item": "printer ink cartridges", "quantity": 50}`
  - Pred: `{ "item": "printer cartridges", "quantity": 50}` ✅ (valid JSON, minor item name shortening)

---

## User Analysis

The following analysis sections are taken directly from the code comments in `kp3113_question_1.py` and `kp3113_question_2.py`. These represent my own analysis and interpretation of the results.

### Problem 1: Analysis

Basically what I found is that Temperature=1.0 gave the highest diversity - around 0.61 distinct-1 with the base prompt and 0.76 with the schema prompt. Logically this makes sense because higher temperature flattens the probability distribution, so the model becomes more willing to pick less likely tokens instead of always going for the safe choice. Greedy decoding had the lowest diversity (0.25 distinct-1) since it deterministically picks the most probable token every time.

The interesting part was JSON validity. With the base prompt, I got 0% JSON validity across all configurations - the model just rambled on without producing any valid JSON. When I added the schema prompt that explicitly says "Output must be valid JSON exactly: {...}", only Temperature=1.0 managed to produce valid JSON (10% validity). The other configurations still got 0%.

The main issue here is that distilgpt2 was not trained to follow instructions. It's just a causal language model that predicts the next token based on what it's seen before. So even when I tell it "output valid JSON", it doesn't really understand that as a command. The only reason Temperature=1.0 worked at all is because the high randomness let it occasionally stumble onto a valid JSON structure by chance.

If I wanted to actually get reliable JSON output, I'd need to either use constrained decoding (like the outlines library) to force the model to only generate valid JSON tokens, or switch to an instruction-tuned model like Flan-T5 that was actually trained to follow format instructions.

### Problem 2: Analysis

Basically what happened with Task A (Sentiment) is that before fine-tuning, the model was getting around 43% accuracy which is not great for a 5-class problem. Looking at the examples, it was confusing similar classes - predicting "negative" instead of "very_negative", or "positive" instead of "very_positive". After SFT, accuracy jumped to about 87%. The model learned to distinguish between the intensity levels properly. Logically this makes sense because instruction tuning teaches the model the exact output format I expect.

For Task B (JSON Extraction), the situation was more interesting. Before SFT, I got 0% JSON validity - the model just output plain text like "15 whiteboard erasers" instead of proper JSON. After training, I hit 100% JSON validity. Field match is around 77% - the model sometimes shortens item names (like "printer cartridges" instead of "printer ink cartridges") but the structure is always correct.

The main blocker I ran into was that FLAN-T5's tokenizer maps curly braces {} to <unk> tokens, which then get stripped during decoding. So the model literally could not output braces no matter how much I trained it. The trick was to add them as regular tokens using `tokenizer.add_tokens(["{", "}"])` and then resize the model embeddings. Once I did that, JSON validity went from 0% to 100% immediately. This taught me that SFT is really about format alignment - the pretrained model already understands the concepts, it just needs to learn the exact output format. And always check if special characters are in your tokenizer's vocabulary first.
