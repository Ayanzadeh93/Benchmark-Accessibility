# Extended ROUGE Metrics Support

## Overview

The VQA evaluation pipeline now supports **all ROUGE types** available in the `rouge-score` library.

## Supported ROUGE Types

| Type | Metric Name | Description |
|------|-------------|-------------|
| **ROUGE-1** | `rouge1_f1` | Unigram (1-word) overlap |
| **ROUGE-2** | `rouge2_f1` | Bigram (2-word phrase) overlap |
| **ROUGE-3** | `rouge3_f1` | Trigram (3-word phrase) overlap |
| **ROUGE-4** | `rouge4_f1` | 4-gram overlap |
| **ROUGE-L** | `rougeL_f1` | Longest Common Subsequence |
| **ROUGE-Lsum** | `rougeLsum_f1` | LCS computed per sentence, then aggregated |

## What Changed

### Files Updated

1. **`vqa/evaluation/metrics.py`**
   - Extended default ROUGE types from `[rouge1, rouge2, rougeL]` to include `rouge3`, `rouge4`, and `rougeLsum`

2. **`vqa/evaluation/evaluate_dual_mode.py`**
   - VQA2 evaluation now computes all 6 ROUGE types
   - Per-question metrics include all ROUGE types
   - CSV output includes all types

3. **`vqa/evaluation/eval_schemas.py`**
   - Added `rouge3_f1`, `rouge4_f1`, `rougeLsum_f1` to all result schemas

4. **`vqa/evaluation/evaluate_vqa_simple.py`**
   - CSV output extended to include all ROUGE types

5. **`vqa/evaluation/vqa_standard_evaluation.py`**
   - Updated to include all ROUGE types in results

6. **`vqa/evaluation/model_evaluation.py`**
   - Extended to save all ROUGE metrics

## Usage

### Automatic (Default)

All ROUGE types are computed by default:

```python
from vqa.evaluation.metrics import compute_rouge_scores

predictions = ["The cat sits on the mat"]
references = ["The cat is on the mat"]

# Returns: rouge1_f1, rouge2_f1, rouge3_f1, rouge4_f1, rougeL_f1, rougeLsum_f1
scores = compute_rouge_scores(predictions, references)
```

### Custom Types

You can specify which types to compute:

```python
# Only compute ROUGE-1 and ROUGE-L
scores = compute_rouge_scores(
    predictions, 
    references, 
    rouge_types=["rouge1", "rougeL"]
)
```

## Output Examples

### JSON Output (`evaluation_vqa2.json`)

```json
{
  "mode": "vqa2",
  "total_samples": 1000,
  "rouge1_f1": 0.7234,
  "rouge2_f1": 0.5421,
  "rouge3_f1": 0.4123,
  "rouge4_f1": 0.3245,
  "rougeL_f1": 0.6987,
  "rougeLsum_f1": 0.7011,
  "bleu": 0.5234,
  "bertscore_f1": 0.8567
}
```

### CSV Output (`evaluation_vqa2_summary.csv`)

```csv
metric,value
mode,vqa2
total_samples,1000
rouge1_f1,0.7234
rouge2_f1,0.5421
rouge3_f1,0.4123
rouge4_f1,0.3245
rougeL_f1,0.6987
rougeLsum_f1,0.7011
bleu,0.5234
bertscore_f1,0.8567
```

## What's NOT Included

The `rouge-score` library does **not** support:

- **ROUGE-W** (Weighted LCS)
- **ROUGE-S** (Skip-bigram)
- **ROUGE-SU** (Skip-bigram + unigram)

These types are defined in the original Lin (2004) paper but require the Perl ROUGE toolkit or other implementations.

## Testing

Run the test script to verify all ROUGE types work:

```bash
python -m vqa.evaluation.test_rouge_extended
```

Expected output:
```
ROUGE Scores:
==================================================
  rouge1_f1      : 0.9267
  rouge2_f1      : 0.7424
  rouge3_f1      : 0.5481
  rouge4_f1      : 0.4286
  rougeL_f1      : 0.9267
  rougeLsum_f1   : 0.9267

✓ All ROUGE types computed successfully!
```

## Interpretation

- **Higher n-grams** (ROUGE-3, ROUGE-4) capture longer phrase matches
  - Useful for evaluating fluency and phrasal correctness
  - Typically lower scores than ROUGE-1/2 (fewer exact matches)

- **ROUGE-L** vs **ROUGE-Lsum**
  - ROUGE-L: LCS on entire text
  - ROUGE-Lsum: LCS per sentence, then averaged (better for multi-sentence text)
  - For short VQA answers, they're often similar

## References

- Lin, C. Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. ACL Workshop.
- Google Research: `rouge-score` Python library
