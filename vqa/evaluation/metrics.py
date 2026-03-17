"""VQA evaluation metrics: accuracy, ROUGE, BLEU, per-question breakdown."""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False

try:
    import bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

try:
    import clip
    import torch
    from PIL import Image
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


def exact_match_accuracy(predictions: List[str], references: List[str]) -> float:
    """
    Compute exact match accuracy (case-insensitive).
    
    Args:
        predictions: List of predicted answers (e.g. ["A", "B", "C"])
        references: List of ground truth answers (e.g. ["A", "C", "C"])
    
    Returns:
        Accuracy as float (0.0 to 1.0)
    """
    if not predictions or not references or len(predictions) != len(references):
        return 0.0
    
    correct = sum(
        1 for pred, ref in zip(predictions, references)
        if str(pred).strip().upper() == str(ref).strip().upper()
    )
    return correct / len(predictions)


def compute_rouge_scores(
    predictions: List[str],
    references: List[str],
    rouge_types: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute ROUGE scores for text predictions.
    
    Args:
        predictions: List of predicted text
        references: List of reference text
        rouge_types: ROUGE types to compute (default: all n-grams + L/Lsum)
    
    Returns:
        Dict with average ROUGE scores
        
    Note:
        rouge-score library supports: rouge1-9, rougeL, rougeLsum
        ROUGE-W, ROUGE-S, ROUGE-SU require the original Perl toolkit
    """
    if not ROUGE_AVAILABLE:
        return {"error": "rouge-score not installed (pip install rouge-score)"}
    
    if not predictions or not references or len(predictions) != len(references):
        return {}
    
    if rouge_types is None:
        # Extended types: unigrams, bigrams, trigrams, 4-grams + LCS variants
        rouge_types = ["rouge1", "rouge2", "rouge3", "rouge4", "rougeL", "rougeLsum"]
    
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    
    scores_by_type: Dict[str, List[float]] = {rt: [] for rt in rouge_types}
    
    for pred, ref in zip(predictions, references):
        pred_text = str(pred).strip()
        ref_text = str(ref).strip()
        if not pred_text or not ref_text:
            continue
        scores = scorer.score(ref_text, pred_text)
        for rt in rouge_types:
            scores_by_type[rt].append(scores[rt].fmeasure)
    
    avg_scores = {}
    for rt, vals in scores_by_type.items():
        if vals:
            avg_scores[f"{rt}_f1"] = sum(vals) / len(vals)
    
    return avg_scores


def compute_bleu_score(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute average sentence-level BLEU score.
    
    Args:
        predictions: List of predicted text
        references: List of reference text
    
    Returns:
        Average BLEU score (0.0 to 1.0)
    """
    if not BLEU_AVAILABLE:
        return 0.0
    
    if not predictions or not references or len(predictions) != len(references):
        return 0.0
    
    smoothing = SmoothingFunction().method1
    bleu_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_text = str(pred).strip()
        ref_text = str(ref).strip()
        if not pred_text or not ref_text:
            continue
        pred_tokens = pred_text.split()
        ref_tokens = [ref_text.split()]
        try:
            score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
            bleu_scores.append(score)
        except Exception:
            continue
    
    if not bleu_scores:
        return 0.0
    return sum(bleu_scores) / len(bleu_scores)


def compute_per_question_accuracy(
    predictions: List[Dict[str, str]],
    references: List[Dict[str, str]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute accuracy per question type.
    
    Args:
        predictions: List of dicts with {"question_id": "...", "answer": "..."}
        references: List of dicts with {"question_id": "...", "answer": "..."}
    
    Returns:
        Dict mapping question_id to {"accuracy": ..., "count": ...}
    """
    if not predictions or not references or len(predictions) != len(references):
        return {}
    
    by_question: Dict[str, Dict[str, List[bool]]] = {}
    
    for pred, ref in zip(predictions, references):
        qid = pred.get("question_id", "unknown")
        pred_ans = str(pred.get("answer", "")).strip().upper()
        ref_ans = str(ref.get("answer", "")).strip().upper()
        
        if qid not in by_question:
            by_question[qid] = {"matches": []}
        by_question[qid]["matches"].append(pred_ans == ref_ans)
    
    results = {}
    for qid, data in by_question.items():
        matches = data["matches"]
        if matches:
            results[qid] = {
                "accuracy": sum(matches) / len(matches),
                "correct": sum(matches),
                "total": len(matches),
            }
    
    return results


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    lang: str = "en",
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute BERTScore for text predictions.
    
    Args:
        predictions: List of predicted text
        references: List of reference text
        lang: Language code (default: "en")
        device: Device for computation (None = auto-detect)
    
    Returns:
        Dict with average BERTScore metrics (precision, recall, f1)
    """
    if not BERTSCORE_AVAILABLE:
        return {"error": "bert-score not installed (pip install bert-score)"}
    
    if not predictions or not references or len(predictions) != len(references):
        return {}
    
    # Filter empty strings
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
    if not valid_pairs:
        return {}
    
    valid_preds, valid_refs = zip(*valid_pairs)
    
    try:
        P, R, F1 = bert_score.score(
            list(valid_preds),
            list(valid_refs),
            lang=lang,
            device=device,
            verbose=False,
        )
        
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
        }
    except Exception as e:
        return {"error": f"BERTScore computation failed: {e}"}


def compute_clip_score_text(
    predictions: List[str],
    references: List[str],
    model=None,
    device: Optional[str] = None,
) -> float:
    """
    Compute CLIP score for text-text similarity (answer vs answer).
    Uses CLIP's text encoder to compare predicted and reference answers.
    
    Args:
        predictions: List of predicted answer texts
        references: List of reference answer texts
        model: Pre-loaded CLIP model (optional, loaded if None)
        device: Device for computation (None = auto-detect)
    
    Returns:
        Average cosine similarity (0.0 to 1.0)
    """
    if not CLIP_AVAILABLE:
        return 0.0
    
    if not predictions or not references or len(predictions) != len(references):
        return 0.0
    
    # Filter empty strings
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
    if not valid_pairs:
        return 0.0
    
    valid_preds, valid_refs = zip(*valid_pairs)
    
    try:
        if model is None:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _ = clip.load("ViT-B/32", device=device)
        
        # Tokenize and encode
        pred_tokens = clip.tokenize(list(valid_preds), truncate=True).to(model.visual.conv1.weight.device)
        ref_tokens = clip.tokenize(list(valid_refs), truncate=True).to(model.visual.conv1.weight.device)
        
        with torch.no_grad():
            pred_features = model.encode_text(pred_tokens)
            ref_features = model.encode_text(ref_tokens)
            
            # Normalize
            pred_features = pred_features / pred_features.norm(dim=-1, keepdim=True)
            ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True)
            
            # Cosine similarity per pair
            similarities = (pred_features * ref_features).sum(dim=-1)
            
            return similarities.mean().item()
    
    except Exception as e:
        import logging
        logging.warning(f"CLIP score computation failed: {e}")
        return 0.0


def compute_clip_score_image_text(
    image_paths: List[str],
    predictions: List[str],
    references: Optional[List[str]] = None,
    model=None,
    preprocess=None,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute CLIP score for image-text alignment (image vs predicted answer).
    Measures how well the predicted answer matches the image.
    
    Args:
        image_paths: List of image file paths
        predictions: List of predicted answer texts
        references: Optional list of reference texts (to compare both)
        model: Pre-loaded CLIP model (optional)
        preprocess: CLIP preprocess transform (optional)
        device: Device for computation (None = auto-detect)
    
    Returns:
        Dict with average scores {"pred_image_score": ..., "ref_image_score": ...}
    """
    if not CLIP_AVAILABLE:
        return {"error": "CLIP not available"}
    
    if not image_paths or not predictions or len(image_paths) != len(predictions):
        return {}
    
    try:
        if model is None or preprocess is None:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)
        
        device = model.visual.conv1.weight.device
        
        pred_scores = []
        ref_scores = [] if references else None
        
        for i, img_path in enumerate(image_paths):
            if not predictions[i].strip():
                continue
            
            try:
                # Load and preprocess image
                image = Image.open(img_path).convert("RGB")
                image_input = preprocess(image).unsqueeze(0).to(device)
                
                # Encode image
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # Encode predicted text
                    pred_tokens = clip.tokenize([predictions[i]], truncate=True).to(device)
                    pred_features = model.encode_text(pred_tokens)
                    pred_features = pred_features / pred_features.norm(dim=-1, keepdim=True)
                    
                    # Cosine similarity
                    pred_score = (image_features * pred_features).sum().item()
                    pred_scores.append(pred_score)
                    
                    # Reference text if provided
                    if references and i < len(references) and references[i].strip():
                        ref_tokens = clip.tokenize([references[i]], truncate=True).to(device)
                        ref_features = model.encode_text(ref_tokens)
                        ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True)
                        ref_score = (image_features * ref_features).sum().item()
                        ref_scores.append(ref_score)
            
            except Exception as e:
                continue
        
        result = {}
        if pred_scores:
            result["clip_image_pred_score"] = sum(pred_scores) / len(pred_scores)
        if ref_scores:
            result["clip_image_ref_score"] = sum(ref_scores) / len(ref_scores)
        
        return result
    
    except Exception as e:
        return {"error": f"CLIP image-text score failed: {e}"}


def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    prediction_texts: Optional[List[str]] = None,
    reference_texts: Optional[List[str]] = None,
    image_paths: Optional[List[str]] = None,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute all available metrics.
    
    Args:
        predictions: List of predicted labels (A/B/C/D)
        references: List of ground truth labels
        prediction_texts: Optional list of predicted answer texts
        reference_texts: Optional list of reference answer texts
        image_paths: Optional list of image paths (for CLIP image-text score)
        device: Device for BERT/CLIP computation (None = auto-detect)
    
    Returns:
        Dict with all computed metrics
    """
    metrics = {}
    
    # Exact match / accuracy
    metrics["accuracy"] = exact_match_accuracy(predictions, references)
    
    # Text-based metrics (if texts provided)
    if prediction_texts and reference_texts:
        rouge_scores = compute_rouge_scores(prediction_texts, reference_texts)
        metrics.update(rouge_scores)
        
        bleu = compute_bleu_score(prediction_texts, reference_texts)
        metrics["bleu"] = bleu
        
        # BERT Score
        bertscore = compute_bertscore(prediction_texts, reference_texts, device=device)
        metrics.update(bertscore)
        
        # CLIP Score (text-text)
        clip_text_score = compute_clip_score_text(prediction_texts, reference_texts, device=device)
        if clip_text_score > 0:
            metrics["clip_text_score"] = clip_text_score
        
        # CLIP Score (image-text) if images provided
        if image_paths:
            clip_image_scores = compute_clip_score_image_text(
                image_paths, prediction_texts, reference_texts, device=device
            )
            metrics.update(clip_image_scores)
    
    return metrics
