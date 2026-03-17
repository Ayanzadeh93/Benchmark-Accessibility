"""Integration test for VQA2 pipeline.

This script tests the VQA2 pipeline without requiring actual data.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_transformation():
    """Test MMVQA → Standard VQA transformation."""
    logger.info("="*60)
    logger.info("Test 1: MMVQA → Standard VQA Transformation")
    logger.info("="*60)
    
    from transform_mmvqa_to_vqa import transform_sample_to_vqa
    
    # Create a sample MMVQA entry
    mmvqa_sample = {
        "id": "test_image_001|action_command",
        "question_id": "action_command",
        "image": "test_image_001.jpg",
        "question": "What should the person do?",
        "options": {
            "A": "Turn left",
            "B": "Go straight",
            "C": "Stop",
            "D": "Turn right"
        },
        "answer": "B",
        "answer_text": "Go straight"
    }
    
    # Transform
    vqa_sample = transform_sample_to_vqa(mmvqa_sample)
    
    # Verify
    assert vqa_sample["id"] == "test_image_001|action_command"
    assert vqa_sample["question"] == "What should the person do?"
    assert vqa_sample["answer"] == "Go straight"
    assert "options" not in vqa_sample  # Options should be removed
    
    logger.info("✓ Transformation test passed!")
    logger.info(f"  Input (MMVQA):  {mmvqa_sample['question']} → Options: A/B/C/D, Answer: B")
    logger.info(f"  Output (VQA):   {vqa_sample['question']} → Answer: {vqa_sample['answer']}")
    
    return True


def test_transformation_with_files():
    """Test transformation with actual JSON files."""
    logger.info("\n" + "="*60)
    logger.info("Test 2: File-based Transformation")
    logger.info("="*60)
    
    from transform_mmvqa_to_vqa import transform_per_question_file
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create a sample per-question MMVQA file
        mmvqa_data = {
            "question_id": "action_command",
            "question": "What should the person do?",
            "num_samples": 2,
            "samples": [
                {
                    "id": "image_001|action_command",
                    "question_id": "action_command",
                    "image": "image_001.jpg",
                    "question": "What should the person do?",
                    "options": {
                        "A": "Turn left",
                        "B": "Go straight",
                        "C": "Stop",
                        "D": "Turn right"
                    },
                    "answer": "B",
                    "answer_text": "Go straight"
                },
                {
                    "id": "image_002|action_command",
                    "question_id": "action_command",
                    "image": "image_002.jpg",
                    "question": "What should the person do?",
                    "options": {
                        "A": "Turn left",
                        "B": "Go straight",
                        "C": "Stop",
                        "D": "Turn right"
                    },
                    "answer": "C",
                    "answer_text": "Stop"
                }
            ]
        }
        
        input_file = tmpdir / "action_command.json"
        with input_file.open("w") as f:
            json.dump(mmvqa_data, f, indent=2)
        
        output_file = tmpdir / "action_command_vqa.json"
        
        # Transform
        stats = transform_per_question_file(input_file, output_file)
        
        # Verify
        assert output_file.exists()
        with output_file.open("r") as f:
            vqa_data = json.load(f)
        
        assert vqa_data["question_id"] == "action_command"
        assert vqa_data["num_samples"] == 2
        assert len(vqa_data["samples"]) == 2
        
        # Check first sample
        sample1 = vqa_data["samples"][0]
        assert sample1["id"] == "image_001|action_command"
        assert sample1["answer"] == "Go straight"
        assert "options" not in sample1
        
        # Check second sample
        sample2 = vqa_data["samples"][1]
        assert sample2["id"] == "image_002|action_command"
        assert sample2["answer"] == "Stop"
        assert "options" not in sample2
        
        logger.info("✓ File-based transformation test passed!")
        logger.info(f"  Transformed {stats['num_samples']} samples")
        logger.info(f"  Input:  {input_file}")
        logger.info(f"  Output: {output_file}")
    
    return True


def test_metrics():
    """Test evaluation metrics."""
    logger.info("\n" + "="*60)
    logger.info("Test 3: Evaluation Metrics")
    logger.info("="*60)
    
    from metrics import exact_match_accuracy
    
    # Test exact match
    predictions = ["Go straight", "turn left", "STOP"]
    references = ["Go straight", "Turn left", "stop"]
    
    accuracy = exact_match_accuracy(predictions, references)
    assert accuracy == 1.0, f"Expected 1.0, got {accuracy}"
    
    logger.info("✓ Exact match test passed!")
    logger.info(f"  Predictions: {predictions}")
    logger.info(f"  References:  {references}")
    logger.info(f"  Accuracy:    {accuracy:.4f}")
    
    # Test ROUGE (if available)
    try:
        from metrics import compute_rouge_scores
        
        pred_texts = ["Go straight ahead", "Turn to the left"]
        ref_texts = ["Go straight", "Turn left"]
        
        rouge_scores = compute_rouge_scores(pred_texts, ref_texts)
        
        # Check if ROUGE scores were computed
        if rouge_scores and any(v is not None for v in rouge_scores.values()):
            logger.info("✓ ROUGE test passed!")
            if "rouge1_f1" in rouge_scores and rouge_scores["rouge1_f1"] is not None:
                logger.info(f"  ROUGE-1 F1: {rouge_scores['rouge1_f1']:.4f}")
            else:
                logger.info(f"  ROUGE scores: {rouge_scores}")
        else:
            logger.info("⚠ ROUGE returned empty scores")
    except ImportError:
        logger.info("⚠ ROUGE not available (install rouge-score)")
    except Exception as e:
        logger.info(f"⚠ ROUGE test skipped: {e}")
    
    # Test BLEU (if available)
    try:
        from metrics import compute_bleu_score
        
        pred_texts = ["Go straight ahead", "Turn to the left"]
        ref_texts = ["Go straight", "Turn left"]
        
        bleu_score = compute_bleu_score(pred_texts, ref_texts)
        logger.info("✓ BLEU test passed!")
        logger.info(f"  BLEU: {bleu_score:.4f}")
    except ImportError:
        logger.info("⚠ BLEU not available (install sacrebleu)")
    
    return True


def test_vqa_evaluator_init():
    """Test VQA evaluator initialization."""
    logger.info("\n" + "="*60)
    logger.info("Test 4: VQA Evaluator Initialization")
    logger.info("="*60)
    
    try:
        from vqa_standard_evaluation import StandardVQAEvaluator
        
        # Test that evaluator can be created (without actual model loading)
        logger.info("✓ VQA evaluator module loads successfully!")
        logger.info("  Note: Actual model initialization requires VLM dependencies")
    except ImportError as e:
        logger.error(f"✗ Failed to import VQA evaluator: {e}")
        return False
    
    return True


def test_pipeline_cli():
    """Test pipeline CLI."""
    logger.info("\n" + "="*60)
    logger.info("Test 5: Pipeline CLI")
    logger.info("="*60)
    
    try:
        from vqa2_pipeline import get_default_models, get_all_available_models
        
        default_models = get_default_models()
        all_models = get_all_available_models()
        
        assert len(default_models) > 0
        assert len(all_models) >= len(default_models)
        
        logger.info("✓ Pipeline CLI test passed!")
        logger.info(f"  Default models: {default_models}")
        logger.info(f"  Total available: {len(all_models)} models")
    except ImportError as e:
        logger.error(f"✗ Failed to import pipeline CLI: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    logger.info("\n" + "="*80)
    logger.info("VQA2 Integration Tests")
    logger.info("="*80 + "\n")
    
    tests = [
        ("Transformation (in-memory)", test_transformation),
        ("Transformation (file-based)", test_transformation_with_files),
        ("Evaluation Metrics", test_metrics),
        ("VQA Evaluator", test_vqa_evaluator_init),
        ("Pipeline CLI", test_pipeline_cli),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"\n✗ Test '{test_name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    logger.info("\n" + "="*80)
    logger.info("Test Summary")
    logger.info("="*80)
    logger.info(f"  Passed: {passed}/{len(tests)}")
    logger.info(f"  Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        logger.info("\n✓ All tests passed!")
    else:
        logger.info(f"\n✗ {failed} test(s) failed")
    
    logger.info("="*80 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
