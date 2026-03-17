#!/usr/bin/env python3
"""Quick test script for Florence-2 on RTX 5070"""

import sys
from pathlib import Path

def test_florence2():
    """Test Florence-2 speed and accuracy"""
    
    # Find a test image
    test_images = []
    for ext in [".jpg", ".jpeg", ".png"]:
        test_images.extend(Path(".").rglob(f"*{ext}"))
    
    if not test_images and len(sys.argv) < 2:
        print("❌ No test image found. Usage: python test_florence2.py <image_path>")
        return
    
    image_path = sys.argv[1] if len(sys.argv) > 1 else str(test_images[0])
    
    print("\n" + "="*70)
    print("Florence-2 Speed Test on RTX 5070")
    print("="*70)
    print(f"Test image: {image_path}")
    
    # Test Florence-2
    print("\n[1/3] Loading Florence-2...")
    from vlm_florence2 import Florence2Extractor
    import time
    
    start = time.time()
    florence = Florence2Extractor(device="cuda", model_size="base")
    load_time = time.time() - start
    
    if not florence.enabled:
        print("❌ Florence-2 failed to load")
        return
    
    print(f"✅ Loaded in {load_time:.2f}s")
    
    # Test extraction speed
    print("\n[2/3] Testing object extraction (5 runs)...")
    times = []
    for i in range(5):
        start = time.time()
        result = florence.extract_objects_from_keyframe(image_path)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s ({result['num_objects']} objects)")
    
    avg_time = sum(times) / len(times)
    print(f"\n✅ Average: {avg_time:.3f}s per image")
    
    # Show results
    print("\n[3/3] Sample extraction result:")
    print(f"  Objects found: {result['num_objects']}")
    print(f"  Scene: {result['scene_description'][:100]}...")
    print("\n  Top objects:")
    for obj in result['objects'][:10]:
        print(f"    - {obj}")
    
    # Compare to target
    print("\n" + "="*70)
    if avg_time <= 2.0:
        print(f"🎉 SUCCESS! {avg_time:.3f}s is UNDER 2 seconds target!")
    elif avg_time <= 3.0:
        print(f"✅ GOOD! {avg_time:.3f}s is under 3 seconds target")
    else:
        print(f"⚠️  {avg_time:.3f}s exceeds 3 second target (may need optimization)")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_florence2()
