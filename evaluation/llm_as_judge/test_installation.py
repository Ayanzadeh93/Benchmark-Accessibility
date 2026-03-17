"""Test script to verify LLM-as-judge installation and setup."""

import sys
from pathlib import Path

# Windows console encoding fix
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, 'strict')


def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    
    try:
        import pandas
        print("  ✅ pandas")
    except ImportError:
        print("  ❌ pandas - Run: pip install pandas")
        return False
    
    try:
        import openpyxl
        print("  ✅ openpyxl")
    except ImportError:
        print("  ❌ openpyxl - Run: pip install openpyxl")
        return False
    
    try:
        import matplotlib
        print("  ✅ matplotlib")
    except ImportError:
        print("  ❌ matplotlib - Run: pip install matplotlib")
        return False
    
    try:
        import seaborn
        print("  ✅ seaborn")
    except ImportError:
        print("  ❌ seaborn - Run: pip install seaborn")
        return False
    
    try:
        import tqdm
        print("  ✅ tqdm")
    except ImportError:
        print("  ❌ tqdm - Run: pip install tqdm")
        return False
    
    try:
        import pydantic
        print("  ✅ pydantic")
    except ImportError:
        print("  ❌ pydantic - Run: pip install pydantic")
        return False
    
    return True


def test_module_structure():
    """Test that all LLM-as-judge modules are present."""
    print("\nTesting module structure...")
    
    base_path = Path(__file__).parent
    
    required_files = [
        "__init__.py",
        "prompts.py",
        "schemas.py",
        "judge_models.py",
        "visualization.py",
        "pipeline.py",
        "cli.py",
        "README.md",
        "QUICKSTART.md",
    ]
    
    all_present = True
    for filename in required_files:
        filepath = base_path / filename
        if filepath.exists():
            print(f"  ✅ {filename}")
        else:
            print(f"  ❌ {filename} - Missing!")
            all_present = False
    
    return all_present


def test_local_imports():
    """Test that local modules can be imported."""
    print("\nTesting local imports...")
    
    import sys
    from pathlib import Path
    
    # Add parent directories to path for import
    eval_dir = Path(__file__).parent.parent.parent
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    
    try:
        from evaluation.llm_as_judge import schemas
        print("  ✅ schemas")
    except ImportError as e:
        print(f"  ❌ schemas - {e}")
        return False
    
    try:
        from evaluation.llm_as_judge import prompts
        print("  ✅ prompts")
    except ImportError as e:
        print(f"  ❌ prompts - {e}")
        return False
    
    try:
        from evaluation.llm_as_judge import judge_models
        print("  ✅ judge_models")
    except ImportError as e:
        print(f"  ❌ judge_models - {e}")
        return False
    
    try:
        from evaluation.llm_as_judge import visualization
        print("  ✅ visualization")
    except ImportError as e:
        print(f"  ❌ visualization - {e}")
        return False
    
    try:
        from evaluation.llm_as_judge import pipeline
        print("  ✅ pipeline")
    except ImportError as e:
        print(f"  ❌ pipeline - {e}")
        return False
    
    return True


def test_api_clients():
    """Test that API client libraries are available."""
    print("\nTesting API client libraries...")
    
    openai_available = True
    anthropic_available = True
    
    try:
        import openai
        print("  ✅ openai")
    except ImportError:
        print("  ⚠️  openai - Run: pip install openai (needed for OpenAI/OpenRouter models)")
        openai_available = False
    
    try:
        import anthropic
        print("  ✅ anthropic")
    except ImportError:
        print("  ⚠️  anthropic - Run: pip install anthropic (needed for Claude models)")
        anthropic_available = False
    
    return openai_available or anthropic_available


def test_api_keys():
    """Check if API keys are configured."""
    print("\nChecking API keys...")
    
    import os
    
    keys_found = []
    keys_missing = []
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("  ✅ OPENAI_API_KEY found")
        keys_found.append("OPENAI_API_KEY")
    else:
        print("  ⚠️  OPENAI_API_KEY not set (needed for GPT models)")
        keys_missing.append("OPENAI_API_KEY")
    
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        print("  ✅ ANTHROPIC_API_KEY found")
        keys_found.append("ANTHROPIC_API_KEY")
    else:
        print("  ⚠️  ANTHROPIC_API_KEY not set (needed for Claude models)")
        keys_missing.append("ANTHROPIC_API_KEY")
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        print("  ✅ OPENROUTER_API_KEY found")
        keys_found.append("OPENROUTER_API_KEY")
    else:
        print("  ⚠️  OPENROUTER_API_KEY not set (needed for OpenRouter models)")
        keys_missing.append("OPENROUTER_API_KEY")
    
    if not keys_found:
        print("\n  ⚠️  No API keys found. You'll need at least one to run evaluations.")
        print("     Set keys in .env file or environment variables:")
        print("       export OPENAI_API_KEY='sk-...'")
        print("       export ANTHROPIC_API_KEY='sk-ant-...'")
        print("       export OPENROUTER_API_KEY='sk-or-...'")
    
    return len(keys_found) > 0


def main():
    """Run all tests."""
    print("="*60)
    print("LLM-as-Judge Installation Test")
    print("="*60)
    
    results = {
        "imports": test_imports(),
        "structure": test_module_structure(),
        "local_imports": test_local_imports(),
        "api_clients": test_api_clients(),
        "api_keys": test_api_keys(),
    }
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():20s}: {status}")
    
    all_passed = all(results.values())
    
    print("="*60)
    
    if all_passed:
        print("\n✅ All tests passed! LLM-as-judge is ready to use.")
        print("\nQuick start:")
        print("  python main.py eval --help")
        print("\nOr see: evaluation/llm_as_judge/QUICKSTART.md")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please install missing dependencies:")
        print("  pip install openai anthropic pandas openpyxl matplotlib seaborn")
        print("\nThen set up API keys:")
        print("  export OPENAI_API_KEY='your-key'")
        return 1


if __name__ == "__main__":
    sys.exit(main())
