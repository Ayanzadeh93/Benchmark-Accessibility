#!/usr/bin/env python3
"""
Configuration helpers for API keys and settings.

Goals:
- Load environment variables from a local `.env` file (if present)
- Never crash the application if `.env` is missing, corrupted, or uses a non-UTF8 encoding
- Provide small helper functions to fetch keys from environment variables
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv

    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False


def load_env_file() -> bool:
    """Load environment variables from `.env` if python-dotenv is available.

    Returns:
        True if a `.env` file existed and was loaded successfully, else False.

    Notes:
        Some editors save `.env` as UTF-16 (BOM) on Windows; python-dotenv defaults to UTF-8.
        We try a small set of encodings and fail gracefully.
    """
    if not _DOTENV_AVAILABLE:
        return False

    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return False

    for enc in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
        try:
            load_dotenv(env_path, encoding=enc)
            return True
        except UnicodeDecodeError:
            continue
        except Exception as e:
            # Any other dotenv parsing error: log and stop trying.
            logger.warning(f"Failed to load .env file ({env_path}) with encoding={enc}: {e}")
            return False

    logger.warning(f"Failed to decode .env file ({env_path}) with supported encodings; skipping.")
    return False


def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key, prioritizing .env file over environment variables."""
    # First try to load from .env file
    load_env_file()

    # Check if .env file has a key
    env_file_key = None
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('OPENAI_API_KEY='):
                        env_file_key = line.split('=', 1)[1].strip()
                        break
        except Exception:
            pass

    # Get environment variable
    env_var_key = os.getenv("OPENAI_API_KEY")

    # Prioritize .env file key, but warn if they differ
    if env_file_key and env_var_key and env_file_key != env_var_key:
        logger.warning(".env file and environment variable have different OPENAI_API_KEY values. Using .env file value.")
        logger.warning(f"Environment variable key ends with: ...{env_var_key[-10:] if env_var_key else 'None'}")
        logger.warning(f".env file key ends with: ...{env_file_key[-10:] if env_file_key else 'None'}")

    api_key = env_file_key or env_var_key
    if not api_key:
        api_key = os.getenv("OPENAI_KEY") or os.getenv("API_KEY")
    return api_key


def set_openai_api_key(api_key: str) -> None:
    """Set OpenAI API key in environment variable (for current session only)."""
    os.environ["OPENAI_API_KEY"] = api_key


def get_huggingface_token() -> Optional[str]:
    """Get Hugging Face token from environment variables."""
    load_env_file()
    token = (
        os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")  # legacy typo/back-compat
    )
    return token


def set_huggingface_token(token: str) -> None:
    """Set HuggingFace token in environment variable (for current session only)."""
    os.environ["HUGGINGFACE_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    os.environ["HF_TOKEN"] = token


def get_anthropic_api_key() -> Optional[str]:
    """Get Anthropic API key from environment variables."""
    load_env_file()
    return os.getenv("ANTHROPIC_API_KEY")


def get_openrouter_api_key() -> Optional[str]:
    """Get OpenRouter API key from environment variables."""
    load_env_file()
    return os.getenv("OPENROUTER_API_KEY")


# Auto-load on import (safe; will not raise)
load_env_file()

# Convenience variables
OPENAI_API_KEY = get_openai_api_key()
HUGGINGFACE_TOKEN = get_huggingface_token()
ANTHROPIC_API_KEY = get_anthropic_api_key()
OPENROUTER_API_KEY = get_openrouter_api_key()

