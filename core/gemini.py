"""Gemini AI communication layer.

Handles API key resolution, cached API calls, and browser fallback.
Lazy-imports google.generativeai so the app works without it installed.
"""

import hashlib
import os
import webbrowser

import streamlit as st

# ---------------------------------------------------------------------------
# Lazy import of google-generativeai
# ---------------------------------------------------------------------------

_genai = None  # None = not yet tried, False = tried and failed


def _get_genai():
    """Lazy-load google.generativeai.  Returns the module or None."""
    global _genai
    if _genai is None:
        try:
            import google.generativeai as genai
            _genai = genai
        except ImportError:
            _genai = False
    return _genai if _genai is not False else None


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------

def get_api_key() -> str | None:
    """Resolve the Gemini API key.

    Priority: (1) Streamlit secrets, (2) environment variable,
    (3) session-state value from the sidebar input.
    """
    # 1. Streamlit secrets (.streamlit/secrets.toml)
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return str(key)
    except Exception:
        pass

    # 2. Environment variable
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key

    # 3. Session state (sidebar input)
    return st.session_state.get("gemini_api_key") or None


def is_api_available() -> bool:
    """True when both the API key and the google-generativeai package exist."""
    return get_api_key() is not None and _get_genai() is not None


# ---------------------------------------------------------------------------
# Response caching
# ---------------------------------------------------------------------------

def _cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def _get_cache() -> dict:
    if "_gemini_cache" not in st.session_state:
        st.session_state["_gemini_cache"] = {}
    return st.session_state["_gemini_cache"]


# ---------------------------------------------------------------------------
# Core query function
# ---------------------------------------------------------------------------

def query_gemini(
    prompt: str,
    max_tokens: int = 500,
    use_cache: bool = True,
) -> str | None:
    """Send a prompt to Gemini and return the response text.

    Returns None if the API is unavailable or an error occurs.
    Caches responses in session state to avoid redundant calls on
    Streamlit reruns.
    """
    cache = _get_cache()
    ck = _cache_key(prompt)

    if use_cache and ck in cache:
        return cache[ck]

    genai = _get_genai()
    api_key = get_api_key()
    if not genai or not api_key:
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.3,
            ),
        )
        text = response.text
        if use_cache:
            cache[ck] = text
        return text
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Browser fallback
# ---------------------------------------------------------------------------

def open_gemini_in_browser() -> None:
    """Open gemini.google.com/app in the default browser."""
    webbrowser.open("https://gemini.google.com/app")


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def clear_gemini_cache() -> None:
    """Wipe all cached Gemini responses."""
    st.session_state["_gemini_cache"] = {}
