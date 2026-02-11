"""
Helper utility functions for the CMI Sales Agent Streamlit app.
"""

import html
import re


def _ensure_model(obj, model_class):
    """Convert a dict to a Pydantic model if needed, or return the object as-is."""
    if isinstance(obj, dict):
        return model_class(**obj)
    return obj


def _rebuild_list(items, model_class):
    """Round-trip a list of Pydantic models through model_dump to ensure valid state."""
    dumped = [item.model_dump() if hasattr(item, 'model_dump') else item for item in items]
    return [_ensure_model(d, model_class) for d in dumped]


def strip_html_tags(text: str) -> str:
    """Remove HTML tags from text for display in non-HTML contexts."""
    if not text:
        return text
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', '', text)
    # Decode common HTML entities
    clean = clean.replace('&nbsp;', ' ')
    clean = clean.replace('&amp;', '&')
    clean = clean.replace('&lt;', '<')
    clean = clean.replace('&gt;', '>')
    return clean.strip()


def escape_for_html(text: str) -> str:
    """Escape text for safe HTML embedding, handling both HTML and markdown patterns."""
    if not text:
        return text
    # First escape HTML special characters
    text = html.escape(text)
    # Also escape backticks which can trigger markdown code blocks
    text = text.replace('`', '&#96;')
    # Escape markdown emphasis patterns
    text = text.replace('**', '&#42;&#42;')
    text = text.replace('__', '&#95;&#95;')
    text = text.replace('\n', ' ').replace('\r', ' ')
    return text


def truncate_text(text: str, max_len: int = 40) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."
