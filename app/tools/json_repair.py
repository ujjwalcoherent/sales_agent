"""
Standalone JSON repair utilities for LLM output parsing.

Handles common LLM output issues:
- Markdown code block wrapping
- Truncated JSON (unclosed brackets/strings)
- Control characters inside strings
- List extraction from wrapper dict keys
"""

import json
import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)

_RE_MD_CODE_START = re.compile(r'^```(?:json)?\n?')
_RE_MD_CODE_END = re.compile(r'\n?```$')
_RE_CONTROL_CHARS = re.compile(r'[\x00-\x1f\x7f]')


def parse_json_response(response: str) -> Dict[str, Any]:
    """Parse JSON from LLM response, handling common issues."""
    cleaned = response.strip()

    # Remove markdown code blocks if present
    if cleaned.startswith("```"):
        cleaned = _RE_MD_CODE_START.sub('', cleaned)
        cleaned = _RE_MD_CODE_END.sub('', cleaned)
        cleaned = cleaned.strip()

    # Extract the JSON structure (array or object) from the response
    json_str = extract_json_string(cleaned)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Common issue: LLMs (especially Ollama) insert literal newlines/tabs
        # inside JSON string values. Try strict=False which allows control chars.
        try:
            return json.loads(json_str, strict=False)
        except json.JSONDecodeError:
            pass
        # Last resort: strip control characters from inside strings
        sanitized = _RE_CONTROL_CHARS.sub(' ', json_str)
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nResponse: {json_str[:500]}")
            return {"error": "Failed to parse JSON", "raw": json_str[:500]}


def extract_json_string(text: str) -> str:
    """Extract the outermost JSON object or array from text using bracket counting."""
    brace_pos = text.find('{')
    bracket_pos = text.find('[')
    if brace_pos == -1 and bracket_pos == -1:
        return text
    # Whichever appears first in the text is the outermost structure
    pairs = [('{', '}'), ('[', ']')]
    if bracket_pos != -1 and (brace_pos == -1 or bracket_pos < brace_pos):
        pairs = [('[', ']'), ('{', '}')]
    for open_ch, close_ch in pairs:
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        # Brackets never balanced — response is truncated.
        truncated = text[start:]
        return repair_truncated_json(truncated)
    return text


def repair_truncated_json(text: str) -> str:
    """Attempt to repair truncated JSON by closing open brackets/strings."""
    # Pass 1: close any open string
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
    if in_string:
        text = text + '"'

    # Pass 2: track bracket nesting as a stack to preserve close order
    stack = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()

    # Remove trailing comma if present
    text = text.rstrip().rstrip(',')
    # Close brackets in reverse nesting order
    close_map = {'{': '}', '[': ']'}
    for bracket in reversed(stack):
        text += close_map[bracket]
    return text


