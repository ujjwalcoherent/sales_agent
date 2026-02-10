"""
Comprehensive LLM Provider Test Suite for Sales Agent
=====================================================
Tests all 5 LLM providers: NVIDIA, Ollama, OpenRouter, Gemini, Groq

For each provider tests:
  - Connectivity (simple prompt)
  - Speed: simple JSON prompt
  - Speed: longer analysis prompt (~500 token output)
  - Output quality: valid JSON generation
  - Edge cases: empty prompt, very long prompt, Hindi prompt
  - Timeout behavior

Usage:
    python test_llm_providers.py
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure the project root is on sys.path so relative imports inside app work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress noisy logs during testing -- we capture what we need ourselves
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("test_llm_providers")
logger.setLevel(logging.INFO)

# ─── Import LLMTool (the class under test) ───────────────────────────────────
from app.tools.llm_tool import LLMTool

# ─── Constants ────────────────────────────────────────────────────────────────

SIMPLE_PROMPT = 'Respond with exactly this JSON: {"status": "ok", "provider": "<your name>"}'

JSON_PROMPT = """Analyze the following Indian business news and return a JSON object with exactly these fields:
- "trend_title": a short title (max 10 words)
- "severity": one of "high", "medium", "low"
- "industries_affected": a list of 3 industries
- "summary": a 2-sentence summary

News: "Reliance Industries announced a $5 billion investment in green hydrogen production facilities across Gujarat and Rajasthan, partnering with European technology providers. This is expected to create 50,000 jobs and position India as a global green hydrogen hub by 2030."

Respond ONLY with valid JSON. No markdown, no explanation."""

LONG_ANALYSIS_PROMPT = """You are a senior business analyst. Write a detailed analysis (at least 400 words) of the following scenario:

India's semiconductor manufacturing ecosystem is rapidly evolving. The government has approved Rs 1.26 lakh crore for three new fabrication plants under the India Semiconductor Mission. Tata Electronics and Vedanta are the primary beneficiaries.

Your analysis must cover:
1. Short-term impact (next 6 months) on the electronics supply chain
2. Medium-term opportunities (1-2 years) for component manufacturers
3. Long-term strategic implications (3-5 years) for India's position in global semiconductor trade
4. Key risks and mitigation strategies
5. Recommended actions for mid-size electronics companies

Be specific with numbers, company names, and actionable recommendations.
Respond in plain text (not JSON)."""

HINDI_PROMPT = """निम्नलिखित भारतीय व्यापार समाचार का विश्लेषण करें और JSON में उत्तर दें:
"भारतीय रिज़र्व बैंक ने डिजिटल ऋणदाताओं के लिए नए KYC नियम घोषित किए हैं।"

JSON format:
{"title": "...", "sector": "...", "impact": "high/medium/low"}

केवल JSON में उत्तर दें।"""

VERY_LONG_PROMPT = ("Repeat the word 'test' and count from 1 to 500. " * 40) + '\nNow respond with: {"done": true}'

EMPTY_PROMPT = ""

SYSTEM_PROMPT = "You are a helpful JSON-only assistant. Always respond with valid JSON."


# ─── Result collection ────────────────────────────────────────────────────────

class TestResult:
    """Holds one test result."""

    def __init__(self, provider: str, test_name: str):
        self.provider = provider
        self.test_name = test_name
        self.success: bool = False
        self.duration_s: float = 0.0
        self.response_length: int = 0
        self.valid_json: Optional[bool] = None  # None means not applicable
        self.error: str = ""
        self.snippet: str = ""  # first 120 chars of response

    @staticmethod
    def _ascii_safe(text: str) -> str:
        """Replace non-ASCII characters so console printing never fails."""
        return text.encode("ascii", errors="replace").decode("ascii")

    def as_row(self) -> Dict[str, Any]:
        status = "PASS" if self.success else "FAIL"
        json_col = ""
        if self.valid_json is True:
            json_col = "valid"
        elif self.valid_json is False:
            json_col = "INVALID"
        elif self.valid_json is None:
            json_col = "n/a"
        return {
            "Provider": self.provider,
            "Test": self.test_name,
            "Status": status,
            "Time(s)": f"{self.duration_s:.2f}",
            "Chars": self.response_length,
            "JSON": json_col,
            "Error/Snippet": self._ascii_safe(self.error if self.error else self.snippet),
        }


ALL_RESULTS: List[TestResult] = []


# ─── Helper: call a provider method directly ─────────────────────────────────

async def call_provider(
    llm: LLMTool,
    provider_name: str,
    method_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    timeout: float = 60.0,
) -> Tuple[Optional[str], float, Optional[str]]:
    """
    Call a specific provider method with a timeout.
    Returns (response_text, duration_seconds, error_string_or_None).
    """
    method = getattr(llm, method_name, None)
    if method is None:
        return None, 0.0, f"Method {method_name} not found on LLMTool"

    start = time.perf_counter()
    try:
        response = await asyncio.wait_for(
            method(prompt, system_prompt, temperature, max_tokens),
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        return response, elapsed, None
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - start
        return None, elapsed, f"Timed out after {timeout:.0f}s"
    except Exception as e:
        elapsed = time.perf_counter() - start
        return None, elapsed, f"{type(e).__name__}: {str(e)[:200]}"


def is_valid_json(text: str) -> Tuple[bool, Optional[dict]]:
    """Try to parse JSON from text (allows markdown fencing)."""
    if not text:
        return False, None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        import re
        cleaned = re.sub(r'^```(?:json)?\n?', '', cleaned)
        cleaned = re.sub(r'\n?```$', '', cleaned)
        cleaned = cleaned.strip()
    try:
        obj = json.loads(cleaned)
        return True, obj
    except json.JSONDecodeError:
        # Try lenient
        try:
            obj = json.loads(cleaned, strict=False)
            return True, obj
        except json.JSONDecodeError:
            return False, None


# ─── Individual test functions ────────────────────────────────────────────────

async def test_connectivity(llm: LLMTool, provider: str, method: str) -> TestResult:
    """Test 1: Can it connect and return anything?"""
    r = TestResult(provider, "connectivity")
    resp, dur, err = await call_provider(llm, provider, method, SIMPLE_PROMPT, SYSTEM_PROMPT, timeout=45)
    r.duration_s = dur
    if err:
        r.error = err
    elif resp:
        r.success = True
        r.response_length = len(resp)
        r.snippet = resp[:120].replace("\n", " ")
    else:
        r.error = "Empty response"
    ALL_RESULTS.append(r)
    return r


async def test_json_speed(llm: LLMTool, provider: str, method: str) -> TestResult:
    """Test 2: Speed for a simple JSON prompt."""
    r = TestResult(provider, "json_speed")
    resp, dur, err = await call_provider(llm, provider, method, JSON_PROMPT, SYSTEM_PROMPT, max_tokens=512, timeout=60)
    r.duration_s = dur
    if err:
        r.error = err
    elif resp:
        r.success = True
        r.response_length = len(resp)
        valid, _ = is_valid_json(resp)
        r.valid_json = valid
        r.snippet = resp[:120].replace("\n", " ")
    else:
        r.error = "Empty response"
    ALL_RESULTS.append(r)
    return r


async def test_long_analysis(llm: LLMTool, provider: str, method: str) -> TestResult:
    """Test 3: Speed for a longer analysis (~500 token output)."""
    r = TestResult(provider, "long_analysis")
    resp, dur, err = await call_provider(llm, provider, method, LONG_ANALYSIS_PROMPT, None, max_tokens=2048, timeout=120)
    r.duration_s = dur
    if err:
        r.error = err
    elif resp:
        r.success = True
        r.response_length = len(resp)
        r.valid_json = None  # plain text expected
        r.snippet = resp[:120].replace("\n", " ")
    else:
        r.error = "Empty response"
    ALL_RESULTS.append(r)
    return r


async def test_json_quality(llm: LLMTool, provider: str, method: str) -> TestResult:
    """Test 4: Does it return well-formed JSON with the requested keys?"""
    r = TestResult(provider, "json_quality")
    resp, dur, err = await call_provider(llm, provider, method, JSON_PROMPT, SYSTEM_PROMPT, max_tokens=512, timeout=60)
    r.duration_s = dur
    if err:
        r.error = err
    elif resp:
        valid, obj = is_valid_json(resp)
        r.valid_json = valid
        if valid and isinstance(obj, dict):
            expected_keys = {"trend_title", "severity", "industries_affected", "summary"}
            present = expected_keys.intersection(obj.keys())
            r.success = len(present) >= 3  # at least 3 of 4 expected keys
            r.snippet = f"keys={list(obj.keys())}"
            if not r.success:
                r.error = f"Missing keys. Got: {list(obj.keys())}"
        elif valid:
            r.success = True
            r.snippet = f"Valid JSON (type={type(obj).__name__})"
        else:
            r.error = f"Invalid JSON: {resp[:100]}"
        r.response_length = len(resp)
    else:
        r.error = "Empty response"
    ALL_RESULTS.append(r)
    return r


async def test_empty_prompt(llm: LLMTool, provider: str, method: str) -> TestResult:
    """Test 5 (edge): What happens with an empty prompt?"""
    r = TestResult(provider, "empty_prompt")
    resp, dur, err = await call_provider(llm, provider, method, EMPTY_PROMPT, None, timeout=30)
    r.duration_s = dur
    if err:
        # An error is actually the *expected* graceful behavior
        r.success = True
        r.error = f"Graceful error: {err[:100]}"
    elif resp:
        # Some providers may still respond
        r.success = True
        r.response_length = len(resp)
        r.snippet = f"Responded anyway ({len(resp)} chars): {resp[:80]}"
    else:
        r.success = True
        r.error = "Empty response (acceptable for empty prompt)"
    ALL_RESULTS.append(r)
    return r


async def test_long_prompt(llm: LLMTool, provider: str, method: str) -> TestResult:
    """Test 6 (edge): Very long prompt (~20k chars)."""
    r = TestResult(provider, "long_prompt")
    resp, dur, err = await call_provider(llm, provider, method, VERY_LONG_PROMPT, None, max_tokens=256, timeout=90)
    r.duration_s = dur
    if err:
        # Could be a context-length error, that's informative
        r.success = "timed out" not in err.lower()  # timeout = bad, error = informative
        r.error = err[:150]
    elif resp:
        r.success = True
        r.response_length = len(resp)
        r.snippet = resp[:120].replace("\n", " ")
    else:
        r.error = "Empty response"
    ALL_RESULTS.append(r)
    return r


async def test_hindi_prompt(llm: LLMTool, provider: str, method: str) -> TestResult:
    """Test 7 (edge): Non-English (Hindi) prompt requesting JSON."""
    r = TestResult(provider, "hindi_prompt")
    resp, dur, err = await call_provider(llm, provider, method, HINDI_PROMPT, None, max_tokens=512, timeout=60)
    r.duration_s = dur
    if err:
        r.error = err[:150]
    elif resp:
        r.success = True
        r.response_length = len(resp)
        valid, _ = is_valid_json(resp)
        r.valid_json = valid
        r.snippet = resp[:120].replace("\n", " ")
    else:
        r.error = "Empty response"
    ALL_RESULTS.append(r)
    return r


async def test_timeout_behavior(llm: LLMTool, provider: str, method: str) -> TestResult:
    """Test 8: Does it respect a very short timeout (3s)?"""
    r = TestResult(provider, "timeout_3s")
    # Use the long analysis prompt with a 3-second timeout to force a timeout
    resp, dur, err = await call_provider(
        llm, provider, method, LONG_ANALYSIS_PROMPT, None, max_tokens=4096, timeout=3.0
    )
    r.duration_s = dur
    if err and "timed out" in err.lower():
        r.success = True  # Timeout is the *expected* behavior here
        r.error = f"Correctly timed out in {dur:.1f}s"
    elif err:
        r.success = True  # Any fast error is fine
        r.error = f"Fast error ({dur:.1f}s): {err[:80]}"
    elif resp:
        # If it responded in under 3s, that's impressively fast
        r.success = True
        r.response_length = len(resp)
        r.snippet = f"Responded in {dur:.1f}s! ({len(resp)} chars)"
    else:
        r.error = "Empty response"
    ALL_RESULTS.append(r)
    return r


# ─── Provider test orchestrator ──────────────────────────────────────────────

async def test_single_provider(
    llm: LLMTool,
    provider_display: str,
    method_name: str,
    is_configured: bool,
    model_name: str,
):
    """Run all tests for one provider."""
    print(f"\n{'='*70}")
    print(f"  TESTING: {provider_display}")
    print(f"  Model:   {model_name}")
    print(f"  Method:  LLMTool.{method_name}")
    print(f"  Status:  {'CONFIGURED' if is_configured else 'NOT CONFIGURED (skipping)'}")
    print(f"{'='*70}")

    if not is_configured:
        # Record skip results for all tests
        for test_name in [
            "connectivity", "json_speed", "long_analysis", "json_quality",
            "empty_prompt", "long_prompt", "hindi_prompt", "timeout_3s"
        ]:
            r = TestResult(provider_display, test_name)
            r.error = "Provider not configured"
            ALL_RESULTS.append(r)
        return

    tests = [
        ("1. Connectivity", test_connectivity),
        ("2. JSON Speed", test_json_speed),
        ("3. Long Analysis", test_long_analysis),
        ("4. JSON Quality", test_json_quality),
        ("5. Empty Prompt", test_empty_prompt),
        ("6. Long Prompt (~20k chars)", test_long_prompt),
        ("7. Hindi Prompt", test_hindi_prompt),
        ("8. Timeout Behavior (3s limit)", test_timeout_behavior),
    ]

    for label, test_func in tests:
        print(f"\n  [{label}] ...", end=" ", flush=True)
        result = await test_func(llm, provider_display, method_name)
        status = "PASS" if result.success else "FAIL"
        extra = ""
        if result.duration_s > 0:
            extra += f" ({result.duration_s:.2f}s)"
        if result.response_length > 0:
            extra += f" [{result.response_length} chars]"
        if result.valid_json is True:
            extra += " [valid JSON]"
        elif result.valid_json is False:
            extra += " [INVALID JSON]"
        safe_extra = extra.encode("ascii", errors="replace").decode("ascii")
        print(f"{status}{safe_extra}")
        if result.error:
            # Truncate long errors for console, make ASCII-safe
            err_display = result.error[:120].encode("ascii", errors="replace").decode("ascii")
            print(f"     -> {err_display}")


# ─── Pretty table printer ────────────────────────────────────────────────────

def print_results_table():
    """Print all results as a formatted table."""
    print("\n\n")
    print("=" * 130)
    print("  COMPREHENSIVE LLM PROVIDER TEST RESULTS")
    print("=" * 130)

    # Column widths
    cols = {
        "Provider": 14,
        "Test": 16,
        "Status": 6,
        "Time(s)": 8,
        "Chars": 6,
        "JSON": 8,
        "Error/Snippet": 60,
    }

    # Header
    header = " | ".join(name.ljust(width) for name, width in cols.items())
    print(header)
    print("-" * len(header))

    current_provider = ""
    for result in ALL_RESULTS:
        row = result.as_row()
        # Print a separator between providers
        if row["Provider"] != current_provider:
            if current_provider:
                print("-" * len(header))
            current_provider = row["Provider"]

        line_parts = []
        for col_name, width in cols.items():
            val = str(row.get(col_name, ""))
            if len(val) > width:
                val = val[: width - 2] + ".."
            line_parts.append(val.ljust(width))
        print(" | ".join(line_parts))

    print("=" * len(header))


def print_summary():
    """Print a provider-level summary."""
    print("\n\n")
    print("=" * 90)
    print("  PROVIDER SUMMARY")
    print("=" * 90)

    providers_seen = []
    provider_results: Dict[str, List[TestResult]] = {}
    for r in ALL_RESULTS:
        if r.provider not in provider_results:
            provider_results[r.provider] = []
            providers_seen.append(r.provider)
        provider_results[r.provider].append(r)

    summary_header = f"{'Provider':<16} | {'Pass':>4} / {'Total':>5} | {'Avg Time':>9} | {'JSON Valid':>10} | {'Verdict':<20}"
    print(summary_header)
    print("-" * len(summary_header))

    for prov in providers_seen:
        results = provider_results[prov]
        total = len(results)
        passed = sum(1 for r in results if r.success)
        times = [r.duration_s for r in results if r.duration_s > 0]
        avg_time = sum(times) / len(times) if times else 0.0
        json_tests = [r for r in results if r.valid_json is not None]
        json_valid = sum(1 for r in json_tests if r.valid_json)
        json_total = len(json_tests)
        json_str = f"{json_valid}/{json_total}" if json_total > 0 else "n/a"

        if passed == 0:
            verdict = "NOT WORKING"
        elif passed == total:
            verdict = "FULLY WORKING"
        elif passed >= total - 2:
            verdict = "MOSTLY WORKING"
        else:
            verdict = "PARTIAL"

        # Color-code (simple ANSI for terminals that support it)
        print(f"{prov:<16} | {passed:>4} / {total:>5} | {avg_time:>8.2f}s | {json_str:>10} | {verdict:<20}")

    print("=" * len(summary_header))

    # Speed ranking
    print("\n  SPEED RANKING (by avg response time across successful tests):")
    speed_data = []
    for prov in providers_seen:
        results = provider_results[prov]
        times = [r.duration_s for r in results if r.success and r.duration_s > 0]
        if times:
            speed_data.append((prov, sum(times) / len(times), min(times), max(times)))
    speed_data.sort(key=lambda x: x[1])
    for i, (prov, avg, mn, mx) in enumerate(speed_data, 1):
        print(f"    {i}. {prov:<14} avg={avg:.2f}s  min={mn:.2f}s  max={mx:.2f}s")

    if not speed_data:
        print("    (No successful tests to rank)")


# ─── Main entry point ────────────────────────────────────────────────────────

async def main():
    print("=" * 70)
    print("  LLM PROVIDER COMPREHENSIVE TEST SUITE")
    print("  Project: sales_agent")
    print(f"  Time:    {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Clear any previous failure cooldowns so all providers are tested fresh
    LLMTool._failed_providers.clear()

    # Initialize LLMTool (with force_groq to configure Groq client)
    llm = LLMTool(mock_mode=False, force_groq=True)
    settings = llm.settings

    print(f"\n  Configuration loaded from .env:")
    print(f"    NVIDIA:      {'YES' if llm._nvidia_configured else 'NO'} (model={settings.nvidia_model})")
    print(f"    Ollama:      {'YES' if settings.use_ollama else 'NO'} (model={settings.ollama_model})")
    print(f"    OpenRouter:  {'YES' if llm._openrouter_configured else 'NO'} (model={settings.openrouter_model})")
    print(f"    Gemini:      {'YES' if llm._gemini_configured else 'NO'} (model={settings.gemini_model})")
    print(f"    Groq:        {'YES' if llm._groq_configured else 'NO'} (model={settings.groq_model})")

    # ── Test each provider ────────────────────────────────────────────────

    # 1. NVIDIA
    await test_single_provider(
        llm, "NVIDIA", "_call_nvidia",
        llm._nvidia_configured,
        settings.nvidia_model,
    )

    # 2. Ollama
    # Check if Ollama is actually running before testing
    ollama_running = False
    if settings.use_ollama:
        ollama_running = await llm.check_ollama_health()
        if not ollama_running:
            print(f"\n  NOTE: Ollama is configured but NOT RUNNING at {settings.ollama_base_url}")
    await test_single_provider(
        llm, "Ollama", "_call_ollama",
        settings.use_ollama and ollama_running,
        settings.ollama_model,
    )

    # 3. OpenRouter
    await test_single_provider(
        llm, "OpenRouter", "_call_openrouter",
        llm._openrouter_configured,
        settings.openrouter_model,
    )

    # 4. Gemini
    await test_single_provider(
        llm, "Gemini", "_call_gemini",
        llm._gemini_configured,
        settings.gemini_model,
    )

    # 5. Groq
    await test_single_provider(
        llm, "Groq", "_call_groq",
        llm._groq_configured,
        settings.groq_model,
    )

    # ── Print results ─────────────────────────────────────────────────────
    print_results_table()
    print_summary()

    print(f"\n  Total tests run: {len(ALL_RESULTS)}")
    print(f"  Total passed:    {sum(1 for r in ALL_RESULTS if r.success)}")
    print(f"  Total failed:    {sum(1 for r in ALL_RESULTS if not r.success)}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
