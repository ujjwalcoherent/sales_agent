"""
LLM Provider Manager with cooldown-aware failover.

Builds pydantic-ai model instances and manages provider health.
Class-level cooldown state is shared across all instances.
"""

import asyncio
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)

# Lazy imports — pydantic-ai may not be installed during initial setup
_pai_available = False
try:
    from pydantic_ai.models import Model
    from pydantic_ai.models.fallback import FallbackModel
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    _pai_available = True
except ImportError:
    pass

# Optional providers
_groq_available = False
try:
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider
    _groq_available = True
except ImportError:
    pass

_google_available = False
try:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider
    _google_available = True
except ImportError:
    pass


class _TokenBucket:
    """Async token bucket — smooths burst requests to a steady per-second rate.

    Allows up to `capacity` requests to fire immediately, then paces subsequent
    requests at `rate` per second. Thread-safe via asyncio.Lock (lazy-initialised
    so it binds to the running event loop on first use).
    """

    def __init__(self, rate: float, capacity: float):
        self._rate = rate          # tokens refilled per second
        self._capacity = capacity  # max burst size
        self._tokens = capacity    # start full so first N calls are instant
        self._last_refill = time.monotonic()
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def acquire(self) -> None:
        """Wait until a token is available, then consume one."""
        async with self._get_lock():
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last_refill = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait = (1.0 - self._tokens) / self._rate
        # Release lock while sleeping so other coroutines can check/wait too
        await asyncio.sleep(wait)
        await self.acquire()  # re-enter to consume token after sleep


class ProviderManager:
    """Manages LLM provider lifecycle with cooldown-based failover.

    Class-level _failed_providers dict is shared across all instances,
    preserving the existing LLMTool behavior where a rate-limited provider
    is skipped by all agents for the cooldown duration.
    """

    _failed_providers: Dict[str, float] = {}
    _failure_counts: Dict[str, int] = {}   # For exponential backoff
    _FAILURE_COOLDOWN = 0.0    # Switch instantly on generic errors
    _RATELIMIT_COOLDOWN = 30.0 # Base 429 cooldown — actual = 30 * 2^(n-1), capped at 300s
    _BILLING_COOLDOWN = 3600.0  # 1 hour for billing/payment errors (effectively permanent)
    _TIMEOUT_COOLDOWN = 0.0    # Switch instantly on timeout
    _billing_disabled: set = set()  # Providers with permanent billing issues
    _lock = threading.Lock()   # Thread-safe for Streamlit multi-thread

    # GCP rate limiter — 1.5 req/s sustained, burst of 2.
    # Vertex AI DSQ (Dynamic Shared Quota) doesn't guarantee fixed RPM.
    # Observed: new $300-credit project gets ~30 RPM actual from DSQ.
    # 1.5 req/s (90 RPM) is 3× the observed capacity — safe headroom.
    # Previous: 4 req/s hit 429 after just 12 calls. 12 req/s was catastrophic.
    _gcp_bucket: Optional[_TokenBucket] = None

    # Vertex AI API Service token cache (class-level so all instances share one refresh)
    _vertex_token: Optional[str] = None
    _vertex_token_expiry: float = 0.0

    def __init__(self, settings=None, mock_mode: bool = False, force_groq: bool = False, disabled_providers: list[str] | None = None):
        self.settings = settings or get_settings()
        self.mock_mode = mock_mode
        self.force_groq = force_groq
        self.disabled_providers = set(disabled_providers or [])
        self._provider_order: List[str] = []  # Track order for failure mapping

    def get_model(self):
        """Get current model with cooldown-aware failover.

        Returns FunctionModel for mock mode, FallbackModel for real providers.
        """
        if self.mock_mode:
            from . import mock_responses
            return FunctionModel(mock_responses.get_mock_response_for_function_model)

        available = self._get_available_providers()
        if not available:
            raise RuntimeError("No LLM providers available (all in cooldown or unconfigured)")
        if len(available) == 1:
            return available[0][1]

        models = [m for _, m in available]
        self._provider_order = [name for name, _ in available]

        return FallbackModel(
            models[0], *models[1:],
            fallback_on=self._should_fallback,
        )

    def get_lite_model(self):
        """Get a cheaper/faster model for classification tasks.

        Uses gemini-2.5-flash-lite via Vertex AI (separate DSQ pool from main model).
        Falls back to Groq (free) then standard chain.
        Classification tasks: event classification, trend validation, lead filtering.
        """
        if self.mock_mode:
            from . import mock_responses
            return FunctionModel(mock_responses.get_mock_response_for_function_model)

        # In offline mode, skip straight to standard chain (Ollama)
        if getattr(self.settings, 'offline_mode', False):
            return self.get_model()

        providers = []
        now = time.time()

        # 1. OpenAI nano FIRST — $0.10/1M input, 500+ RPM, great for classification
        if getattr(self.settings, 'openai_api_key', '') and self._is_provider_available("OpenAINano", now):
            providers.append(self._build_openai_model(lite=True))

        # 2. GeminiDirect flash-lite — separate DSQ pool, cheapest
        # Uses "GeminiDirectLite" cooldown (independent from flash-2.0)
        has_vertex = self.settings.gcp_project_id or self.settings.vertex_express_api_key
        if _google_available and has_vertex and self._is_provider_available("GeminiDirectLite", now):
            providers.append(self._build_gemini_direct_model(lite=True))

        # 3. Groq — FREE, instant catch for OpenAI/GeminiDirect 429s
        if self.settings.groq_api_key and self._is_provider_available("Groq", now):
            model = self._build_groq_model()
            if model:
                providers.append(model)

        # 3. Standard chain fallback (NVIDIA, OpenRouter, Ollama)
        for _, m in self._get_available_providers():
            if m not in providers:
                providers.append(m)

        if not providers:
            return self.get_model()
        if len(providers) == 1:
            return providers[0]

        return FallbackModel(
            providers[0], *providers[1:],
            fallback_on=self._should_fallback,
        )

    def get_provider_names(self) -> List[str]:
        """Get current provider order (set after get_model() call)."""
        return list(self._provider_order)

    def _is_provider_available(self, name: str, now: float) -> bool:
        """Check if a provider is available (not disabled and not in cooldown)."""
        if name in self.disabled_providers:
            return False
        return not self._is_cooling_down(name, now)

    def _get_available_providers(self) -> List[Tuple[str, "Model"]]:
        """Build ordered provider list, skipping cooled-down and disabled providers.

        Priority order (OpenAI-first for reliability + RPM):
        1. OpenAI (GPT-4.1-mini — 500+ RPM, best structured output + tool calling)
        2. GeminiDirect (Vertex AI Gemini — $300 trial, 300 RPM Tier 1 via DSQ)
        3. VertexLlama (Llama 3.3 70B via Model Garden — same GCP project)
        4. NVIDIA (separate API, no daily token limit — good fallback)
        5. Groq (free but 100K tokens/day — last cloud resort)
        6. OpenRouter (cloud proxy)
        7. Ollama (local fallback)

        Rate limiter at 1.5 req/s (90 RPM) keeps GeminiDirect under DSQ ceiling.
        Exponential backoff on 429s. flash-lite uses separate cooldown (GeminiDirectLite).
        """
        providers = []
        now = time.time()

        offline = getattr(self.settings, 'offline_mode', False)

        # In offline mode, Ollama goes FIRST (skip cloud providers)
        if offline and self.settings.use_ollama and self._is_provider_available("Ollama", now):
            logger.info(f"LLM: OFFLINE MODE — using Ollama ({self.settings.ollama_model}) at {self.settings.ollama_base_url}")
            providers.append(("Ollama", self._build_ollama_model()))
            self._provider_order = [name for name, _ in providers]
            return providers

        # 1. OpenAI FIRST — GPT-4.1-mini (500+ RPM, best structured output + tool calling)
        if getattr(self.settings, 'openai_api_key', '') and self._is_provider_available("OpenAI", now):
            providers.append(("OpenAI", self._build_openai_model()))

        # 2. GeminiDirect — $300 Vertex AI trial, 300 RPM Tier 1 (DSQ)
        has_gemini = self.settings.gcp_project_id or self.settings.vertex_express_api_key or self.settings.gemini_api_key
        if _google_available and has_gemini and self._is_provider_available("GeminiDirect", now):
            providers.append(("GeminiDirect", self._build_gemini_direct_model()))

        # 3. VertexLlama — same GCP project, separate model endpoint
        if (self.settings.gcp_project_id
                and getattr(self.settings, 'vertex_llama_model', '')
                and self._is_provider_available("VertexLlama", now)):
            model = self._build_vertex_openai_model(self.settings.vertex_llama_model)
            if model:
                providers.append(("VertexLlama", model))

        # 4. NVIDIA (separate API, no TPD limits — good GCP fallback)
        if self.settings.nvidia_api_key and self._is_provider_available("NVIDIA", now):
            providers.append(("NVIDIA", self._build_nvidia_model()))

        # 5. Groq (free but only 100K tokens/day — use sparingly)
        if self.settings.groq_api_key and self._is_provider_available("Groq", now):
            model = self._build_groq_model()
            if model:
                providers.append(("Groq", model))

        # 6. OpenRouter (cloud proxy fallback)
        if self.settings.openrouter_api_key and self._is_provider_available("OpenRouter", now):
            providers.append(("OpenRouter", self._build_openrouter_model()))

        # 7. Ollama (local fallback)
        if self.settings.use_ollama and self._is_provider_available("Ollama", now):
            providers.append(("Ollama", self._build_ollama_model()))

        self._provider_order = [name for name, _ in providers]
        return providers

    def get_structured_output_model(self):
        """Get model for structured JSON output — GCP + Groq chain.

        Providers known to fail structured output (excluded):
        - VertexLlama: "forced function calling (mode=ANY) not supported"
        - NVIDIA DeepSeek: "Invalid grammar request" (JSON schema validation)
        - OpenRouter: 402 payment required (no credits)

        Priority:
        1. OpenAI — GPT-4.1-mini (confirmed structured output + function calling, 500+ RPM)
        2. GeminiDirect — native structured output via Vertex AI function calling
        3. Groq — FREE, confirmed function calling (active secondary, not last resort)
        4. Ollama — local fallback

        Raises RuntimeError when all providers are in cooldown so LLMService
        can wait for the shortest cooldown to expire rather than cascading into
        known-failing providers.
        """
        if self.mock_mode:
            from . import mock_responses
            return FunctionModel(mock_responses.get_mock_response_for_function_model)

        providers = []
        now = time.time()

        # 1. OpenAI — GPT-4.1-mini (confirmed structured output + function calling, 500+ RPM)
        if getattr(self.settings, 'openai_api_key', '') and self._is_provider_available("OpenAI", now):
            providers.append(("OpenAI", self._build_openai_model()))

        # 2. GeminiDirect — native structured output via Vertex AI function calling
        has_gemini = self.settings.gcp_project_id or self.settings.vertex_express_api_key or self.settings.gemini_api_key
        if _google_available and has_gemini and self._is_provider_available("GeminiDirect", now):
            providers.append(("GeminiDirect", self._build_gemini_direct_model()))

        # 3. Groq — FREE active secondary (confirmed function calling, 30 RPM)
        if self.settings.groq_api_key and self._is_provider_available("Groq", now):
            model = self._build_groq_model()
            if model:
                providers.append(("Groq", model))

        # 4. Ollama — local last resort (no rate limits)
        if self.settings.use_ollama and self._is_provider_available("Ollama", now):
            providers.append(("Ollama", self._build_ollama_model()))

        # NOTE: VertexLlama excluded — 400 on tool_choice='required' (mode=ANY not supported)
        # NOTE: NVIDIA excluded — 400 "Invalid grammar request" on JSON schema structured output

        if not providers:
            raise RuntimeError("No LLM providers available for structured output (all in cooldown)")

        if len(providers) == 1:
            return providers[0][1]

        models = [m for _, m in providers]
        self._provider_order = [name for name, _ in providers]
        return FallbackModel(models[0], *models[1:], fallback_on=self._should_fallback)

    def _should_fallback(self, exc: Exception) -> bool:
        """Custom fallback condition — always fallback and apply cooldowns immediately.

        Uses model_name from the exception to infer the correct provider,
        then applies cooldown so concurrent/subsequent calls skip it.
        """
        error_str = str(exc)
        is_hard_failure = (
            "402" in error_str or "Payment Required" in error_str
            or "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
            or ("403" in error_str and "billing" in error_str.lower())
            or ("401" in error_str and ("not found" in error_str.lower() or "unauthorized" in error_str.lower()))
            or "timeout" in error_str.lower() or "timed out" in error_str.lower()
        )
        if is_hard_failure:
            provider_name = self._infer_provider_from_error(exc)
            if provider_name != "unknown":
                # Only log if not already in billing-disabled set (prevents spam)
                if provider_name not in self._billing_disabled:
                    logger.warning(f"Hard failure on {provider_name}, applying cooldown: {error_str[:200]}")
                self.record_failure(provider_name, exc)
            else:
                logger.warning(f"Hard failure (unknown provider): {error_str[:200]}")
        return True  # Always try next provider

    def record_failure(self, provider_name: str, error: Exception):
        """Record a provider failure with appropriate cooldown duration."""
        # Persist cross-run failure so next startup knows about it
        try:
            from .provider_health import provider_health
            provider_health.record_failure(provider_name, str(error)[:200])
        except Exception:
            pass  # Health tracking is best-effort

        with self._lock:
            error_str = str(error)
            now = time.time()

            if "402" in error_str or "Payment Required" in error_str:
                self._billing_disabled.add(provider_name)
                if provider_name not in self._failed_providers:
                    logger.warning(f"{provider_name}: Payment required — disabled for session")
                self._failed_providers[provider_name] = now
            elif "403" in error_str and "billing" in error_str.lower():
                self._billing_disabled.add(provider_name)
                if provider_name not in self._failed_providers:
                    logger.warning(f"{provider_name}: Billing disabled (403) — disabled for session")
                self._failed_providers[provider_name] = now
            elif "401" in error_str and ("not found" in error_str.lower() or "unauthorized" in error_str.lower()):
                self._billing_disabled.add(provider_name)
                if provider_name not in self._failed_providers:
                    logger.warning(f"{provider_name}: Auth failed (401) — disabled for session")
                self._failed_providers[provider_name] = now
            elif "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                # Exponential backoff: 30s, 60s, 120s, 300s (capped)
                count = self._failure_counts.get(provider_name, 0) + 1
                self._failure_counts[provider_name] = count
                _max_cd = getattr(self.settings, 'provider_ratelimit_max_seconds', 120.0)
                cooldown = min(self._RATELIMIT_COOLDOWN * (2 ** (count - 1)), _max_cd)
                logger.info(f"{provider_name}: Rate limited — {int(cooldown)}s cooldown (#{count})")
                self._failed_providers[provider_name] = now - (self._FAILURE_COOLDOWN - cooldown)
            elif "404" in error_str and ("does not exist" in error_str.lower() or "model_not_found" in error_str.lower()):
                self._billing_disabled.add(provider_name)
                logger.warning(f"{provider_name}: Model not found (404) — disabled for session. Check model name in config.")
            elif "timeout" in error_str.lower() or "timed out" in error_str.lower():
                effective_start = now - (self._FAILURE_COOLDOWN - self._TIMEOUT_COOLDOWN)
                logger.warning(f"{provider_name}: Timeout — cooldown {int(self._TIMEOUT_COOLDOWN)}s")
                self._failed_providers[provider_name] = effective_start
            else:
                # Transient error — no cooldown, just log
                logger.info(f"{provider_name}: Transient error (no cooldown): {error_str[:200]}")

    def _is_cooling_down(self, provider_name: str, now: float) -> bool:
        """Check if provider is in cooldown period."""
        with self._lock:
            # Billing-disabled providers stay down for the entire session
            if provider_name in self._billing_disabled:
                return True
            if provider_name in self._failed_providers:
                fail_time = self._failed_providers[provider_name]
                if now - fail_time < self._FAILURE_COOLDOWN:
                    return True
                else:
                    del self._failed_providers[provider_name]
                    # Reset failure count on successful recovery
                    self._failure_counts.pop(provider_name, None)
                    logger.info(f"{provider_name} cooldown expired, re-enabling")
        return False

    def _infer_provider_from_error(self, exc: Exception) -> str:
        """Infer which provider an exception came from using model_name and URL.

        This avoids the positional-index mapping bug where cached FallbackModel
        provider order diverges from the current _provider_order.
        """
        err = str(exc)
        s = self.settings

        # Extract model_name from pydantic-ai error format: "model_name: xyz,"
        match = re.search(r'model_name:\s*([^,\s]+)', err)
        model_name = match.group(1) if match else ""

        # Direct model name match (most reliable)
        # CRITICAL: flash-lite and flash-2.0 are in SEPARATE GCP DSQ pools.
        # A flash-lite 429 must NOT put flash-2.0 into cooldown.
        if model_name == s.gemini_lite_model:
            if "openrouter" in err.lower():
                return "OpenRouter"
            return "GeminiDirectLite"
        if model_name == s.gemini_model:
            if "openrouter" in err.lower():
                return "OpenRouter"
            return "GeminiDirect"
        if model_name == getattr(s, 'vertex_deepseek_model', ''):
            return "VertexDeepSeek"
        if model_name == getattr(s, 'vertex_llama_model', ''):
            return "VertexLlama"
        if model_name == s.nvidia_model:
            return "NVIDIA"
        if model_name == s.ollama_model:
            return "Ollama"
        if model_name == s.openrouter_model:
            return "OpenRouter"
        if model_name == getattr(s, 'groq_model', ''):
            return "Groq"
        # OpenAI models
        if model_name == getattr(s, 'openai_model', 'gpt-4.1-mini'):
            return "OpenAI"
        if model_name == getattr(s, 'openai_lite_model', 'gpt-4.1-nano'):
            return "OpenAINano"

        # URL-based fallback
        if "integrate.api.nvidia" in err:
            return "NVIDIA"
        if "openrouter.ai" in err:
            return "OpenRouter"
        if "localhost:11434" in err:
            return "Ollama"
        if "api.openai.com" in err:
            if "nano" in model_name:
                return "OpenAINano"
            return "OpenAI"
        # Vertex API Service uses aiplatform.googleapis.com — check model name first
        if "aiplatform.googleapis.com" in err:
            if "deepseek" in err.lower():
                return "VertexDeepSeek"
            if "llama" in err.lower():
                return "VertexLlama"
            return "GeminiDirect"
        if "googleapis.com" in err:
            return "GeminiDirect"

        return "unknown"

    # --- Provider constructors ---

    def _build_nvidia_model(self):
        """NVIDIA via OpenAI-compatible endpoint."""
        return OpenAIChatModel(
            model_name=self.settings.nvidia_model,
            provider=OpenAIProvider(
                base_url=self.settings.nvidia_base_url,
                api_key=self.settings.nvidia_api_key,
            ),
        )

    def _build_ollama_model(self):
        """Ollama via OpenAI-compatible endpoint (generic model — legacy)."""
        return OpenAIChatModel(
            model_name=self.settings.ollama_model,
            provider=OpenAIProvider(
                base_url=f"{self.settings.ollama_base_url}/v1",
            ),
        )

    def _build_ollama_gen_model(self):
        """Ollama generation model (phi3.5-custom — fast synthesis, NO tool calling).

        Use for: synthesis, impact analysis, generation tasks.
        Do NOT use for: agents that need tool/function calling.
        """
        model_name = getattr(self.settings, 'ollama_gen_model', 'phi3.5-custom:latest')
        return OpenAIChatModel(
            model_name=model_name,
            provider=OpenAIProvider(
                base_url=f"{self.settings.ollama_base_url}/v1",
            ),
        )

    def _build_ollama_tool_model(self):
        """Ollama tool-calling model (llama3.2:3b — ONLY local model with tool support).

        Use for: ReAct agents, CausalCouncil, any @agent.tool decorated agent.
        Hardware note: MX550 2GB VRAM — do NOT swap to 7B+, it runs on CPU at ~1 tok/s.
        """
        model_name = getattr(self.settings, 'ollama_tool_model', 'llama3.2:3b')
        return OpenAIChatModel(
            model_name=model_name,
            provider=OpenAIProvider(
                base_url=f"{self.settings.ollama_base_url}/v1",
            ),
        )

    def _build_openrouter_model(self):
        """OpenRouter via OpenAI-compatible endpoint."""
        return OpenAIChatModel(
            model_name=self.settings.openrouter_model,
            provider=OpenAIProvider(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.settings.openrouter_api_key,
            ),
        )

    def _build_openai_model(self, lite: bool = False):
        """OpenAI native API — GPT-4.1-mini (gen) or GPT-4.1-nano (lite).

        Uses pydantic-ai's OpenAIChatModel with OpenAIProvider pointed at
        the official api.openai.com. No custom base_url needed.
        """
        model_name = (
            getattr(self.settings, 'openai_lite_model', 'gpt-4.1-nano')
            if lite
            else getattr(self.settings, 'openai_model', 'gpt-4.1-mini')
        )
        return OpenAIChatModel(
            model_name=model_name,
            provider=OpenAIProvider(
                api_key=self.settings.openai_api_key,
            ),
        )

    def _build_gemini_direct_model(self, lite: bool = False):
        """Gemini via Vertex AI or Google AI Studio.

        Priority:
        1. Full Vertex AI (service account — 60+ RPM, uses GCP credits)
        2. Vertex Express (API key — 10 RPM, free tier)
        3. Gemini API key (Google AI Studio — 15 RPM, free tier)

        Args:
            lite: If True, use gemini_lite_model (flash-lite) for classification tasks.
        """
        model_name = self.settings.gemini_lite_model if lite else self.settings.gemini_model

        # Priority 1: Full Vertex AI with service account (60+ RPM, $300 credits)
        if self.settings.gcp_project_id and self.settings.gcp_service_account_file:
            sa_path = self.settings.gcp_service_account_file
            # Resolve relative paths from project root
            if not os.path.isabs(sa_path):
                sa_path = str(Path(__file__).resolve().parents[2] / sa_path)
            if os.path.exists(sa_path):
                try:
                    from google.oauth2.service_account import Credentials
                    credentials = Credentials.from_service_account_file(
                        sa_path,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                    logger.info(f"Using full Vertex AI (project={self.settings.gcp_project_id}, "
                                f"location={self.settings.gcp_vertex_location}, model={model_name})")
                    return GoogleModel(
                        model_name=model_name,
                        provider=GoogleProvider(
                            vertexai=True,
                            project=self.settings.gcp_project_id,
                            location=self.settings.gcp_vertex_location,
                            credentials=credentials,
                        ),
                    )
                except Exception as e:
                    logger.warning(f"Full Vertex AI auth failed, falling back: {e}")

        # Priority 2: Vertex Express (API key — 10 RPM free tier)
        if self.settings.vertex_express_api_key:
            return GoogleModel(
                model_name=model_name,
                provider=GoogleProvider(
                    vertexai=True,
                    api_key=self.settings.vertex_express_api_key,
                ),
            )

        # Priority 3: Gemini API key (Google AI Studio)
        return GoogleModel(
            model_name=model_name,
            provider=GoogleProvider(
                api_key=self.settings.gemini_api_key,
            ),
        )

    def _build_gemini_via_openrouter(self):
        """Gemini routed through OpenRouter (fallback if direct unavailable)."""
        return OpenAIChatModel(
            model_name=f"google/{self.settings.gemini_model}",
            provider=OpenAIProvider(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.settings.openrouter_api_key,
            ),
        )

    def _get_vertex_access_token(self) -> Optional[str]:
        """Return a valid Vertex AI OAuth2 access token, refreshing when < 60s remain.

        Vertex AI API Service partner models (DeepSeek, Llama 4) accept the same
        GCP service account credentials as the native Gemini models — the access token
        is passed as a Bearer `api_key` to OpenAIProvider.  Tokens are cached at the
        class level so all ProviderManager instances share one refresh cycle.
        """
        sa_path = self.settings.gcp_service_account_file
        if not (self.settings.gcp_project_id and sa_path):
            return None
        if not os.path.isabs(sa_path):
            sa_path = str(Path(__file__).resolve().parents[2] / sa_path)
        if not os.path.exists(sa_path):
            return None

        now = time.time()
        if ProviderManager._vertex_token and now < ProviderManager._vertex_token_expiry - 60:
            return ProviderManager._vertex_token

        try:
            from google.oauth2.service_account import Credentials
            import google.auth.transport.requests

            creds = Credentials.from_service_account_file(
                sa_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            creds.refresh(google.auth.transport.requests.Request())
            ProviderManager._vertex_token = creds.token
            expiry = creds.expiry.timestamp() if creds.expiry else now + 3600
            ProviderManager._vertex_token_expiry = expiry
            logger.debug("Vertex AI access token refreshed (expires in ~1h)")
            return ProviderManager._vertex_token
        except Exception as e:
            logger.warning(f"Vertex AI token refresh failed: {e}")
            return None

    def _build_vertex_openai_model(self, model_name: str):
        """Vertex AI API Service model via OpenAI-compatible endpoint.

        Used for Model Garden partner models (DeepSeek V3.2, Llama 4) that are
        not natively in pydantic-ai's GoogleModel but share the same GCP project
        and service account.  The endpoint is the standard Vertex AI OpenAI proxy.

        Args:
            model_name: Vertex model ID, e.g. "deepseek/deepseek-v3-2" or
                        "meta/llama-4-scout-17b-16e-instruct-maas".
        Returns None when GCP credentials are unavailable.
        """
        token = self._get_vertex_access_token()
        if not token:
            return None
        location = self.settings.gcp_vertex_location
        project = self.settings.gcp_project_id
        base_url = (
            f"https://{location}-aiplatform.googleapis.com/v1beta1"
            f"/projects/{project}/locations/{location}/endpoints/openapi"
        )
        return OpenAIChatModel(
            model_name=model_name,
            provider=OpenAIProvider(
                base_url=base_url,
                api_key=token,
            ),
        )

    def _build_groq_model(self, tool_capable: bool = False):
        """Groq model — uses native GroqModel if available, else OpenRouter.

        Args:
            tool_capable: If True, use GROQ_TOOL_MODEL (llama-3.3-70b-versatile)
                          which has confirmed tool/function calling support.
                          If False, use GROQ_MODEL (default: qwen-qwen3-32b)
                          which is better for synthesis/generation tasks.
        """
        model_name = (
            getattr(self.settings, 'groq_tool_model', 'llama-3.3-70b-versatile')
            if tool_capable
            else self.settings.groq_model
        )
        # If model name has OpenRouter-style prefix, route through OpenRouter
        if "/" in model_name and not _groq_available:
            if self.settings.openrouter_api_key:
                return OpenAIChatModel(
                    model_name=model_name,
                    provider=OpenAIProvider(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=self.settings.openrouter_api_key,
                    ),
                )
            return None
        if _groq_available and self.settings.groq_api_key:
            # max_retries=0: let OUR FallbackModel switch providers on 429
            # instead of Groq SDK retrying 4-5x internally (adds 20-40s latency).
            from groq import AsyncGroq
            groq_client = AsyncGroq(
                api_key=self.settings.groq_api_key,
                max_retries=0,
            )
            return GroqModel(
                model_name=model_name,
                provider=GroqProvider(groq_client=groq_client),
            )
        return None

    def get_tool_capable_model(self):
        """Get a model guaranteed to support tool/function calling.

        GCP + Groq priority — Gemini native function calling + Groq free fallback.

        Priority:
        1. OpenAI — GPT-4.1-mini (full parallel function calling, 500+ RPM)
        2. GeminiDirect — Gemini 2.0 Flash native function calling
        3. Groq llama-3.3-70b-versatile — FREE active secondary (30 RPM)
        4. VertexLlama — Llama 3.3 70B via Model Garden (paid account)
        5. Ollama llama3.2:3b — local offline fallback

        Use this for agentic agents that call tools (search, scrape, etc.)
        rather than just producing structured output.
        """
        if self.mock_mode:
            from . import mock_responses
            return FunctionModel(mock_responses.get_mock_response_for_function_model)

        now = time.time()
        providers = []

        # 1. OpenAI FIRST — GPT-4.1-mini (full parallel function calling, 500+ RPM)
        if getattr(self.settings, 'openai_api_key', '') and not self._is_cooling_down("OpenAI", now):
            providers.append(("OpenAI-Tool", self._build_openai_model()))

        # 2. GeminiDirect — native function calling, $300 Vertex AI trial
        has_gemini = self.settings.gcp_project_id or self.settings.vertex_express_api_key or self.settings.gemini_api_key
        if _google_available and has_gemini and not self._is_cooling_down("GeminiDirect", now):
            providers.append(("GeminiDirect-Tool", self._build_gemini_direct_model()))

        # 3. Groq — FREE active secondary (confirmed tool calling, 30 RPM)
        if self.settings.groq_api_key and not self._is_cooling_down("Groq", now):
            model = self._build_groq_model(tool_capable=True)
            if model:
                providers.append(("Groq-Tool", model))

        # 4. VertexLlama — Llama 3.3 70B (paid account required)
        if (self.settings.gcp_project_id
                and getattr(self.settings, 'vertex_llama_model', '')
                and not self._is_cooling_down("VertexLlama", now)):
            model = self._build_vertex_openai_model(self.settings.vertex_llama_model)
            if model:
                providers.append(("VertexLlama-Tool", model))

        # 5. Ollama llama3.2:3b — offline fallback (only local model with tool support)
        if self.settings.use_ollama and not self._is_cooling_down("Ollama", now):
            providers.append(("Ollama-Tool", self._build_ollama_tool_model()))

        if not providers:
            raise RuntimeError("No tool-capable LLM provider available")
        if len(providers) == 1:
            return providers[0][1]

        models = [m for _, m in providers]
        return FallbackModel(models[0], *models[1:], fallback_on=self._should_fallback)

    # --- Health checks ---

    async def check_ollama_health(self) -> bool:
        """Check if Ollama is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.settings.ollama_base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    async def get_provider_status(self) -> Dict[str, bool]:
        """Get status of all LLM providers for UI display."""
        has_gcp = bool(self.settings.gcp_project_id and self.settings.gcp_service_account_file)
        nvidia_ok = bool(self.settings.nvidia_api_key)
        ollama_ok = await self.check_ollama_health() if self.settings.use_ollama else False
        vertex_full_ok = has_gcp and _google_available
        vertex_express_ok = bool(self.settings.vertex_express_api_key) and _google_available
        gemini_ok = bool(self.settings.gemini_api_key)
        groq_ok = bool(self.settings.groq_api_key)
        openrouter_ok = bool(self.settings.openrouter_api_key)
        openai_ok = bool(getattr(self.settings, 'openai_api_key', ''))
        vertex_deepseek_ok = has_gcp and bool(getattr(self.settings, 'vertex_deepseek_model', ''))
        vertex_llama_ok = has_gcp and bool(getattr(self.settings, 'vertex_llama_model', ''))

        return {
            "nvidia": nvidia_ok,
            "ollama": ollama_ok,
            "vertex_full": vertex_full_ok,
            "gemini_direct": vertex_express_ok,
            "gemini": gemini_ok,
            "groq": groq_ok,
            "openrouter": openrouter_ok,
            "openai": openai_ok,
            "vertex_deepseek": vertex_deepseek_ok,
            "vertex_llama": vertex_llama_ok,
            "any_available": (
                nvidia_ok or ollama_ok or vertex_full_ok or vertex_express_ok
                or gemini_ok or groq_ok or openrouter_ok or openai_ok
                or vertex_deepseek_ok or vertex_llama_ok
            ),
        }

    def get_shortest_cooldown_remaining(self) -> float:
        """Seconds until the next non-billing-disabled provider exits cooldown.

        Returns 0.0 if any provider is already available.
        Used by LLMService to wait instead of immediately raising when all
        providers are temporarily rate-limited.
        """
        if self._get_available_providers():
            return 0.0
        with self._lock:
            remaining = [
                fail_time - time.time()
                for name, fail_time in self._failed_providers.items()
                if name not in self._billing_disabled
            ]
        valid = [r for r in remaining if r > 0]
        return min(valid) if valid else 0.0

    @classmethod
    async def acquire_gcp_rate_limit(cls) -> None:
        """Throttle GCP (Gemini + VertexLlama) calls to ≤1.5 req/s.

        Vertex AI DSQ observed capacity: ~30 RPM for new $300-credit projects.
        1.5 req/s (90 RPM) = 3× observed, with burst of 2 for tool-call pairs.
        """
        if cls._gcp_bucket is None:
            cls._gcp_bucket = _TokenBucket(rate=1.5, capacity=2.0)
        await cls._gcp_bucket.acquire()

    @classmethod
    def reset_cooldowns(cls):
        """Clear all cooldowns and failure counts for a fresh pipeline run."""
        with cls._lock:
            cls._failed_providers.clear()
            cls._billing_disabled.clear()
            cls._failure_counts.clear()
