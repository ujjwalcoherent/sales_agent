"""
LLM Provider Manager with cooldown-aware failover.

Builds pydantic-ai model instances and manages provider health.
Class-level cooldown state is shared across all instances.
"""

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


class ProviderManager:
    """Manages LLM provider lifecycle with cooldown-based failover.

    Class-level _failed_providers dict is shared across all instances,
    preserving the existing LLMTool behavior where a rate-limited provider
    is skipped by all agents for the cooldown duration.
    """

    _failed_providers: Dict[str, float] = {}
    _FAILURE_COOLDOWN = 0.0    # Switch instantly on generic errors
    _RATELIMIT_COOLDOWN = 30.0 # 30s for rate-limit (429) — prevents hammering
    _BILLING_COOLDOWN = 3600.0  # 1 hour for billing/payment errors (effectively permanent)
    _TIMEOUT_COOLDOWN = 0.0    # Switch instantly on timeout
    _billing_disabled: set = set()  # Providers with permanent billing issues
    _lock = threading.Lock()   # Thread-safe for Streamlit multi-thread

    # Vertex AI API Service token cache (class-level so all instances share one refresh)
    _vertex_token: Optional[str] = None
    _vertex_token_expiry: float = 0.0

    def __init__(self, settings=None, mock_mode: bool = False, force_groq: bool = False):
        self.settings = settings or get_settings()
        self.mock_mode = mock_mode
        self.force_groq = force_groq
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

        Uses gemini-2.5-flash-lite via Vertex Express if available,
        otherwise falls back to the standard model chain.
        Classification tasks: event classification, trend validation, lead filtering.
        """
        if self.mock_mode:
            from . import mock_responses
            return FunctionModel(mock_responses.get_mock_response_for_function_model)

        # In offline mode, skip straight to standard chain (Ollama)
        if getattr(self.settings, 'offline_mode', False):
            return self.get_model()

        # Prefer flash-lite for classification (cheaper, faster)
        # But skip if GeminiDirect is in cooldown (billing/rate limit)
        has_vertex = self.settings.gcp_project_id or self.settings.vertex_express_api_key
        now = time.time()
        if _google_available and has_vertex and not self._is_cooling_down("GeminiDirect", now):
            lite = self._build_gemini_direct_model(lite=True)
            available = self._get_available_providers()
            if available:
                return FallbackModel(
                    lite, *[m for _, m in available],
                    fallback_on=self._should_fallback,
                )
            return lite

        # No lite model available or in cooldown — fall back to standard chain
        return self.get_model()

    def get_provider_names(self) -> List[str]:
        """Get current provider order (set after get_model() call)."""
        return list(self._provider_order)

    def _get_available_providers(self) -> List[Tuple[str, "Model"]]:
        """Build ordered provider list, skipping cooled-down providers.

        Priority order (correct for $300 GCP free trial):
        1. GeminiDirect (gemini-2.5-flash-lite via Vertex — COVERED by $300 credits, native structured output)
        2. Groq (free tier, 14K req/day — zero cost, fast fallback)
        3. NVIDIA (separate API key, capable)
        4. VertexLlama (meta/llama-3.3-70b-instruct-maas — LAST: $300 trial does NOT cover
                        third-party MaaS partner models; upgrade to paid account to use this)
        5. OpenRouter (cloud proxy fallback)
        6. Ollama (local fallback)

        NOTE: $300 free trial credits only cover Google's own Vertex AI models (Gemini).
        Partner/MaaS models (Llama, DeepSeek) require upgrading to a full paid account.
        """
        providers = []
        now = time.time()

        offline = getattr(self.settings, 'offline_mode', False)

        # In offline mode, Ollama goes FIRST (skip cloud providers)
        if offline and self.settings.use_ollama and not self._is_cooling_down("Ollama", now):
            logger.info(f"LLM: OFFLINE MODE — using Ollama ({self.settings.ollama_model}) at {self.settings.ollama_base_url}")
            providers.append(("Ollama", self._build_ollama_model()))
            self._provider_order = [name for name, _ in providers]
            return providers

        # GeminiDirect FIRST — covered by $300 free trial, native structured output support
        has_gemini = self.settings.gcp_project_id or self.settings.vertex_express_api_key or self.settings.gemini_api_key
        if _google_available and has_gemini and not self._is_cooling_down("GeminiDirect", now):
            providers.append(("GeminiDirect", self._build_gemini_direct_model()))

        # Groq — free tier (no GCP credits needed), fast, 14K req/day
        if self.settings.groq_api_key and not self._is_cooling_down("Groq", now):
            model = self._build_groq_model()
            if model:
                providers.append(("Groq", model))

        # DeepSeek V3.2 via Vertex AI API Service (NOT covered by free trial — paid only)
        if (self.settings.gcp_project_id
                and getattr(self.settings, 'vertex_deepseek_model', '')
                and not self._is_cooling_down("VertexDeepSeek", now)):
            model = self._build_vertex_openai_model(self.settings.vertex_deepseek_model)
            if model:
                providers.append(("VertexDeepSeek", model))

        # VertexLlama LAST — NOT covered by $300 free trial (third-party MaaS partner model).
        # Will 429/fail on free trial accounts. Keep as fallback in case account is upgraded.
        if (self.settings.gcp_project_id
                and getattr(self.settings, 'vertex_llama_model', '')
                and not self._is_cooling_down("VertexLlama", now)):
            model = self._build_vertex_openai_model(self.settings.vertex_llama_model)
            if model:
                providers.append(("VertexLlama", model))

        # NVIDIA (capable but often rate-limited)
        if self.settings.nvidia_api_key and not self._is_cooling_down("NVIDIA", now):
            providers.append(("NVIDIA", self._build_nvidia_model()))

        # OpenRouter (cloud fallback)
        if self.settings.openrouter_api_key and not self._is_cooling_down("OpenRouter", now):
            providers.append(("OpenRouter", self._build_openrouter_model()))

        # Ollama (local fallback)
        if self.settings.use_ollama and not self._is_cooling_down("Ollama", now):
            providers.append(("Ollama", self._build_ollama_model()))

        self._provider_order = [name for name, _ in providers]
        return providers

    def get_structured_output_model(self):
        """Get model for structured JSON output — intentionally skips VertexLlama.

        VertexLlama via OpenAI-compat endpoint returns 400 when pydantic-ai sends
        tool_choice='required' (forced function calling / mode=ANY). This method
        returns a model chain that only includes providers which support structured
        output natively.

        Use this in LLMService.run_structured() and _get_or_create_agent() when
        output_type is not str.
        """
        if self.mock_mode:
            from . import mock_responses
            return FunctionModel(mock_responses.get_mock_response_for_function_model)

        providers = []
        now = time.time()

        # GeminiDirect first — native structured output via Vertex AI function calling
        has_gemini = self.settings.gcp_project_id or self.settings.vertex_express_api_key or self.settings.gemini_api_key
        if _google_available and has_gemini and not self._is_cooling_down("GeminiDirect", now):
            providers.append(("GeminiDirect", self._build_gemini_direct_model()))

        # Groq — confirmed function calling support
        if self.settings.groq_api_key and not self._is_cooling_down("Groq", now):
            model = self._build_groq_model()
            if model:
                providers.append(("Groq", model))

        # NVIDIA — JSON grammar sometimes works
        if self.settings.nvidia_api_key and not self._is_cooling_down("NVIDIA", now):
            providers.append(("NVIDIA", self._build_nvidia_model()))

        # OpenRouter (if not billing-disabled)
        if self.settings.openrouter_api_key and not self._is_cooling_down("OpenRouter", now):
            providers.append(("OpenRouter", self._build_openrouter_model()))

        # Ollama last resort
        if self.settings.use_ollama and not self._is_cooling_down("Ollama", now):
            providers.append(("Ollama", self._build_ollama_model()))

        # NOTE: VertexLlama intentionally excluded — returns 400 for tool_choice='required'

        if not providers:
            # Fall back to full provider list (includes VertexLlama) if nothing else available
            logger.warning("No structured-output-capable providers available, falling back to full list")
            return self.get_model()

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
                logger.warning(f"{provider_name}: Rate limited — cooldown {int(self._RATELIMIT_COOLDOWN)}s")
                self._failed_providers[provider_name] = now - (self._FAILURE_COOLDOWN - self._RATELIMIT_COOLDOWN)
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
        if model_name in (s.gemini_model, s.gemini_lite_model):
            # Distinguish GeminiDirect vs OpenRouter-routed Gemini
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

        # URL-based fallback
        if "integrate.api.nvidia" in err:
            return "NVIDIA"
        if "openrouter.ai" in err:
            return "OpenRouter"
        if "localhost:11434" in err:
            return "Ollama"
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

        VertexLlama (meta/llama-3.3-70b-instruct-maas) is primary — uses GCP credits.
        Falls back to Groq llama-3.3-70b-versatile if Vertex rate-limits.

        Use this for Phase 3 agentic agents that call tools (search, scrape, etc.)
        rather than just producing structured output.
        """
        if self.mock_mode:
            from . import mock_responses
            return FunctionModel(mock_responses.get_mock_response_for_function_model)

        now = time.time()
        providers = []

        # Groq FIRST for tool calling — free, confirmed tool-calling support, no trial restrictions
        if self.settings.groq_api_key and not self._is_cooling_down("Groq", now):
            model = self._build_groq_model(tool_capable=True)
            if model:
                providers.append(("Groq-Tool", model))

        # VertexLlama — NOT covered by $300 free trial, but try if available (paid accounts)
        if (self.settings.gcp_project_id
                and getattr(self.settings, 'vertex_llama_model', '')
                and not self._is_cooling_down("VertexLlama", now)):
            model = self._build_vertex_openai_model(self.settings.vertex_llama_model)
            if model:
                providers.append(("VertexLlama-Tool", model))

        # Fall back to standard providers (may still work for tool calls)
        standard = self._get_available_providers()
        for name, m in standard:
            if name not in ("VertexLlama", "Groq"):  # Already added above
                providers.append((name, m))

        # Ollama llama3.2:3b — offline tool-calling fallback (only local model with tool support)
        if self.settings.use_ollama and not self._is_cooling_down("Ollama", now):
            if not any(name == "Ollama" for name, _ in providers):
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
            "vertex_deepseek": vertex_deepseek_ok,
            "vertex_llama": vertex_llama_ok,
            "any_available": (
                nvidia_ok or ollama_ok or vertex_full_ok or vertex_express_ok
                or gemini_ok or groq_ok or openrouter_ok
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
    def reset_cooldowns(cls):
        """Clear all cooldowns. Useful for testing."""
        with cls._lock:
            cls._failed_providers.clear()
            cls._billing_disabled.clear()
