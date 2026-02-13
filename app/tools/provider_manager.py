"""
LLM Provider Manager with cooldown-aware failover.

Builds pydantic-ai model instances and manages provider health.
Class-level cooldown state is shared across all instances.
"""

import logging
import threading
import time
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


class ProviderManager:
    """Manages LLM provider lifecycle with cooldown-based failover.

    Class-level _failed_providers dict is shared across all instances,
    preserving the existing LLMTool behavior where a rate-limited provider
    is skipped by all agents for the cooldown duration.
    """

    _failed_providers: Dict[str, float] = {}
    _FAILURE_COOLDOWN = 300.0  # 5 minutes for payment/rate limit errors
    _TIMEOUT_COOLDOWN = 60.0   # 1 minute for timeout errors
    _lock = threading.Lock()   # Thread-safe for Streamlit multi-thread

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

    def get_provider_names(self) -> List[str]:
        """Get current provider order (set after get_model() call)."""
        return list(self._provider_order)

    def _get_available_providers(self) -> List[Tuple[str, "Model"]]:
        """Build ordered provider list, skipping cooled-down providers."""
        providers = []
        now = time.time()

        # Groq first (only if forced for deep reasoning)
        if self.force_groq and self.settings.groq_api_key and not self._is_cooling_down("Groq", now):
            model = self._build_groq_model()
            if model:
                providers.append(("Groq", model))

        # NVIDIA (primary — OpenAI-compatible endpoint)
        if self.settings.nvidia_api_key and not self._is_cooling_down("NVIDIA", now):
            providers.append(("NVIDIA", self._build_nvidia_model()))

        # Ollama (local, free)
        if self.settings.use_ollama and not self._is_cooling_down("Ollama", now):
            providers.append(("Ollama", self._build_ollama_model()))

        # OpenRouter (cloud fallback)
        if self.settings.openrouter_api_key and not self._is_cooling_down("OpenRouter", now):
            providers.append(("OpenRouter", self._build_openrouter_model()))

        # Gemini (routed through OpenRouter to avoid Streamlit event loop issues)
        if self.settings.gemini_api_key and self.settings.openrouter_api_key and not self._is_cooling_down("Gemini", now):
            providers.append(("Gemini", self._build_gemini_via_openrouter()))

        self._provider_order = [name for name, _ in providers]
        return providers

    def _should_fallback(self, exc: Exception) -> bool:
        """Custom fallback condition — always fallback and record the failure."""
        error_str = str(exc)
        # We record failures but always return True to try the next provider
        # The actual cooldown is applied by record_failure + _is_cooling_down
        is_hard_failure = (
            "402" in error_str or "Payment Required" in error_str
            or "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
            or "timeout" in error_str.lower() or "timed out" in error_str.lower()
        )
        if is_hard_failure:
            logger.warning(f"Hard failure detected, will apply cooldown: {error_str[:200]}")
        return True  # Always try next provider

    def record_failure(self, provider_name: str, error: Exception):
        """Record a provider failure with appropriate cooldown duration."""
        with self._lock:
            error_str = str(error)
            now = time.time()

            if "402" in error_str or "Payment Required" in error_str:
                logger.warning(f"{provider_name}: Payment required — cooldown {int(self._FAILURE_COOLDOWN)}s")
                self._failed_providers[provider_name] = now
            elif "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                logger.warning(f"{provider_name}: Rate limited — cooldown {int(self._FAILURE_COOLDOWN)}s")
                self._failed_providers[provider_name] = now
            elif "timeout" in error_str.lower() or "timed out" in error_str.lower():
                # 60s effective cooldown (store offset time so expiry check works)
                effective_start = now - (self._FAILURE_COOLDOWN - self._TIMEOUT_COOLDOWN)
                logger.warning(f"{provider_name}: Timeout — cooldown {int(self._TIMEOUT_COOLDOWN)}s")
                self._failed_providers[provider_name] = effective_start
            else:
                # Transient error — no cooldown, just log
                logger.info(f"{provider_name}: Transient error (no cooldown): {error_str[:200]}")

    def _is_cooling_down(self, provider_name: str, now: float) -> bool:
        """Check if provider is in cooldown period."""
        with self._lock:
            if provider_name in self._failed_providers:
                fail_time = self._failed_providers[provider_name]
                if now - fail_time < self._FAILURE_COOLDOWN:
                    remaining = int(self._FAILURE_COOLDOWN - (now - fail_time))
                    logger.debug(f"Skipping {provider_name} (cooldown, {remaining}s remaining)")
                    return True
                else:
                    del self._failed_providers[provider_name]
                    logger.info(f"{provider_name} cooldown expired, re-enabling")
        return False

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
        """Ollama via OpenAI-compatible endpoint."""
        return OpenAIChatModel(
            model_name=self.settings.ollama_model,
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

    def _build_gemini_via_openrouter(self):
        """Gemini routed through OpenRouter to avoid Streamlit event loop issues."""
        return OpenAIChatModel(
            model_name=f"google/{self.settings.gemini_model}",
            provider=OpenAIProvider(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.settings.openrouter_api_key,
            ),
        )

    def _build_groq_model(self):
        """Groq model — uses native GroqModel if available, else OpenRouter."""
        model_name = self.settings.groq_model
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
            return GroqModel(
                model_name=model_name,
                provider=GroqProvider(api_key=self.settings.groq_api_key),
            )
        return None

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
        nvidia_ok = bool(self.settings.nvidia_api_key)
        ollama_ok = await self.check_ollama_health() if self.settings.use_ollama else False
        gemini_ok = bool(self.settings.gemini_api_key)
        groq_ok = bool(self.settings.groq_api_key)
        openrouter_ok = bool(self.settings.openrouter_api_key)

        return {
            "nvidia": nvidia_ok,
            "ollama": ollama_ok,
            "gemini": gemini_ok,
            "groq": groq_ok,
            "openrouter": openrouter_ok,
            "any_available": nvidia_ok or ollama_ok or gemini_ok or groq_ok or openrouter_ok,
        }

    @classmethod
    def reset_cooldowns(cls):
        """Clear all cooldowns. Useful for testing."""
        with cls._lock:
            cls._failed_providers.clear()
