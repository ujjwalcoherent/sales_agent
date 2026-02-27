"""
LLM Service — pydantic-ai backed replacement for LLMTool.

Provides both typed structured output (Track A) and backward-compatible
dict-based methods (Track B) matching the old LLMTool signatures.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from ..config import get_settings
from . import json_repair
from .provider_manager import ProviderManager

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

# Import FallbackExceptionGroup if available (pydantic-ai >= 0.2)
try:
    from pydantic_ai.models.fallback import FallbackExceptionGroup
except ImportError:
    FallbackExceptionGroup = ExceptionGroup  # type: ignore[misc]


class LLMService:
    """High-level LLM service backed by pydantic-ai.

    Drop-in replacement for LLMTool with the same method signatures.
    Internally uses pydantic-ai Agents with cached instances for reuse.
    """

    # Cache agents by (output_type, system_prompt_hash, retries) across all instances
    _agent_cache: Dict[tuple, Agent] = {}

    def __init__(self, mock_mode: bool = False, force_groq: bool = False, lite: bool = False, disabled_providers: list[str] | None = None):
        self.settings = get_settings()
        self.mock_mode = mock_mode or self.settings.mock_mode
        self.force_groq = force_groq
        self.lite = lite  # Use cheaper model for classification tasks
        self.provider_manager = ProviderManager(
            settings=self.settings,
            mock_mode=self.mock_mode,
            force_groq=force_groq,
            disabled_providers=disabled_providers or [],
        )
        self.last_provider: Optional[str] = None
        if getattr(self.settings, 'offline_mode', False):
            logger.info(f"LLM ({'lite' if lite else 'full'}): OFFLINE → Ollama/{self.settings.ollama_model}")
        elif self.mock_mode:
            logger.info(f"LLM ({'lite' if lite else 'full'}): MOCK mode")
        else:
            logger.info(f"LLM ({'lite' if lite else 'full'}): ONLINE mode (cloud providers active)")

    def _get_or_create_agent(self, output_type: type, system_prompt: str, retries: int = 2) -> Agent:
        """Get or create a cached pydantic-ai Agent.

        Cache key includes cooldown state so the FallbackModel is rebuilt
        when providers enter/exit cooldown (prevents trying dead providers).

        For structured output (output_type != str): uses get_structured_output_model()
        which skips VertexLlama (it returns 400 for tool_choice='required').
        For text output (output_type == str): uses get_model() with VertexLlama primary.
        """
        cooldown_key = frozenset(ProviderManager._failed_providers.keys())
        needs_structured = output_type is not str
        key = (output_type, hash(system_prompt), retries, self.mock_mode, self.lite, cooldown_key, needs_structured)
        if key not in self._agent_cache:
            if self.lite:
                model = self.provider_manager.get_lite_model()
            elif needs_structured:
                # VertexLlama doesn't support forced function calling — use dedicated chain
                model = self.provider_manager.get_structured_output_model()
            else:
                model = self.provider_manager.get_model()
            self._agent_cache[key] = Agent(
                model,
                output_type=output_type,
                system_prompt=system_prompt,
                retries=retries,
            )
        return self._agent_cache[key]

    # ── Cooldown-aware agent creation ─────────────────────────────

    async def _get_agent_with_cooldown_wait(
        self, output_type: type, system_prompt: str, retries: int = 2,
    ) -> Agent:
        """Create agent, waiting for cooldown if all providers are temporarily down.

        Waits at most `llm_max_provider_wait` seconds (default 30s). If no
        provider becomes available by then, raises RuntimeError immediately
        instead of hanging the pipeline.
        """
        max_wait = getattr(self.settings, 'llm_max_provider_wait', 30.0)
        try:
            return self._get_or_create_agent(output_type, system_prompt, retries)
        except RuntimeError as e:
            if "No LLM providers available" not in str(e):
                raise
            wait_sec = self.provider_manager.get_shortest_cooldown_remaining()
            if wait_sec <= 0 or wait_sec > max_wait:
                raise RuntimeError(
                    f"All LLM providers exhausted (next available in {wait_sec:.0f}s, "
                    f"max wait {max_wait:.0f}s)"
                ) from e
            logger.info(f"All providers in cooldown — waiting {wait_sec:.1f}s (max {max_wait:.0f}s)")
            await asyncio.sleep(wait_sec + 1.0)
            # One retry after the wait — if still down, raise immediately
            return self._get_or_create_agent(output_type, system_prompt, retries)

    def has_available_provider(self) -> bool:
        """Check if any LLM provider is currently available (not in cooldown).

        Callers can use this to skip LLM calls gracefully when all providers
        are exhausted, instead of waiting and hanging.
        """
        try:
            avail = self.provider_manager._get_available_providers()
            return len(avail) > 0
        except Exception:
            return False

    # ── Track A: Typed structured output (new API) ──────────────────

    async def run_structured(
        self,
        prompt: str,
        system_prompt: str = "",
        output_type: Type[T] = str,  # type: ignore[assignment]
        retries: int = 2,
        temperature: float = 0.3,
    ) -> T:
        """Generate structured output validated by pydantic-ai.

        Uses the Pydantic model's own validators for LLM output coercion.
        When all providers are rate-limited, waits up to `llm_max_provider_wait`
        seconds for the shortest cooldown, then raises instead of hanging.
        """
        agent = await self._get_agent_with_cooldown_wait(output_type, system_prompt, retries)
        try:
            if not self.mock_mode:
                await ProviderManager.acquire_gcp_rate_limit()
            result = await agent.run(
                prompt,
                model_settings=ModelSettings(temperature=temperature),
            )
            self.last_provider = self._extract_provider_name(result)
            return result.output
        except (FallbackExceptionGroup, ExceptionGroup) as eg:
            self._process_failures(eg)
            raise RuntimeError(f"All LLM providers failed: {eg}") from eg

    # ── Track B: Backward-compatible wrappers (same LLMTool signatures) ──

    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema_hint: Optional[str] = None,
        pydantic_model: Optional[Type] = None,
        required_keys: Optional[set] = None,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Drop-in replacement for LLMTool.generate_json().

        If pydantic_model is provided, uses Track A (typed structured output).
        Otherwise uses Track B (raw text + JSON repair).
        """
        retries = max_retries if max_retries is not None else self.settings.llm_json_max_retries
        sys_prompt = system_prompt or ""

        # Track A: typed structured output
        if pydantic_model:
            try:
                result = await self.run_structured(
                    prompt=prompt,
                    system_prompt=sys_prompt,
                    output_type=pydantic_model,
                    retries=retries,
                )
                return result.model_dump() if hasattr(result, 'model_dump') else dict(result)
            except Exception as e:
                logger.warning(f"Track A failed, falling through to Track B: {e}")

        # Track B: raw text → JSON repair → dict
        json_instruction = "\nYou must respond with valid JSON only. No markdown, no explanation."
        try:
            agent = await self._get_agent_with_cooldown_wait(str, sys_prompt + json_instruction, retries)
        except RuntimeError:
            return {"error": "All LLM providers exhausted"}
        try:
            if not self.mock_mode:
                await ProviderManager.acquire_gcp_rate_limit()
            result = await agent.run(
                prompt,
                model_settings=ModelSettings(temperature=0.3),
            )
            self.last_provider = self._extract_provider_name(result)
            parsed = json_repair.parse_json_response(result.output)

            # Validate required keys if specified
            if required_keys and isinstance(parsed, dict):
                missing = required_keys - set(parsed.keys())
                if missing:
                    logger.warning(f"Missing required keys: {missing}")

            return parsed
        except (FallbackExceptionGroup, ExceptionGroup) as eg:
            self._process_failures(eg)
            return {"error": str(eg)}
        except Exception as e:
            logger.error(f"generate_json failed: {e}")
            return {"error": str(e)}

    async def generate_list(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Drop-in replacement for LLMTool.generate_list()."""
        json_instruction = """
You must respond with a valid JSON array only. No markdown, no explanation.
Each item in the array should be a JSON object.
Do not wrap the response in ```json``` code blocks."""
        full_system = (system_prompt or "") + "\n" + json_instruction

        raw = await self.generate_json(prompt=prompt, system_prompt=full_system.strip())
        return json_repair.extract_list_from_response(raw)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        json_mode: bool = False,
    ) -> str:
        """Drop-in replacement for LLMTool.generate()."""
        if self.mock_mode:
            from . import mock_responses
            return mock_responses.get_mock_response(prompt, json_mode)

        agent = await self._get_agent_with_cooldown_wait(str, system_prompt or "", retries=1)
        try:
            await ProviderManager.acquire_gcp_rate_limit()
            result = await agent.run(
                prompt,
                model_settings=ModelSettings(temperature=temperature, max_tokens=max_tokens),
            )
            self.last_provider = self._extract_provider_name(result)
            response = result.output
            if response:
                return response
            raise ValueError("Empty response")
        except (FallbackExceptionGroup, ExceptionGroup) as eg:
            self._process_failures(eg)
            raise RuntimeError(f"All LLM providers failed: {eg}") from eg

    # ── Provider status (for Streamlit sidebar) ──────────────────────

    async def get_provider_status(self) -> Dict[str, bool]:
        """Delegate to ProviderManager for UI display."""
        return await self.provider_manager.get_provider_status()

    async def check_ollama_health(self) -> bool:
        """Check if Ollama is available."""
        return await self.provider_manager.check_ollama_health()

    # ── Internal helpers ─────────────────────────────────────────────

    def _process_failures(self, eg: BaseException):
        """Extract provider failures from exception group and record cooldowns.

        Uses model-name-based provider inference instead of positional indexing
        to avoid misattributing errors when cached agents have stale provider lists.
        """
        exceptions = getattr(eg, 'exceptions', [eg])
        for exc in exceptions:
            provider_name = self.provider_manager._infer_provider_from_error(exc)
            err_msg = str(exc)[:150]
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                logger.warning(f"  {provider_name}: Rate limited (429)")
            elif "402" in err_msg or "Payment" in err_msg:
                logger.warning(f"  {provider_name}: Payment required (402)")
            elif "timeout" in err_msg.lower():
                logger.warning(f"  {provider_name}: Timeout")
            else:
                logger.warning(f"  {provider_name}: {err_msg}")
            self.provider_manager.record_failure(provider_name, exc)

    def _extract_provider_name(self, result) -> str:
        """Try to extract which provider was used from the result."""
        try:
            # Check response messages for model info
            messages = result.all_messages()
            for msg in reversed(messages):
                if hasattr(msg, 'model_name'):
                    return msg.model_name
        except Exception:
            pass
        # Fall back to first available provider name
        names = self.provider_manager.get_provider_names()
        return names[0] if names else "unknown"

    @classmethod
    def clear_cache(cls):
        """Clear agent cache. Useful for testing or config changes."""
        cls._agent_cache.clear()
