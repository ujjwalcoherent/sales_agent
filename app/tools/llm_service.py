"""
LLM Service — pydantic-ai backed replacement for LLMTool.

Provides both typed structured output (Track A) and backward-compatible
dict-based methods (Track B) matching the old LLMTool signatures.
"""

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

    def __init__(self, mock_mode: bool = False, force_gemini: bool = False, force_groq: bool = False):
        self.settings = get_settings()
        self.mock_mode = mock_mode or self.settings.mock_mode
        self.force_groq = force_groq
        self.provider_manager = ProviderManager(
            settings=self.settings,
            mock_mode=self.mock_mode,
            force_groq=force_groq,
        )
        self.last_provider: Optional[str] = None

    def _get_or_create_agent(self, output_type: type, system_prompt: str, retries: int = 2) -> Agent:
        """Get or create a cached pydantic-ai Agent.

        Agents are cached by (output_type, system_prompt_hash, retries) for reuse.
        This avoids creating new httpx client pools per call.
        """
        key = (output_type, hash(system_prompt), retries, self.mock_mode)
        if key not in self._agent_cache:
            model = self.provider_manager.get_model()
            self._agent_cache[key] = Agent(
                model,
                output_type=output_type,
                system_prompt=system_prompt,
                retries=retries,
            )
        return self._agent_cache[key]

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
        """
        agent = self._get_or_create_agent(output_type, system_prompt, retries)
        try:
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
        agent = self._get_or_create_agent(str, sys_prompt + json_instruction, retries)
        try:
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

        agent = self._get_or_create_agent(str, system_prompt or "", retries=1)
        try:
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
        """Extract provider failures from exception group and record cooldowns."""
        providers = self.provider_manager.get_provider_names()
        exceptions = getattr(eg, 'exceptions', [eg])
        for i, exc in enumerate(exceptions):
            provider_name = providers[i] if i < len(providers) else f"unknown-{i}"
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
