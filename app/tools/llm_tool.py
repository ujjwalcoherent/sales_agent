"""
LLM Tool with multi-provider support and automatic fallback.

Provider priority:
1. NVIDIA AI Endpoints (Kimi K2.5 - cloud, fast)
2. Ollama (local, free, no rate limits)
3. OpenRouter (cloud fallback)
4. Gemini (cloud fallback)
5. Groq (ONLY if force_groq=True, for deep reasoning)

Failed cloud providers are auto-skipped for 5 minutes after rate limit/payment errors.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Type

import httpx
from ollama import AsyncClient as OllamaAsyncClient

from ..config import get_settings

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
genai = None
NEW_GENAI = False
AsyncGroq = None
GROQ_AVAILABLE = False

try:
    from google import genai as _genai
    from google.genai import types
    genai = _genai
    NEW_GENAI = True
except ImportError:
    try:
        import google.generativeai as _genai
        genai = _genai
    except ImportError:
        pass

try:
    from groq import AsyncGroq as _AsyncGroq
    AsyncGroq = _AsyncGroq
    GROQ_AVAILABLE = True
except ImportError:
    AsyncGroq = None
    pass


class LLMTool:
    """
    LLM wrapper with automatic fallback across multiple providers.
    NVIDIA (Kimi K2.5) is PRIMARY, then Ollama (local), then cloud fallbacks.
    """

    # Track providers that failed recently (class-level, shared across instances)
    _failed_providers: Dict[str, float] = {}
    _FAILURE_COOLDOWN = 300.0  # Skip failed cloud providers for 5 minutes

    def __init__(self, mock_mode: bool = False, force_gemini: bool = False, force_groq: bool = False):
        """Initialize LLM tool."""
        self.settings = get_settings()
        self.mock_mode = mock_mode or self.settings.mock_mode
        self.force_gemini = force_gemini
        self.force_groq = force_groq

        self._gemini_client = None
        self._groq_client = None
        self._gemini_configured = False
        self._groq_configured = False
        self._nvidia_configured = bool(self.settings.nvidia_api_key)
        self._openrouter_configured = bool(self.settings.openrouter_api_key)
        self.last_provider: Optional[str] = None

        self._configure_providers()

    def _configure_providers(self) -> None:
        """Configure available LLM providers."""
        if self._nvidia_configured:
            logger.info(f"NVIDIA configured: {self.settings.nvidia_model}")

        if self._openrouter_configured:
            logger.info(f"OpenRouter configured: {self.settings.openrouter_model}")

        if self.settings.groq_api_key and GROQ_AVAILABLE:
            try:
                self._groq_client = AsyncGroq(api_key=self.settings.groq_api_key)
                self._groq_configured = True
                logger.info(f"Groq configured: {self.settings.groq_model}")
            except Exception as e:
                logger.warning(f"Failed to configure Groq: {e}")

        if self.settings.gemini_api_key and genai is not None:
            try:
                if NEW_GENAI:
                    self._gemini_client = genai.Client(api_key=self.settings.gemini_api_key)
                else:
                    genai.configure(api_key=self.settings.gemini_api_key)
                self._gemini_configured = True
                logger.info("Gemini configured")
            except Exception as e:
                logger.warning(f"Failed to configure Gemini: {e}")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        json_mode: bool = False
    ) -> str:
        """Generate text using LLM with automatic fallback.

        Priority: Ollama (local, free) → Cloud providers (rate-limited).
        Cloud providers that recently failed are auto-skipped for 5 minutes.
        """
        import time

        if self.mock_mode:
            return self._get_mock_response(prompt, json_mode)

        # Build provider chain based on configuration
        providers = self._get_provider_chain()
        errors = []
        now = time.time()

        for provider_name, provider_func in providers:
            # Skip cloud providers that failed recently (rate limit cooldown)
            if provider_name != "Ollama" and provider_name in LLMTool._failed_providers:
                fail_time = LLMTool._failed_providers[provider_name]
                if now - fail_time < LLMTool._FAILURE_COOLDOWN:
                    remaining = int(LLMTool._FAILURE_COOLDOWN - (now - fail_time))
                    logger.info(f"Skipping {provider_name} (failed {remaining}s ago, cooldown active)")
                    errors.append(f"{provider_name}: Skipped (cooldown)")
                    continue
                else:
                    # Cooldown expired, try again
                    del LLMTool._failed_providers[provider_name]

            try:
                logger.info(f"Trying {provider_name}...")
                response = await provider_func(prompt, system_prompt, temperature, max_tokens)
                if response:
                    self.last_provider = provider_name
                    logger.info(f"Success via {provider_name}")
                    return response
                else:
                    raise ValueError("Empty response")
            except Exception as e:
                error_str = str(e)

                # Detect HARD failures (don't retry, move to next provider immediately)
                is_payment_required = "402" in error_str or "Payment Required" in error_str
                is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str

                if is_payment_required:
                    logger.warning(f"{provider_name}: No credits/quota - skipping for {int(LLMTool._FAILURE_COOLDOWN)}s")
                    LLMTool._failed_providers[provider_name] = now
                    errors.append(f"{provider_name}: Payment required (no credits)")
                    continue

                if is_rate_limit:
                    logger.warning(f"{provider_name}: Rate limited - skipping for {int(LLMTool._FAILURE_COOLDOWN)}s")
                    LLMTool._failed_providers[provider_name] = now
                    errors.append(f"{provider_name}: Rate limited")
                    continue

                # Timeout errors get a shorter cooldown (60s) to avoid wasting time
                is_timeout = "timed out" in error_str.lower() or "timeout" in error_str.lower()
                if is_timeout:
                    logger.warning(f"{provider_name}: Timed out - skipping for 60s")
                    LLMTool._failed_providers[provider_name] = now - (LLMTool._FAILURE_COOLDOWN - 60)  # 60s effective cooldown
                    errors.append(f"{provider_name}: Timeout")
                    continue

                # For other errors, log and move on (but don't add cooldown for transient errors)
                logger.warning(f"{provider_name} failed: {e}")
                errors.append(f"{provider_name}: {e}")
                continue

        # All providers failed - provide helpful error
        error_summary = "; ".join(errors[-3:])  # Last 3 errors
        raise RuntimeError(f"All LLM providers failed. Errors: {error_summary}")

    def _get_provider_chain(self) -> List[tuple]:
        """Get ordered list of available providers.

        Priority: NVIDIA → Ollama → OpenRouter → Gemini.
        Exception: force_groq puts Groq first for deep reasoning tasks.
        """
        providers = []

        # Groq forced first for deep reasoning tasks only
        if self.force_groq and self._groq_configured:
            providers.append(("Groq", self._call_groq))

        # NVIDIA FIRST (Kimi K2.5 via NVIDIA AI Endpoints)
        if self._nvidia_configured:
            providers.append(("NVIDIA", self._call_nvidia))

        # Ollama SECOND (local, free, no rate limits)
        if self.settings.use_ollama:
            providers.append(("Ollama", self._call_ollama))

        # Cloud providers as fallback
        if self._openrouter_configured:
            providers.append(("OpenRouter", self._call_openrouter))

        if self._gemini_configured:
            providers.append(("Gemini", self._call_gemini))

        return providers
    
    async def _call_nvidia(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> Optional[str]:
        """Call NVIDIA AI Endpoints (OpenAI-compatible API)."""
        url = f"{self.settings.nvidia_base_url}/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.settings.nvidia_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.settings.nvidia_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(url, json=payload, headers=headers)

                if response.status_code != 200:
                    body = response.text[:500]
                    raise RuntimeError(
                        f"NVIDIA API {response.status_code}: {body}"
                    )

                data = response.json()

                # Handle NVIDIA error responses that return 200 with error body
                if "error" in data:
                    raise RuntimeError(f"NVIDIA API error: {data['error']}")

                choices = data.get("choices", [])
                if not choices:
                    raise ValueError(f"NVIDIA returned no choices: {str(data)[:300]}")

                content = choices[0].get("message", {}).get("content", "")
                if not content:
                    raise ValueError("NVIDIA returned empty content")

                logger.info(f"NVIDIA: Got {len(content)} chars response")
                return content

        except httpx.TimeoutException:
            raise RuntimeError("NVIDIA API request timed out after 45s")
        except httpx.ConnectError as e:
            raise RuntimeError(f"NVIDIA API connection failed: {e}")
        except RuntimeError:
            raise  # Re-raise our own errors with detail
        except Exception as e:
            raise RuntimeError(f"NVIDIA API unexpected error ({type(e).__name__}): {e}")

    async def _call_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> Optional[str]:
        """Call Ollama using the official Python client."""
        import asyncio

        try:
            # First check if Ollama is reachable with a short timeout
            logger.info(f"Ollama: Checking connection to {self.settings.ollama_base_url}...")
            async with httpx.AsyncClient(timeout=10.0) as http_client:
                try:
                    health_check = await http_client.get(f"{self.settings.ollama_base_url}/api/tags")
                    if health_check.status_code != 200:
                        raise ConnectionError(f"Ollama not responding (status {health_check.status_code})")
                except httpx.ConnectError:
                    raise ConnectionError(
                        f"Cannot connect to Ollama at {self.settings.ollama_base_url}. "
                        f"Is Ollama running? Try: ollama serve"
                    )
                except httpx.TimeoutException:
                    raise ConnectionError(
                        f"Ollama health check timed out at {self.settings.ollama_base_url}. "
                        f"Ollama might be starting up or overloaded."
                    )

                # Check if the model exists
                models = health_check.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                target_model = self.settings.ollama_model.split(":")[0]
                if target_model not in model_names and f"{target_model}:latest" not in [m.get("name") for m in models]:
                    available = ", ".join(model_names[:5]) if model_names else "none"
                    raise ValueError(
                        f"Model '{self.settings.ollama_model}' not found. "
                        f"Available: {available}. Try: ollama pull {self.settings.ollama_model}"
                    )

            logger.info(f"Ollama: Model {self.settings.ollama_model} found, sending request...")

            # Use longer timeout for actual generation (300 seconds = 5 minutes)
            client = OllamaAsyncClient(
                host=self.settings.ollama_base_url,
                timeout=300.0,  # Increased from 180 to 300 seconds
            )

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = await client.chat(
                model=self.settings.ollama_model,
                messages=messages,
                options={"temperature": temperature, "num_predict": max_tokens},
            )

            content = response.message.content if response and response.message else None
            if not content:
                raise ValueError("Ollama returned empty response")

            logger.info(f"Ollama: Got {len(content)} chars response")
            return content

        except ConnectionError:
            raise  # Re-raise connection errors with their detailed message
        except httpx.TimeoutException:
            raise TimeoutError(
                f"Ollama request timed out after 300s. Model '{self.settings.ollama_model}' may be too slow. "
                f"Try a smaller model like 'mistral' or 'phi'."
            )
        except Exception as e:
            error_msg = str(e) if str(e) else type(e).__name__
            raise RuntimeError(f"Ollama error: {error_msg}")

    async def _call_openrouter(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> Optional[str]:
        """Call OpenRouter API (OpenAI-compatible)."""
        url = "https://openrouter.ai/api/v1/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/coherent-market-insights",
            "X-Title": "CMI Lead Agent"
        }

        payload = {
            "model": self.settings.openrouter_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                raise RuntimeError(f"OpenRouter API {response.status_code}: {response.text[:500]}")

            data = response.json()
            return data["choices"][0]["message"]["content"]
    
    async def _call_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> Optional[str]:
        """Call Gemini API."""
        if NEW_GENAI and self._gemini_client:
            # New google-genai package — use async client with proper system_instruction
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            if system_prompt:
                config.system_instruction = system_prompt

            response = await self._gemini_client.aio.models.generate_content(
                model=self.settings.gemini_model,
                contents=prompt,
                config=config,
            )
            return response.text
        else:
            # Old google-generativeai package
            model = genai.GenerativeModel(
                self.settings.gemini_model,
                system_instruction=system_prompt if system_prompt else None,
            )

            generation_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )

            response = await model.generate_content_async(
                prompt,
                generation_config=generation_config
            )

            return response.text
    
    async def _call_groq(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> Optional[str]:
        """Call Groq API."""
        if not self._groq_client:
            return None

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        model = self.settings.groq_model
        call_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1,
            "stream": False,
        }

        # Add reasoning_effort for 120B model
        if "120b" in model.lower() or "oss" in model.lower():
            call_params["reasoning_effort"] = "high"

        completion = await self._groq_client.chat.completions.create(**call_params)
        response_content = completion.choices[0].message.content or ""

        # For 120B model, fallback to reasoning field if content is empty
        if not response_content:
            message = completion.choices[0].message
            if hasattr(message, 'reasoning') and message.reasoning:
                response_content = message.reasoning

        logger.debug(f"Groq response: {len(response_content)} chars")
        return response_content
    
    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema_hint: Optional[str] = None,
        pydantic_model: Optional[Type] = None,
        required_keys: Optional[set] = None,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON response with validation and retry.

        V2: Adds retry loop on parse failure or validation failure.
        When pydantic_model is provided, validates output against it and feeds
        validation errors back to the LLM on retry. When required_keys is
        provided, checks for their presence.

        Args:
            prompt: User prompt
            system_prompt: Optional system instructions
            schema_hint: Optional JSON schema description
            pydantic_model: Optional Pydantic BaseModel class for validation
            required_keys: Optional set of keys that must be present in result
            max_retries: Max retry attempts (default from env LLM_JSON_MAX_RETRIES)

        Returns:
            Parsed JSON dictionary. On total failure, returns dict with "error" key.
        """
        if max_retries is None:
            try:
                max_retries = int(getattr(get_settings(), 'llm_json_max_retries', 2))
            except Exception:
                max_retries = 2

        json_instruction = """
You must respond with valid JSON only. No markdown, no explanation, just the JSON object.
Do not wrap the response in ```json``` code blocks.
"""
        if schema_hint:
            json_instruction += f"\nExpected format:\n{schema_hint}"

        full_system = (system_prompt or "") + "\n" + json_instruction
        current_prompt = prompt
        last_error = None

        for attempt in range(1, max_retries + 1):
            t0 = time.time()
            try:
                response = await self.generate(
                    prompt=current_prompt,
                    system_prompt=full_system.strip(),
                    temperature=0.3,
                    json_mode=True
                )

                result = self._parse_json_response(response)
                elapsed_ms = int((time.time() - t0) * 1000)

                # Check for parse error
                if isinstance(result, dict) and "error" in result and len(result) <= 2:
                    last_error = result.get("error", "JSON parse failed")
                    logger.warning(
                        f"generate_json attempt {attempt}/{max_retries}: "
                        f"parse error: {last_error} ({elapsed_ms}ms)"
                    )
                    if attempt < max_retries:
                        current_prompt = (
                            f"{prompt}\n\n"
                            f"IMPORTANT: Your previous response was not valid JSON. "
                            f"Error: {last_error}. Please fix and respond with ONLY valid JSON."
                        )
                        continue
                    return result

                # Check required keys
                if required_keys:
                    missing = required_keys - set(result.keys())
                    if missing:
                        last_error = f"Missing required keys: {missing}"
                        logger.warning(
                            f"generate_json attempt {attempt}/{max_retries}: "
                            f"{last_error} ({elapsed_ms}ms)"
                        )
                        if attempt < max_retries:
                            current_prompt = (
                                f"{prompt}\n\n"
                                f"IMPORTANT: Your previous response was missing these required fields: "
                                f"{', '.join(missing)}. Include ALL required fields."
                            )
                            continue

                # Pydantic model validation
                if pydantic_model is not None:
                    try:
                        pydantic_model.model_validate(result)
                        logger.debug(
                            f"generate_json: Pydantic validation passed "
                            f"(attempt {attempt}, {elapsed_ms}ms)"
                        )
                    except Exception as ve:
                        last_error = str(ve)[:500]
                        logger.warning(
                            f"generate_json attempt {attempt}/{max_retries}: "
                            f"validation error: {last_error[:200]} ({elapsed_ms}ms)"
                        )
                        if attempt < max_retries:
                            current_prompt = (
                                f"{prompt}\n\n"
                                f"IMPORTANT: Your previous JSON failed validation. "
                                f"Errors: {last_error[:300]}. Fix these issues."
                            )
                            continue
                        # On final attempt, return what we have (validators will coerce)

                logger.debug(
                    f"generate_json: success (attempt {attempt}, "
                    f"provider={self.last_provider}, {elapsed_ms}ms)"
                )
                return result

            except RuntimeError as e:
                # All providers failed — no point retrying
                last_error = str(e)
                logger.error(f"generate_json: provider failure on attempt {attempt}: {e}")
                return {"error": str(e)}

            except Exception as e:
                elapsed_ms = int((time.time() - t0) * 1000)
                last_error = str(e)
                logger.warning(
                    f"generate_json attempt {attempt}/{max_retries} "
                    f"exception: {e} ({elapsed_ms}ms)"
                )
                if attempt >= max_retries:
                    return {"error": f"Failed after {max_retries} attempts: {last_error}"}

        return {"error": f"Exhausted {max_retries} retries. Last: {last_error}"}

    @staticmethod
    def validate_json_structure(
        result: Dict[str, Any], required_keys: set
    ) -> tuple:
        """
        Lightweight structural validation without Pydantic.

        Returns:
            (is_valid: bool, missing_keys: List[str])
        """
        if not isinstance(result, dict):
            return False, list(required_keys)
        missing = required_keys - set(result.keys())
        return len(missing) == 0, list(missing)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling common issues."""
        cleaned = response.strip()

        # Remove markdown code blocks if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\n?', '', cleaned)
            cleaned = re.sub(r'\n?```$', '', cleaned)
            cleaned = cleaned.strip()

        # Extract the JSON structure (array or object) from the response
        json_str = self._extract_json_string(cleaned)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Common issue: LLMs (especially Ollama) insert literal newlines/tabs
            # inside JSON string values. Try strict=False which allows control chars.
            try:
                return json.loads(json_str, strict=False)
            except json.JSONDecodeError:
                pass
            # Last resort: strip control characters from inside strings
            sanitized = re.sub(r'[\x00-\x1f\x7f]', ' ', json_str)
            try:
                return json.loads(sanitized)
            except json.JSONDecodeError as e2:
                logger.error(f"Failed to parse JSON: {e2}\nResponse: {json_str[:500]}")
                return {"error": "Failed to parse JSON", "raw": json_str[:500]}

    @staticmethod
    def _extract_json_string(text: str) -> str:
        """Extract the outermost JSON object or array from text using bracket counting."""
        # Determine which delimiter appears first and try that one first
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
            # Try to repair by closing open brackets.
            truncated = text[start:]
            return LLMTool._repair_truncated_json(truncated)
        return text

    @staticmethod
    def _repair_truncated_json(text: str) -> str:
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
    
    async def generate_list(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate a list of JSON objects."""
        json_instruction = """
You must respond with a valid JSON array only. No markdown, no explanation.
Each item in the array should be a JSON object.
Do not wrap the response in ```json``` code blocks.
"""
        full_system = (system_prompt or "") + "\n" + json_instruction
        
        response = await self.generate(
            prompt=prompt,
            system_prompt=full_system.strip(),
            temperature=0.3,
            json_mode=True
        )
        
        result = self._parse_json_response(response)
        
        # If we got a dict with a list inside, extract it
        if isinstance(result, dict):
            for key in ["items", "results", "data", "companies", "contacts", "trends"]:
                if key in result and isinstance(result[key], list):
                    return result[key]
            return [result]
        
        return result if isinstance(result, list) else [result]
    
    def _get_mock_response(self, prompt: str, json_mode: bool) -> str:
        """Return mock response for testing."""
        import hashlib

        prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16) % 5
        prompt_lower = prompt.lower()

        if json_mode or "json" in prompt_lower:
            # Check for companies FIRST (company prompts may also contain "trend")
            if ("compan" in prompt_lower and "find" in prompt_lower) or "real" in prompt_lower or "extract" in prompt_lower:
                return self._get_mock_company_response(prompt_lower)
            if "synthesize" in prompt_lower or "cluster" in prompt_lower:
                # TrendSynthesizer mock response - uses different field names
                mock_synth_trends = [
                    {
                        "trend_title": "RBI Mandates Stricter KYC for Digital Lenders",
                        "trend_summary": "The Reserve Bank of India has announced comprehensive new KYC requirements affecting over 500 fintech lenders across India. Companies must achieve full compliance within 90 days or face regulatory penalties. This represents the most significant regulatory shift in digital lending since 2021.",
                        "trend_type": "regulation",
                        "severity": "high",
                        "key_entities": ["RBI", "Lendingkart", "Capital Float", "ZestMoney", "Paytm Lending"],
                        "key_facts": ["90-day compliance deadline", "Affects 500+ fintech lenders", "New video KYC requirements"],
                        "key_numbers": ["500+ lenders affected", "90 days deadline", "Rs 50 lakh penalty"],
                        "primary_sectors": ["fintech", "banking", "nbfc"],
                        "secondary_sectors": ["regtech", "identity_verification"],
                        "affected_regions": ["Mumbai", "Bangalore", "Delhi NCR"],
                        "is_national": True,
                        "lifecycle_stage": "emerging",
                        "confidence_explanation": "Multiple tier-1 sources reporting with official RBI circular reference"
                    },
                    {
                        "trend_title": "Zepto Raises $200M at $5B Valuation",
                        "trend_summary": "Quick commerce startup Zepto has closed a massive $200 million funding round, valuing the company at $5 billion. The funds will be used to expand dark store network to 50 new cities, intensifying competition with Blinkit and Swiggy Instamart.",
                        "trend_type": "funding",
                        "severity": "high",
                        "key_entities": ["Zepto", "Blinkit", "Swiggy Instamart", "BigBasket"],
                        "key_facts": ["$200M Series F round", "Expansion to 50 cities", "10-minute delivery focus"],
                        "key_numbers": ["$200 million raised", "$5 billion valuation", "50 new cities"],
                        "primary_sectors": ["retail", "logistics", "ecommerce"],
                        "secondary_sectors": ["cold_chain", "warehousing"],
                        "affected_regions": ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"],
                        "is_national": True,
                        "lifecycle_stage": "growing",
                        "confidence_explanation": "Official company announcement with investor confirmation"
                    },
                    {
                        "trend_title": "Government Approves 3 New Semiconductor Fabs",
                        "trend_summary": "The Union Cabinet has approved Rs 1.26 lakh crore investment for setting up three new semiconductor fabrication plants under the India Semiconductor Mission. Tata Electronics and Vedanta are key beneficiaries, marking a major milestone in India's electronics manufacturing ambitions.",
                        "trend_type": "policy",
                        "severity": "high",
                        "key_entities": ["Tata Electronics", "Vedanta", "Ministry of Electronics", "India Semiconductor Mission"],
                        "key_facts": ["3 new fabs approved", "Rs 1.26 lakh crore investment", "Part of Make in India initiative"],
                        "key_numbers": ["Rs 1.26 lakh crore", "3 fabs", "100,000 jobs expected"],
                        "primary_sectors": ["manufacturing", "electronics", "semiconductors"],
                        "secondary_sectors": ["chemicals", "equipment"],
                        "affected_regions": ["Gujarat", "Karnataka", "Tamil Nadu"],
                        "is_national": True,
                        "lifecycle_stage": "emerging",
                        "confidence_explanation": "Official government announcement with cabinet approval"
                    }
                ]
                return json.dumps(mock_synth_trends[prompt_hash % 3])
            elif "trend" in prompt_lower or "news" in prompt_lower:
                mock_trends = [
                    {
                        "trend_title": "RBI Mandates Stricter KYC for Digital Lenders",
                        "summary": "Reserve Bank of India announced new KYC requirements affecting 500+ fintech lenders. Companies must comply within 90 days.",
                        "severity": "high",
                        "industries_affected": ["Fintech", "Digital Lending", "NBFC", "Banking"],
                        "keywords": ["RBI", "KYC", "digital lending", "compliance", "fintech regulation"],
                        "trend_type": "regulation",
                        "urgency": "Immediate compliance required - 90 day deadline"
                    },
                    {
                        "trend_title": "Zepto Raises $200M, Valued at $5B",
                        "summary": "Quick commerce startup Zepto closed massive funding round, plans expansion to 50 new cities. Competitor pressure intensifies.",
                        "severity": "high",
                        "industries_affected": ["Quick Commerce", "E-commerce", "Logistics", "Retail"],
                        "keywords": ["Zepto", "quick commerce", "funding", "dark stores", "Blinkit", "Instamart"],
                        "trend_type": "funding",
                        "urgency": "Competitors need to respond to market pressure"
                    },
                    {
                        "trend_title": "Swiggy Announces 400 Employee Layoffs",
                        "summary": "Food delivery giant Swiggy cuts 400 jobs in restructuring ahead of IPO. Focus shifts to profitability over growth.",
                        "severity": "medium",
                        "industries_affected": ["Food Tech", "Gig Economy", "HR Tech", "Recruitment"],
                        "keywords": ["Swiggy", "layoffs", "IPO", "restructuring", "food delivery"],
                        "trend_type": "layoffs",
                        "urgency": "Talent available in market, HR solutions needed"
                    },
                    {
                        "trend_title": "Government Approves 3 New Semiconductor Fabs",
                        "summary": "Cabinet clears Rs 1.26 lakh crore investment for semiconductor manufacturing. Major opportunity for electronics ecosystem.",
                        "severity": "high",
                        "industries_affected": ["Semiconductors", "Electronics", "Manufacturing", "IT Hardware"],
                        "keywords": ["semiconductor", "PLI scheme", "electronics manufacturing", "chip fab"],
                        "trend_type": "policy",
                        "urgency": "First-mover advantage in emerging ecosystem"
                    },
                    {
                        "trend_title": "Reliance Jio Partners with NVIDIA for AI Cloud",
                        "summary": "Jio and NVIDIA announce strategic partnership for enterprise AI infrastructure in India. New AI cloud services launching.",
                        "severity": "high",
                        "industries_affected": ["Cloud Computing", "AI/ML", "Enterprise IT", "Data Centers"],
                        "keywords": ["Jio", "NVIDIA", "AI cloud", "enterprise AI", "GPU computing"],
                        "trend_type": "partnership",
                        "urgency": "Enterprises need AI strategy now"
                    }
                ]
                return json.dumps(mock_trends[prompt_hash])
            elif "impact" in prompt_lower or "consultant" in prompt_lower or "direct" in prompt_lower:
                mock_impacts = [
                    {
                        "direct_impact": ["Digital Lending NBFCs", "Fintech Payment Apps", "P2P Lending Platforms", "Buy Now Pay Later Companies"],
                        "direct_impact_reasoning": "RBI's new KYC norms directly mandate these companies to overhaul their customer verification processes. They face a 90-day compliance deadline with penalties for non-compliance. Most mid-size lenders lack dedicated compliance teams.",
                        "indirect_impact": ["RegTech Software Providers", "Identity Verification Services", "Compliance Consulting Firms", "Customer Onboarding Platforms"],
                        "indirect_impact_reasoning": "As lenders scramble to comply, they'll need technology and consulting support. RegTech demand will surge. KYC verification vendors will see increased volumes. Compliance consultants will be hired for gap assessments.",
                        "additional_verticals": ["Banking Software Vendors", "Data Analytics Firms", "Cybersecurity Companies", "Legal Advisory Firms", "Document Management Solutions", "API Integration Specialists"],
                        "additional_verticals_reasoning": "The KYC overhaul requires system upgrades (software vendors), data handling changes (analytics), security enhancements (cybersecurity), legal review (law firms), and document workflows (DMS). API specialists needed for Aadhaar/PAN integration.",
                        "positive_sectors": ["Fintech Lenders", "RegTech", "Compliance Consulting", "Identity Verification"],
                        "negative_sectors": ["Unorganized Moneylenders"],
                        "business_opportunities": [
                            "KYC compliance gap assessment for mid-size NBFCs",
                            "Regulatory landscape mapping for fintech lenders",
                            "Cost-benefit analysis of compliance technology options",
                            "Benchmarking study of KYC best practices",
                            "Vendor evaluation for identity verification solutions"
                        ],
                        "relevant_services": ["Market Monitoring", "Consulting and Advisory Services", "Technology Research"],
                        "target_roles": ["CEO", "Chief Compliance Officer", "VP Operations", "Director Risk Management"],
                        "pitch_angle": "Navigate RBI's KYC mandate with expert compliance guidance",
                        "reasoning": "90-day deadline creates urgency. Mid-size lenders need external expertise to avoid penalties and operational disruption."
                    },
                    {
                        "direct_impact": ["Quick Commerce Startups", "Grocery Delivery Apps", "Dark Store Operators", "Last-Mile Logistics"],
                        "direct_impact_reasoning": "Zepto's $200M funding and 50-city expansion directly threatens competitors like Blinkit, Instamart, and BigBasket. They must respond with their own expansion or differentiation strategies.",
                        "indirect_impact": ["Cold Chain Infrastructure", "Warehouse Real Estate", "Gig Economy Platforms", "Packaging Suppliers"],
                        "indirect_impact_reasoning": "More dark stores = more cold storage demand. Warehouse rentals in urban areas will spike. Delivery fleet demand increases (gig platforms). Packaging for quick deliveries needs to scale.",
                        "additional_verticals": ["FMCG Brands", "Local Kirana Tech", "Retail Analytics", "Urban Planning Consultants", "Electric Vehicle Logistics", "Micro-fulfillment Technology"],
                        "additional_verticals_reasoning": "FMCG brands need quick commerce strategy. Kirana stores need tech to compete. Analytics firms help optimize dark store locations. EV logistics for sustainable delivery. Micro-fulfillment tech enables speed.",
                        "positive_sectors": ["Quick Commerce", "Cold Chain", "Logistics Tech", "Warehouse Real Estate"],
                        "negative_sectors": ["Traditional Retail", "Kirana Stores without Tech"],
                        "business_opportunities": [
                            "Competitive intelligence on quick commerce landscape",
                            "Dark store location strategy and feasibility study",
                            "Supply chain optimization for 10-minute delivery",
                            "Market entry strategy for regional quick commerce players",
                            "FMCG brand strategy for quick commerce channel"
                        ],
                        "relevant_services": ["Competitive Intelligence", "Market Intelligence", "Industry Analysis"],
                        "target_roles": ["CEO", "Chief Strategy Officer", "VP Supply Chain", "Director Business Development"],
                        "pitch_angle": "Win the quick commerce battle with strategic intelligence",
                        "reasoning": "Funding war intensifies. Mid-size players need competitive insights to survive or find their niche."
                    },
                    {
                        "direct_impact": ["Semiconductor Manufacturing", "Electronics Assembly", "PCB Manufacturers", "Chip Design Companies"],
                        "direct_impact_reasoning": "Rs 1.26 lakh crore investment directly benefits semiconductor fabs (Tata, Vedanta) and creates demand for local component suppliers. Electronics assemblers can now source chips domestically.",
                        "indirect_impact": ["Electronics Contract Manufacturing", "Consumer Electronics Brands", "Automotive Electronics", "Telecom Equipment"],
                        "indirect_impact_reasoning": "Local chip supply enables contract manufacturers to reduce import dependency. Consumer electronics brands can 'Made in India' premium. Auto electronics benefits from local semiconductor supply. 5G equipment manufacturing becomes viable.",
                        "additional_verticals": ["Specialty Chemicals for Semiconductors", "Cleanroom Equipment", "Industrial Gases", "Water Treatment for Fabs", "Skilled Workforce Training", "Logistics for Sensitive Components"],
                        "additional_verticals_reasoning": "Chip fabs need ultra-pure chemicals, cleanroom infrastructure, specialty gases, treated water systems. Training institutes will prepare workforce. Sensitive component logistics will emerge as a specialty.",
                        "positive_sectors": ["Semiconductors", "Electronics Manufacturing", "Chemical Suppliers", "Industrial Equipment"],
                        "negative_sectors": ["Chip Importers", "Trading Companies"],
                        "business_opportunities": [
                            "Semiconductor ecosystem supplier identification",
                            "Market entry feasibility for specialty chemical companies",
                            "Workforce skill gap analysis for chip manufacturing",
                            "Supply chain mapping for electronics components",
                            "Competitive analysis of global semiconductor players entering India"
                        ],
                        "relevant_services": ["Industry Analysis", "Procurement Intelligence", "Cross Border Expansion"],
                        "target_roles": ["CEO", "VP Manufacturing", "Chief Procurement Officer", "Director Strategy"],
                        "pitch_angle": "Capitalize on India's semiconductor revolution",
                        "reasoning": "Historic opportunity for electronics ecosystem. Companies need market intelligence to position themselves in the emerging value chain."
                    }
                ]
                return json.dumps(mock_impacts[prompt_hash % 3])
            elif "contact" in prompt_lower or "person" in prompt_lower:
                return json.dumps({
                    "person_name": "Rahul Sharma",
                    "role": "CTO",
                    "linkedin_url": "https://linkedin.com/in/rahul-sharma"
                })
            elif "email" in prompt_lower or "outreach" in prompt_lower or "pitch" in prompt_lower:
                mock_emails = [
                    {
                        "subject": "RBI's New KYC Norms - Impact Assessment for Lenders",
                        "body": f"Hi there,\n\nI noticed the RBI's new KYC mandate and thought of your company. With the 90-day compliance deadline, many fintech lenders are scrambling to understand the full impact on their operations.\n\nAt Coherent Market Insights, we've been tracking this regulatory shift closely. We can help with:\n\n• Regulatory compliance landscape assessment\n• Cost-impact analysis of new KYC requirements\n• Benchmarking against industry best practices\n\nWould you be open to a 15-minute call to discuss how we might support your compliance strategy?\n\nBest regards,\nCoherent Market Insights Team"
                    },
                    {
                        "subject": "Quick Commerce Battle - Competitive Intelligence Opportunity",
                        "body": f"Hi there,\n\nWith Zepto's $200M raise and aggressive expansion plans, the quick commerce landscape is shifting rapidly. I thought this might be relevant for your strategic planning.\n\nAt Coherent Market Insights, we help companies navigate competitive disruptions through:\n\n• Competitor profiling and strategy analysis\n• Market share tracking and benchmarking\n• Go-to-market strategy recommendations\n\nWould a 15-minute call be useful to explore how we can help you stay ahead of the competition?\n\nBest regards,\nCoherent Market Insights Team"
                    },
                    {
                        "subject": "Semiconductor Policy - Supply Chain Opportunity Analysis",
                        "body": f"Hi there,\n\nThe government's Rs 1.26 lakh crore semiconductor investment opens significant opportunities for the electronics value chain. I wanted to reach out given your position in the industry.\n\nAt Coherent Market Insights, we specialize in:\n\n• Supply chain opportunity mapping\n• Supplier identification and profiling\n• Market entry feasibility studies\n\nWould you be interested in a 15-minute discussion about how to capitalize on this policy shift?\n\nBest regards,\nCoherent Market Insights Team"
                    }
                ]
                return json.dumps(mock_emails[prompt_hash % 3])
        
        return "Mock LLM response for testing purposes."
    
    def _get_mock_company_response(self, prompt_lower: str) -> str:
        """Return mock company data based on sector keywords in prompt."""
        # Sector keyword mappings to company data
        sector_companies = {
            "oil_energy": [
                {"company_name": "Petrosol Energy Services", "company_size": "mid", "industry": "Oil Equipment", "website": "https://petrosol.in", "description": "Oil field equipment supplier, 180 employees", "intent_signal": "Facing margin pressure", "reason_relevant": "Needs procurement intelligence"},
                {"company_name": "Gujarat Oilfield Services", "company_size": "mid", "industry": "Oil Services", "website": "https://gosindia.com", "description": "Oilfield services, 220 employees", "intent_signal": "Restructuring operations", "reason_relevant": "Needs strategic consulting"},
            ],
            "fintech": [
                {"company_name": "Lendingkart", "company_size": "mid", "industry": "Fintech", "website": "https://lendingkart.com", "description": "SME lending platform, 150 employees", "intent_signal": "Compliance overhaul needed", "reason_relevant": "Affected by RBI KYC norms"},
                {"company_name": "Capital Float", "company_size": "mid", "industry": "Fintech", "website": "https://capitalfloat.com", "description": "Digital lending, 200 employees", "intent_signal": "Seeking regulatory guidance", "reason_relevant": "Must comply with regulations"},
            ],
            "logistics": [
                {"company_name": "Country Delight", "company_size": "mid", "industry": "Quick Commerce", "website": "https://countrydelight.in", "description": "Farm-fresh delivery, 250 employees", "intent_signal": "Expanding to new cities", "reason_relevant": "Quick delivery competitor"},
                {"company_name": "Delhivery Express", "company_size": "mid", "industry": "Logistics", "website": "https://delhivery.com", "description": "Last-mile logistics, 280 employees", "intent_signal": "Supply chain optimization", "reason_relevant": "Logistics opportunity"},
            ],
            "electronics": [
                {"company_name": "VVDN Technologies", "company_size": "mid", "industry": "Electronics Manufacturing", "website": "https://vvdntech.com", "description": "Electronics design, 250 employees", "intent_signal": "Entering semiconductor supply chain", "reason_relevant": "Semiconductor ecosystem"},
                {"company_name": "Syrma SGS Technology", "company_size": "mid", "industry": "Electronics", "website": "https://syrmasgs.com", "description": "EMS provider, 200 employees", "intent_signal": "Diversifying components", "reason_relevant": "Sourcing opportunity"},
            ],
            "hr": [
                {"company_name": "Xpheno", "company_size": "mid", "industry": "HR Tech", "website": "https://xpheno.com", "description": "Specialist staffing, 120 employees", "intent_signal": "Growing talent acquisition", "reason_relevant": "Talent opportunity"},
                {"company_name": "PeopleStrong", "company_size": "mid", "industry": "HR Tech", "website": "https://peoplestrong.com", "description": "HR technology, 250 employees", "intent_signal": "Workforce optimization", "reason_relevant": "HR tech needs"},
            ],
            "regtech": [
                {"company_name": "Signzy", "company_size": "mid", "industry": "RegTech", "website": "https://signzy.com", "description": "Digital KYC solutions, 150 employees", "intent_signal": "KYC demand surge", "reason_relevant": "Compliance solutions"},
                {"company_name": "IDfy", "company_size": "mid", "industry": "RegTech", "website": "https://idfy.com", "description": "Identity verification, 180 employees", "intent_signal": "Expanding verification services", "reason_relevant": "Compliance provider"},
            ],
            "default": [
                {"company_name": "Moglix", "company_size": "mid", "industry": "B2B Commerce", "website": "https://moglix.com", "description": "Industrial B2B marketplace, 280 employees", "intent_signal": "Seeking procurement intelligence", "reason_relevant": "Supply chain needs"},
                {"company_name": "OfBusiness", "company_size": "mid", "industry": "B2B Commerce", "website": "https://ofbusiness.com", "description": "B2B raw materials, 250 employees", "intent_signal": "Expanding supplier network", "reason_relevant": "Procurement opportunity"},
            ],
        }

        # Match sector based on keywords
        if any(kw in prompt_lower for kw in ["oil", "energy", "fuel", "petro"]):
            companies = sector_companies["oil_energy"]
        elif any(kw in prompt_lower for kw in ["fintech", "lending", "nbfc"]):
            companies = sector_companies["fintech"]
        elif any(kw in prompt_lower for kw in ["logistics", "delivery", "commerce"]):
            companies = sector_companies["logistics"]
        elif any(kw in prompt_lower for kw in ["semiconductor", "electronics", "manufacturing"]):
            companies = sector_companies["electronics"]
        elif any(kw in prompt_lower for kw in ["hr", "recruitment", "talent"]):
            companies = sector_companies["hr"]
        elif any(kw in prompt_lower for kw in ["regtech", "compliance"]):
            companies = sector_companies["regtech"]
        else:
            companies = sector_companies["default"]

        return json.dumps(companies)
    
    async def check_ollama_health(self) -> bool:
        """Check if Ollama is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.settings.ollama_base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
    
    async def get_provider_status(self) -> Dict[str, bool]:
        """Get status of all LLM providers."""
        nvidia_ok = self._nvidia_configured
        ollama_ok = await self.check_ollama_health() if self.settings.use_ollama else False
        gemini_ok = self._gemini_configured
        groq_ok = self._groq_configured
        openrouter_ok = self._openrouter_configured

        return {
            "nvidia": nvidia_ok,
            "ollama": ollama_ok,
            "gemini": gemini_ok,
            "groq": groq_ok,
            "openrouter": openrouter_ok,
            "any_available": nvidia_ok or ollama_ok or gemini_ok or groq_ok or openrouter_ok,
        }
