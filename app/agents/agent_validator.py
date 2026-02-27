"""
Agent invocation validator — detects silent agent failures.

The core problem: pydantic-ai agents sometimes return empty schema defaults
without calling any tools, because the LLM thinks it has already answered.
This is different from a legitimate "no results found" outcome.

Patterns:
  1. tool_calls=0, errors=0  → agent forgot to try → retry with forcing instruction
  2. tool_calls>0, results=0 → tried, found nothing → legitimate (propagate up)
  3. tool_calls=0, errors>0  → API down → don't retry, let provider_health handle

The forcing instruction appended on retry:
  "You MUST call at least one tool before responding."
This is the simplest nudge that works reliably with function-calling LLMs.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_FORCING_INSTRUCTION = (
    "\n\n[RETRY ENFORCEMENT] You MUST invoke at least one tool before producing "
    "your final answer. Do NOT return empty results without tool calls. "
    "The pipeline cannot proceed without you taking action."
)


@dataclass
class InvocationCheck:
    """Result of checking whether an agent actually did work."""
    agent_name: str
    is_valid: bool
    should_retry: bool
    reason: str
    tool_calls: int = 0
    result_count: int = 0
    error_count: int = 0


def check_invocation(
    agent_name: str,
    tool_calls: int,
    result_count: int,
    error_count: int = 0,
) -> InvocationCheck:
    """
    Classify whether the agent invocation was valid or a silent failure.

    Args:
        agent_name:    Name for logging
        tool_calls:    Number of tool calls made by the agent
        result_count:  Number of useful outputs produced
        error_count:   Number of exception-level errors (API failures)
    """
    base = dict(agent_name=agent_name, tool_calls=tool_calls,
                result_count=result_count, error_count=error_count)

    # Pattern 1: Agent gave up without trying — retry with instruction
    if tool_calls == 0 and error_count == 0:
        return InvocationCheck(
            is_valid=False, should_retry=True,
            reason=f"{agent_name}: 0 tools called, 0 errors — agent didn't try",
            **base,
        )

    # Pattern 2: API errors dominated — don't retry, circuit breaker handles it
    if error_count > 0 and tool_calls == 0:
        return InvocationCheck(
            is_valid=False, should_retry=False,
            reason=f"{agent_name}: {error_count} API errors — provider issue",
            **base,
        )

    # Pattern 3: Tools ran, nothing found — legitimate result
    if tool_calls > 0 and result_count == 0:
        return InvocationCheck(
            is_valid=True, should_retry=False,
            reason=f"{agent_name}: {tool_calls} tools ran, 0 results (legit)",
            **base,
        )

    # Pattern 4: Normal success
    return InvocationCheck(
        is_valid=True, should_retry=False,
        reason=f"{agent_name}: {tool_calls} tools, {result_count} results [OK]",
        **base,
    )


def with_forcing_instruction(system_prompt: str) -> str:
    """Append mandatory tool-call instruction to a system prompt for retry."""
    return system_prompt + _FORCING_INSTRUCTION
