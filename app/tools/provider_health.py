"""
Provider health tracking — circuit breaker for LLM API failures.

When Groq rate-limits at 2pm, at 2:05pm the system should try Ollama instead.
This module stores failure history in a lightweight JSON file and implements
exponential backoff: 5min → 15min → 60min → 24h.

Schema:
  {provider_name: {last_failure_time, failure_count, status, backoff_until, last_success}}

Persisted at: app/data/provider_health.json
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_HEALTH_DB = Path("app/data/provider_health.json")
# Minimal backoff: 15s → 30s → 45s → 60s (seconds-based now, not minutes)
# Rate limits reset in 60s. Long backoffs kill the pipeline.
_BACKOFF_MINUTES = [1, 1, 2, 3]  # Kept for backwards compat but barely used


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ProviderStatus:
    provider_name: str
    failure_count: int = 0
    status: str = "healthy"          # healthy | degraded | broken
    last_failure_time: Optional[str] = None   # ISO UTC
    backoff_until: Optional[str] = None       # ISO UTC
    last_successful_call: Optional[str] = None

    def is_available(self) -> bool:
        if self.status == "healthy":
            return True
        if self.backoff_until:
            cutoff = datetime.fromisoformat(self.backoff_until)
            if cutoff.tzinfo is None:
                cutoff = cutoff.replace(tzinfo=timezone.utc)
            return _utcnow() >= cutoff
        return False


class ProviderHealthTracker:
    """
    Lightweight circuit breaker for LLM providers.

    Usage:
        tracker = ProviderHealthTracker()
        if tracker.is_available("groq"):
            try:
                result = await groq_call(...)
                tracker.record_success("groq")
            except Exception as e:
                tracker.record_failure("groq", str(e))
    """

    def __init__(self, db_path: Path = _HEALTH_DB):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._statuses: Dict[str, ProviderStatus] = {}
        self._load()

    def _load(self):
        if self.db_path.exists():
            try:
                with open(self.db_path) as f:
                    raw = json.load(f)
                self._statuses = {k: ProviderStatus(**v) for k, v in raw.items()}
            except Exception:
                self._statuses = {}

    def _save(self):
        try:
            with open(self.db_path, "w") as f:
                json.dump({k: asdict(v) for k, v in self._statuses.items()}, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist provider health: {e}")

    def _get(self, provider: str) -> ProviderStatus:
        if provider not in self._statuses:
            self._statuses[provider] = ProviderStatus(provider_name=provider)
        return self._statuses[provider]

    def record_failure(self, provider: str, error: str = ""):
        """Record an API failure and set exponential backoff.

        Rate limit errors (429) get shorter backoffs since they reset quickly.
        Resets failure count if the last failure was over 10 minutes ago
        (prevents stale failure counts from causing excessive backoffs).
        """
        s = self._get(provider)

        # Auto-reset if last failure was >10 min ago (stale)
        if s.last_failure_time:
            try:
                last = datetime.fromisoformat(s.last_failure_time)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                if (_utcnow() - last).total_seconds() > 600:
                    s.failure_count = 0
            except Exception:
                pass

        s.last_failure_time = _utcnow().isoformat()
        s.failure_count += 1

        # Backoff times from config (avoids hardcoded magic numbers)
        try:
            from app.config import get_settings
            _cfg = get_settings()
            _rl_base = _cfg.provider_ratelimit_base_seconds
            _rl_max = _cfg.provider_ratelimit_max_seconds
            _err_base = _cfg.provider_error_base_seconds
            _broken_at = _cfg.provider_broken_threshold
        except Exception:
            _rl_base, _rl_max, _err_base, _broken_at = 15.0, 120.0, 30.0, 8

        is_rate_limit = "429" in error or "rate" in error.lower()
        if is_rate_limit:
            backoff = timedelta(seconds=min(_rl_base * s.failure_count, _rl_max))
        else:
            backoff = timedelta(seconds=min(_err_base * s.failure_count, _rl_max))

        s.backoff_until = (_utcnow() + backoff).isoformat()
        s.status = "broken" if s.failure_count >= _broken_at else "degraded"

        logger.debug(f"Provider '{provider}' failure #{s.failure_count}: {error[:60]}")
        self._save()

    def record_success(self, provider: str):
        """Reset health on successful call."""
        s = self._get(provider)
        s.failure_count = 0
        s.status = "healthy"
        s.backoff_until = None
        s.last_successful_call = _utcnow().isoformat()
        self._save()

    def is_available(self, provider: str) -> bool:
        return self._get(provider).is_available()

    def filter_available(self, providers: List[str]) -> List[str]:
        """Return only currently-healthy providers from the list."""
        return [p for p in providers if self.is_available(p)]

    def reset_for_new_run(self):
        """Reset all provider health for a new pipeline run.

        Stale failures from previous runs (hours ago) should not block
        the current run. Clears all statuses and persists empty state.
        """
        if self._statuses:
            logger.info(f"Provider health: resetting {len(self._statuses)} provider statuses for new run")
        self._statuses = {}
        self._save()

    def status_report(self) -> Dict[str, dict]:
        return {k: asdict(v) for k, v in self._statuses.items()}


# Module-level singleton — imported by provider_manager and llm_service
provider_health = ProviderHealthTracker()
