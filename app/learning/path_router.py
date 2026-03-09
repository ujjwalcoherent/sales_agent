"""
Q-Learning Path Router — chooses which of the 3 pipeline paths to activate.

Inspired by Ruflo pattern (19.8k stars): state → Q-values → action selection.
Simple Q-table stored in data/path_router.json — no neural net at this scale.

3 Paths:
  INDUSTRY_FIRST  — Discover companies across a target industry
  COMPANY_FIRST   — Track specific accounts from user's list
  REPORT_DRIVEN   — Match recent news to user's report/product

State features (discrete, low-cardinality — Q-table stays small):
  - industry_active: bool  (profile has target_industries)
  - account_list_active: bool  (profile has account_list)
  - report_active: bool  (profile has report_summary)
  - news_volume: "low" | "medium" | "high"  (article count from last run)
  - last_path_grade: "A" | "B" | "C" | "D" | "F"  (MetaReasoner grade)

Action: weight vector over the 3 paths (sums to 1.0).
Reward: email_engagement_rate for the run (fully automated via signal_bus).

Learning:
  Q(s, a) ← Q(s, a) + α * [r + γ * max Q(s', a') - Q(s, a)]
  α=0.10 (slow learner — stable learning), γ=0.90 (value future rewards)
  ε-greedy exploration: ε starts at 0.30, decays to 0.05 over 50 runs.

References:
  Ruflo Q-Learning router (SONA: RETRIEVE→JUDGE→DISTILL→CONSOLIDATE→ROUTE)
  Watkins & Dayan (1992) Q-Learning convergence theorem
  Sutton & Barto RL book Ch.6 (TD learning)
"""

import json
import logging
import math
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_ROUTER_PATH = Path("./data/path_router.json")

# Q-Learning hyperparameters
_ALPHA = 0.10     # Learning rate (slow — stable)
_GAMMA = 0.90     # Discount factor (value future rewards)
_EPSILON_START = 0.30   # Initial exploration rate
_EPSILON_MIN = 0.05     # Floor exploration (always explore a little)
_EPSILON_DECAY_RUNS = 50  # Runs to decay from start to min

# Path identifiers
PATH_INDUSTRY = "industry_first"
PATH_COMPANY = "company_first"
PATH_REPORT = "report_driven"
ALL_PATHS = [PATH_INDUSTRY, PATH_COMPANY, PATH_REPORT]


@dataclass
class RouterState:
    """Discrete state for Q-table indexing.

    Kept intentionally low-cardinality (< 100 unique states) so the
    Q-table stays small and converges in < 50 runs.
    """
    industry_active: bool = False    # Profile has target_industries
    account_list_active: bool = False  # Profile has account_list (≥1 company)
    report_active: bool = False      # Profile has report_summary
    news_volume: str = "medium"      # "low" (<100 articles) / "medium" / "high" (>500)
    last_grade: str = "B"            # MetaReasoner run grade: A/B/C/D/F

    def to_key(self) -> str:
        """Serialize state to Q-table key string."""
        return (
            f"i{int(self.industry_active)}"
            f"a{int(self.account_list_active)}"
            f"r{int(self.report_active)}"
            f"v{self.news_volume[0]}"   # l/m/h
            f"g{self.last_grade}"
        )

    @staticmethod
    def from_context(
        industry_active: bool,
        account_list_active: bool,
        report_active: bool,
        article_count: int,
        last_grade: str = "B",
    ) -> "RouterState":
        if article_count < 100:
            volume = "low"
        elif article_count < 500:
            volume = "medium"
        else:
            volume = "high"
        return RouterState(
            industry_active=industry_active,
            account_list_active=account_list_active,
            report_active=report_active,
            news_volume=volume,
            last_grade=last_grade or "B",
        )


@dataclass
class PathWeights:
    """Output of the router — normalized weights over the 3 paths."""
    industry_first: float = 0.334
    company_first: float = 0.333
    report_driven: float = 0.333

    def to_dict(self) -> Dict[str, float]:
        return {
            PATH_INDUSTRY: self.industry_first,
            PATH_COMPANY: self.company_first,
            PATH_REPORT: self.report_driven,
        }

    def primary_path(self) -> str:
        """Return the path with the highest weight."""
        d = self.to_dict()
        return max(d, key=d.get)

    def active_paths(self, threshold: float = 0.20) -> List[str]:
        """Return paths with weight >= threshold (parallel activation)."""
        return [p for p, w in self.to_dict().items() if w >= threshold]


class PathRouter:
    """Q-Learning router for 3-path pipeline activation.

    Persistently learns from email engagement signals (fully automated —
    no human feedback required). Q-table stored in data/path_router.json.

    Usage:
        router = PathRouter.load()
        state = RouterState.from_context(...)
        weights = router.select_paths(state)
        # ... run pipeline ...
        router.update(state, chosen_path, reward=engagement_rate)
        router.save()
    """

    def __init__(self):
        # Q-table: {state_key: {path: q_value}}
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.total_runs: int = 0
        self.path_counts: Dict[str, int] = {p: 0 for p in ALL_PATHS}
        self.path_rewards: Dict[str, float] = {p: 0.0 for p in ALL_PATHS}

    @property
    def epsilon(self) -> float:
        """Current exploration rate — decays with runs."""
        if self.total_runs >= _EPSILON_DECAY_RUNS:
            return _EPSILON_MIN
        decay = (self.total_runs / _EPSILON_DECAY_RUNS)
        return _EPSILON_START - (_EPSILON_START - _EPSILON_MIN) * decay

    def _get_q_values(self, state_key: str) -> Dict[str, float]:
        """Get Q-values for a state, initializing with uniform prior if unseen."""
        if state_key not in self.q_table:
            # Uniform initialization — equal value for all paths
            self.q_table[state_key] = {p: 0.0 for p in ALL_PATHS}
        return self.q_table[state_key]

    def select_paths(
        self,
        state: RouterState,
        available_paths: Optional[List[str]] = None,
    ) -> PathWeights:
        """Select path weights using ε-greedy policy.

        If ε-greedy triggers exploration: sample a random path as primary.
        Otherwise: exploit Q-values, converting them to a softmax weight distribution.

        Args:
            state: current pipeline context
            available_paths: paths that are actually available (filtered by profile config)

        Returns:
            PathWeights normalized to sum to 1.0
        """
        available = available_paths or ALL_PATHS
        # Filter to paths that are activated in the profile
        activated = self._filter_by_profile(state, available)

        if not activated:
            activated = [PATH_INDUSTRY]  # Fallback — industry first always possible

        state_key = state.to_key()
        q_values = self._get_q_values(state_key)

        if random.random() < self.epsilon:
            # ε-greedy exploration: pick a random path as dominant
            chosen = random.choice(activated)
            logger.debug(f"[path_router] ε-greedy explore → {chosen} (ε={self.epsilon:.3f})")
        else:
            # Exploit: best Q-value among activated paths
            chosen = max(activated, key=lambda p: q_values.get(p, 0.0))
            logger.debug(f"[path_router] exploit → {chosen} (Q={q_values.get(chosen, 0.0):.3f})")

        # Convert Q-values to softmax weights for parallel path activation
        weights = self._q_to_weights(q_values, activated, chosen)
        logger.info(
            f"[path_router] state={state_key} primary={chosen} "
            f"weights={{{', '.join(f'{p[:3]}={w:.2f}' for p, w in weights.to_dict().items())}}}"
        )
        return weights

    def update(
        self,
        state: RouterState,
        chosen_path: str,
        reward: float,
        next_state: Optional[RouterState] = None,
    ) -> None:
        """Q-Learning update after one pipeline run.

        Args:
            state: state at the time of path selection
            chosen_path: which path was activated as primary
            reward: engagement signal [0.0, 1.0]
                    e.g. email_open_rate, reply_rate, lead_converted_rate
            next_state: state at the end of the run (if None, uses same state)
        """
        state_key = state.to_key()
        q_values = self._get_q_values(state_key)

        # Max Q-value for next state (for bootstrapping future value)
        if next_state:
            next_key = next_state.to_key()
            next_q = self._get_q_values(next_key)
            max_next_q = max(next_q.values()) if next_q else 0.0
        else:
            max_next_q = max(q_values.values())

        # Q-Learning update rule: Q(s,a) ← Q(s,a) + α[r + γ*max Q(s') - Q(s,a)]
        old_q = q_values.get(chosen_path, 0.0)
        new_q = old_q + _ALPHA * (reward + _GAMMA * max_next_q - old_q)
        q_values[chosen_path] = round(new_q, 4)

        # Track statistics
        self.total_runs += 1
        self.path_counts[chosen_path] = self.path_counts.get(chosen_path, 0) + 1
        self.path_rewards[chosen_path] = self.path_rewards.get(chosen_path, 0.0) + reward

        logger.debug(
            f"[path_router] update: {chosen_path} Q={old_q:.3f}→{new_q:.3f} "
            f"reward={reward:.3f} runs={self.total_runs}"
        )

    def get_stats(self) -> Dict:
        """Return routing statistics for monitoring."""
        avg_rewards = {
            p: round(self.path_rewards.get(p, 0.0) / max(self.path_counts.get(p, 1), 1), 3)
            for p in ALL_PATHS
        }
        return {
            "total_runs": self.total_runs,
            "epsilon": round(self.epsilon, 3),
            "path_counts": self.path_counts,
            "avg_rewards": avg_rewards,
            "q_table_states": len(self.q_table),
        }

    def publish_to_bus(self, bus) -> None:
        """Publish routing stats to LearningSignalBus (Phase 1)."""
        stats = self.get_stats()
        # Path router doesn't have a dedicated bus slot yet — extend signal_bus if needed
        # For now: log for monitoring
        logger.info(
            f"[path_router] stats: runs={stats['total_runs']}, "
            f"ε={stats['epsilon']}, avg_rewards={stats['avg_rewards']}"
        )

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _filter_by_profile(state: RouterState, available: List[str]) -> List[str]:
        """Filter paths by what the profile actually has configured."""
        result = []
        for p in available:
            if p == PATH_INDUSTRY and state.industry_active:
                result.append(p)
            elif p == PATH_COMPANY and state.account_list_active:
                result.append(p)
            elif p == PATH_REPORT and state.report_active:
                result.append(p)
        # If nothing is configured, fall back to industry_first
        return result or [PATH_INDUSTRY]

    @staticmethod
    def _q_to_weights(
        q_values: Dict[str, float],
        activated: List[str],
        chosen: str,
    ) -> PathWeights:
        """Convert Q-values to normalized path weights using softmax.

        The chosen path gets a 1.5× bonus before softmax to ensure it's dominant
        but other paths can still run in parallel if their Q-values are close.
        """
        # Apply softmax over activated paths
        values = {}
        for p in activated:
            q = q_values.get(p, 0.0)
            bonus = 1.5 if p == chosen else 1.0
            values[p] = math.exp(q * bonus)

        total = sum(values.values()) or 1.0
        normalized = {p: round(v / total, 3) for p, v in values.items()}

        # Unactivated paths get 0.0
        weights = PathWeights(
            industry_first=normalized.get(PATH_INDUSTRY, 0.0),
            company_first=normalized.get(PATH_COMPANY, 0.0),
            report_driven=normalized.get(PATH_REPORT, 0.0),
        )

        # Renormalize to exactly 1.0
        total_w = weights.industry_first + weights.company_first + weights.report_driven
        if total_w > 0:
            weights.industry_first = round(weights.industry_first / total_w, 3)
            weights.company_first = round(weights.company_first / total_w, 3)
            weights.report_driven = round(1.0 - weights.industry_first - weights.company_first, 3)

        return weights

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Path = _ROUTER_PATH) -> None:
        """Persist Q-table to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "q_table": self.q_table,
            "total_runs": self.total_runs,
            "path_counts": self.path_counts,
            "path_rewards": self.path_rewards,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"[path_router] saved ({len(self.q_table)} states, {self.total_runs} runs)")

    @classmethod
    def load(cls, path: Path = _ROUTER_PATH) -> "PathRouter":
        """Load persisted Q-table, or start fresh."""
        router = cls()
        if not path.exists():
            logger.info("[path_router] no saved Q-table — starting fresh")
            return router
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            router.q_table = data.get("q_table", {})
            router.total_runs = data.get("total_runs", 0)
            router.path_counts = data.get("path_counts", {p: 0 for p in ALL_PATHS})
            router.path_rewards = data.get("path_rewards", {p: 0.0 for p in ALL_PATHS})
            logger.info(
                f"[path_router] loaded: {len(router.q_table)} states, "
                f"{router.total_runs} total runs"
            )
        except Exception as exc:
            logger.warning(f"[path_router] load failed ({exc}) — starting fresh")
        return router
