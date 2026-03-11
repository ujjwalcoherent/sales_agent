"""
Contact Quality Bandit — Thompson Sampling for contact role selection.

Problem: Given a business event (e.g., "funding"), which job role to contact
at a company of a given size? The wrong role = no reply. The right role = deal.

Solution: Thompson Sampling on (role × event_type × company_size) triples.
Each arm = one combination. Beta posterior updated from:
  - Email opened → reward += 0.3 (weak positive signal)
  - Email replied → reward += 1.0 (strong positive: they want to talk)
  - Email bounced → reward += 0.0 (neutral — contact exists, wasn't actionable)
  - Lead converted → reward += 1.5 (strongest: resulted in a sale/meeting)

Prior initialization: Beta(2, 1) for roles that commonly receive cold outreach
(VP, Director-level), Beta(1, 2) for roles that rarely do (CEO in enterprise).

Architecture:
  - Same Beta posterior math as source_bandit.py (reuses pattern)
  - State stored in data/contact_bandit.json (JSON-serializable)
  - Signals arrive via LearningSignalBus (no direct coupling to email_agent)
  - Used by contact_agent.py to rank candidate contacts before fetching details

References:
  Chapelle & Li (2011) Thompson Sampling for CTR — arXiv:1111.1797
  Russo et al. (2018) Tutorial on Thompson Sampling — arXiv:1707.02038
  Li et al. (2010) Contextual bandits LinUCB — WWW 2010
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_BANDIT_PATH = Path("./data/contact_bandit.json")

# Reward weights (all normalized to [0.0, 1.5])
REWARD_EMAIL_OPEN = 0.3
REWARD_EMAIL_REPLY = 1.0
REWARD_EMAIL_BOUNCE = 0.0
REWARD_LEAD_CONVERTED = 1.5
REWARD_EMAIL_SKIPPED = -0.1  # Implicit negative: lead generated but no email sent

# Prior initialization based on role reachability literature
# (higher α = more optimistic prior = better starting point for exploration)
_ROLE_PRIORS: Dict[str, Tuple[float, float]] = {
    # Enterprise roles: VPs are reachable, C-Suite is not
    "VP": (3.0, 1.0),
    "Director": (3.0, 1.0),
    "Head of": (2.5, 1.0),
    "Manager": (2.0, 1.5),
    "CTO": (2.0, 2.0),    # Reachable at mid-market, not enterprise
    "CFO": (1.5, 2.0),
    "CMO": (2.0, 1.5),
    "COO": (1.5, 2.0),
    "CEO": (1.5, 2.5),    # Reachable at SMB, not enterprise
    "Founder": (2.5, 1.5),  # Typically SMB, high engagement
    "CISO": (2.5, 1.0),   # Security events → CISO is exactly right
    "CLO": (2.0, 1.5),
}
_DEFAULT_PRIOR = (2.0, 2.0)  # Beta(2, 2) — moderate uncertainty


@dataclass
class ContactArm:
    """One arm of the contact bandit: (role × event_type × company_size).

    Beta(α, β) posterior:
      α = prior_alpha + sum(rewards)
      β = prior_beta + sum(1 - rewards) [capped at 1 per interaction]
    """
    role_key: str         # Normalized role label (e.g., "VP Finance")
    event_type: str       # "funding", "expansion", "technology_adoption", etc.
    company_size: str     # "smb", "mid_market", "enterprise"
    alpha: float = 2.0    # Beta prior α
    beta: float = 2.0     # Beta prior β
    total_contacts: int = 0
    total_reward: float = 0.0

    @property
    def arm_key(self) -> str:
        return f"{self.role_key}|{self.event_type}|{self.company_size}"

    @property
    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sample(self) -> float:
        """Thompson Sampling: draw from Beta(α, β) posterior."""
        return random.betavariate(max(self.alpha, 0.001), max(self.beta, 0.001))

    def update(self, reward: float) -> None:
        """Update posterior with observed reward.

        Reward is normalized to [0, 1] for Beta update:
          - reward > 1.0 → treated as 1.0 for β update (bonus applied to α)
          - reward < 0 → increases β (negative signal)
        """
        self.total_contacts += 1
        self.total_reward += reward

        # Normalize reward for Beta(α, β) update
        # α += success weight, β += failure weight
        success_weight = min(max(reward, 0.0), 1.0)
        failure_weight = 1.0 - success_weight

        # Extra α bonus for above-threshold rewards (converted lead, strong reply)
        if reward >= REWARD_EMAIL_REPLY:
            success_weight = min(reward, 1.5) / 1.5  # Scale to [0,1]
            failure_weight = 0.0  # Full success — no failure penalty

        self.alpha += success_weight
        self.beta += failure_weight

    def to_dict(self) -> Dict:
        return {
            "role_key": self.role_key,
            "event_type": self.event_type,
            "company_size": self.company_size,
            "alpha": round(self.alpha, 4),
            "beta": round(self.beta, 4),
            "total_contacts": self.total_contacts,
            "total_reward": round(self.total_reward, 4),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ContactArm":
        return cls(**d)


class ContactBandit:
    """Thompson Sampling bandit for contact role × event_type × company_size.

    Learns which role to contact for each (event_type, company_size) combination
    from email engagement signals — no human feedback required.

    Usage:
        bandit = ContactBandit.load()

        # Rank candidate roles before fetching contact details
        ranked = bandit.rank_roles(
            roles=["VP Finance", "CFO", "Director of FP&A"],
            event_type="funding",
            company_size="enterprise",
        )

        # After email outcome:
        bandit.update("CFO", "funding", "enterprise", reward=REWARD_EMAIL_REPLY)
        bandit.save()
    """

    def __init__(self):
        self.arms: Dict[str, ContactArm] = {}
        self.total_updates: int = 0

    def _get_arm(self, role: str, event_type: str, company_size: str) -> ContactArm:
        """Get or create an arm for this (role, event, size) triple."""
        # Normalize role key: extract role category for prior lookup
        role_category = self._normalize_role(role)
        arm_key = f"{role}|{event_type}|{company_size}"

        if arm_key not in self.arms:
            # Initialize with informed prior based on role category
            prior_a, prior_b = _ROLE_PRIORS.get(role_category, _DEFAULT_PRIOR)

            # Adjust prior by company_size: CEOs/Founders at enterprise get pessimistic prior
            if company_size == "enterprise" and role_category in ("CEO", "Founder"):
                prior_b += 2.0  # Strong pessimistic prior — CEOs don't read cold email at scale
            elif company_size == "smb" and role_category in ("VP", "Director", "Head of"):
                prior_b += 1.0  # VP in SMB → CEO is usually the right contact

            self.arms[arm_key] = ContactArm(
                role_key=role,
                event_type=event_type,
                company_size=company_size,
                alpha=prior_a,
                beta=prior_b,
            )
        return self.arms[arm_key]

    def rank_roles(
        self,
        roles: List[str],
        event_type: str,
        company_size: str,
        n_samples: int = 1,
    ) -> List[Tuple[str, float]]:
        """Rank roles using Thompson Sampling — draw from each arm's posterior.

        Args:
            roles: candidate role titles to rank
            event_type: type of triggering event
            company_size: "smb" | "mid_market" | "enterprise"
            n_samples: number of Thompson samples per arm (averaging reduces variance)

        Returns:
            List of (role, score) sorted by score descending.
            Higher score = contact this role first.
        """
        results = []
        for role in roles:
            arm = self._get_arm(role, event_type, company_size)
            # Average over n_samples to reduce variance (especially for new arms)
            score = sum(arm.sample() for _ in range(n_samples)) / n_samples
            results.append((role, round(score, 4)))

        results.sort(key=lambda x: x[1], reverse=True)
        logger.debug(
            f"[contact_bandit] ranked {len(roles)} roles for "
            f"{event_type}/{company_size}: {results[:3]}"
        )
        return results

    def update(
        self,
        role: str,
        event_type: str,
        company_size: str,
        reward: float,
    ) -> None:
        """Update the arm's posterior with an observed reward signal.

        Args:
            role: the role that was contacted
            event_type: triggering event type
            company_size: company size category
            reward: use the REWARD_* constants or float in [0.0, 1.5]
        """
        arm = self._get_arm(role, event_type, company_size)
        old_mean = arm.posterior_mean
        arm.update(reward)
        self.total_updates += 1
        logger.debug(
            f"[contact_bandit] update: {role}/{event_type}/{company_size} "
            f"reward={reward:.2f} posterior={old_mean:.3f}→{arm.posterior_mean:.3f}"
        )

    def get_top_roles(
        self,
        event_type: str,
        company_size: str,
        top_n: int = 3,
    ) -> List[Tuple[str, float]]:
        """Return top roles by posterior mean (exploitation only — no sampling).

        Use this for reporting, not for selection during live pipeline.
        """
        relevant = [
            (arm.role_key, arm.posterior_mean)
            for arm in self.arms.values()
            if arm.event_type == event_type and arm.company_size == company_size
        ]
        relevant.sort(key=lambda x: x[1], reverse=True)
        return relevant[:top_n]

    def get_arm_stats(self) -> List[Dict]:
        """Return all arm statistics for monitoring."""
        return [
            {
                **arm.to_dict(),
                "posterior_mean": round(arm.posterior_mean, 3),
            }
            for arm in sorted(self.arms.values(), key=lambda a: a.posterior_mean, reverse=True)
        ]

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _normalize_role(role: str) -> str:
        """Extract role category prefix for prior lookup."""
        role_upper = role.strip()
        for category in sorted(_ROLE_PRIORS.keys(), key=len, reverse=True):
            if role_upper.startswith(category) or category in role_upper:
                return category
        return "Manager"  # fallback

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Path = _BANDIT_PATH) -> None:
        """Persist arm states to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "arms": {k: v.to_dict() for k, v in self.arms.items()},
            "total_updates": self.total_updates,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"[contact_bandit] saved {len(self.arms)} arms")

    @classmethod
    def load(cls, path: Path = _BANDIT_PATH) -> "ContactBandit":
        """Load persisted arms, or start fresh."""
        bandit = cls()
        if not path.exists():
            logger.info("[contact_bandit] no saved state — starting fresh with priors")
            return bandit
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for arm_dict in data.get("arms", {}).values():
                arm = ContactArm.from_dict(arm_dict)
                bandit.arms[arm.arm_key] = arm
            bandit.total_updates = data.get("total_updates", 0)
            logger.info(f"[contact_bandit] loaded {len(bandit.arms)} arms")
        except Exception as exc:
            logger.warning(f"[contact_bandit] load failed ({exc}) — starting fresh")
        return bandit
