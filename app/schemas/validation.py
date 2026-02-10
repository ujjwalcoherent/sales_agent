"""
Validation result models for cross-validation between synthesis and source data.

Used by the ValidatorAgent to score how well LLM synthesis output is grounded
in the original source articles. Prevents hallucinated company names, fabricated
trends, and ungrounded causal chains from propagating downstream.

Verdict flow:
  PASS   -> synthesis accepted as-is, moves to impact/company agents
  REVISE -> specific feedback sent back to LLM for targeted regeneration
  REJECT -> synthesis discarded, cluster skipped (too fabricated to fix)

Groundedness scoring uses NER entity overlap, keyword intersection, and
embedding similarity — NO additional LLM call required.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from uuid import UUID, uuid4


class ValidationVerdict(str, Enum):
    """Outcome of cross-validation check."""
    PASS = "pass"
    REVISE = "revise"
    REJECT = "reject"


class FieldGroundedness(BaseModel):
    """
    Groundedness assessment for a single field in the synthesis output.

    Each field (trend_title, affected_companies, causal_chain, etc.) gets
    an independent score based on how much evidence supports it in the
    source articles.

    Scoring method varies by field type:
    - Entity fields (key_entities, affected_companies): NER overlap ratio
    - Text fields (trend_title, trend_summary): keyword overlap + embedding sim
    - Structured fields (causal_chain, event_5w1h): keyword overlap per item
    """
    field_name: str
    score: float = Field(ge=0.0, le=1.0, default=0.0)
    method: str = ""  # "ner_overlap", "keyword_overlap", "embedding_similarity", "combined"
    evidence_found: List[str] = Field(default_factory=list)  # Matching source snippets
    ungrounded_items: List[str] = Field(default_factory=list)  # Items with no source evidence
    detail: str = ""  # Human-readable explanation

    @field_validator('score', mode='before')
    @classmethod
    def clamp_score(cls, v):
        if v is None:
            return 0.0
        return max(0.0, min(1.0, float(v)))


class ValidationRound(BaseModel):
    """
    Result of a single validation round (one pass through the validator).

    Multiple rounds occur when the verdict is REVISE and the synthesizer
    regenerates with feedback.
    """
    round_number: int = 1
    verdict: ValidationVerdict = ValidationVerdict.REJECT
    overall_score: float = Field(ge=0.0, le=1.0, default=0.0)
    field_scores: List[FieldGroundedness] = Field(default_factory=list)
    feedback: List[str] = Field(default_factory=list)  # Specific revision instructions
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def failing_fields(self) -> List[FieldGroundedness]:
        """Fields that scored below the pass threshold."""
        return [f for f in self.field_scores if f.score < 0.5]


class ValidationResult(BaseModel):
    """
    Complete validation result for a single cluster's synthesis.

    Tracks all rounds of back-and-forth between synthesizer and validator.
    The final_verdict is the outcome of the last round.
    """
    id: UUID = Field(default_factory=uuid4)
    cluster_id: Optional[int] = None  # HDBSCAN cluster label
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Validation rounds (1 = first check, 2+ = after revision)
    rounds: List[ValidationRound] = Field(default_factory=list)
    total_rounds: int = 0

    # Final outcome
    final_verdict: ValidationVerdict = ValidationVerdict.REJECT
    final_score: float = Field(ge=0.0, le=1.0, default=0.0)

    # Source evidence summary
    source_entity_count: int = 0  # Total unique entities from source articles
    source_keyword_count: int = 0  # Total unique keywords from source articles
    synthesis_entity_count: int = 0  # Entities claimed by synthesis
    entity_overlap_ratio: float = 0.0  # What fraction of claimed entities exist in sources

    # Performance
    elapsed_ms: int = 0

    @property
    def passed(self) -> bool:
        return self.final_verdict == ValidationVerdict.PASS

    @property
    def was_revised(self) -> bool:
        return self.total_rounds > 1

    def summary(self) -> str:
        """One-line summary for logging."""
        status = self.final_verdict.value.upper()
        rounds_str = f" ({self.total_rounds} rounds)" if self.total_rounds > 1 else ""
        return (
            f"[{status}] score={self.final_score:.2f}, "
            f"entities={self.entity_overlap_ratio:.0%}, "
            f"elapsed={self.elapsed_ms}ms{rounds_str}"
        )
