"""
Unified data models for the Intelligence Engine.

All 22 agents share these types — typed I/O at every boundary (MetaGPT pattern).
Models are immutable-friendly: use Field(default_factory=...) not mutable defaults.

Hierarchy:
  DiscoveryScope → pipeline → IntelligenceResult
  Each agent step: TypedInput → TypedOutput (never raw dicts)
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════════════════════

class DiscoveryMode(str, Enum):
    """Entry path for the intelligence pipeline."""
    COMPANY_FIRST = "company_first"    # Deep intel on named companies
    INDUSTRY_FIRST = "industry_first"  # Discover companies in an industry
    REPORT_DRIVEN = "report_driven"    # Corroborate analyst report claims



class IndustryOrder(int, Enum):
    FIRST = 1   # Direct players (drug manufacturers)
    SECOND = 2  # Adjacent players (CROs, medtech)


class EventGranularity(str, Enum):
    MAJOR = "major"   # Earnings, M&A, product launch
    SUB = "sub"       # Analyst reaction, partner details
    NANO = "nano"     # Board appointments, office leases


class AgentRequestType(str, Enum):
    RETRY_CLUSTER = "retry_cluster"
    FETCH_MORE = "fetch_more"
    SPLIT_ENTITY = "split_entity"
    MERGE_CLUSTERS = "merge_clusters"
    RECLASSIFY = "reclassify"
    VALIDATE_ENTITY = "validate_entity"


class RequestPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"


class ValidationFailureAction(str, Enum):
    """What ValidationAgent signals when a check fails."""
    RECLUSTER = "recluster"           # Check 1 coherence fail
    MERGE_NEAREST = "merge_nearest"   # Check 3 size fail
    FLAG_SYNDICATION = "flag_syndication"  # Check 7 fail
    DROP = "drop"                     # Unrecoverable


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

class DiscoveryScope(BaseModel):
    """Typed entry point for all 3 discovery paths.

    All pipeline parameters flow from here — no ambient globals.
    """
    mode: DiscoveryMode = DiscoveryMode.COMPANY_FIRST

    # Company-First / Account mode
    companies: List[str] = Field(default_factory=list)   # ["NVIDIA", "CrowdStrike"]

    # Industry-First
    industry: Optional[str] = None                       # "Technology > Cybersecurity"
    industry_order: IndustryOrder = IndustryOrder.FIRST

    # Report-Driven
    report_text: Optional[str] = None                    # paste analyst report

    # Common
    region: str = "GLOBAL"                               # ISO code or "GLOBAL"
    hours: int = 120                                     # Look-back window
    user_products: List[str] = Field(default_factory=list)  # What the user sells

    # Advanced
    max_rounds: int = 3
    event_granularity: str = "major+sub"
    mock_mode: bool = False

    @field_validator("companies", mode="before")
    @classmethod
    def strip_companies(cls, v: List[str]) -> List[str]:
        return [c.strip() for c in v if c.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# ARTICLE MODELS  (FetchAgent → DedupAgent → FilterAgent)
# ══════════════════════════════════════════════════════════════════════════════

class RawArticle(BaseModel):
    """Article as fetched — may contain duplicates, noise, missing fields."""
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    url: str
    title: str = ""
    summary: str = ""
    full_text: str = ""
    source_name: str = ""
    source_url: str = ""
    published_at: Optional[datetime] = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    language: str = "en"
    fetch_method: str = ""                     # "rss", "tavily", "google_news"


class Article(BaseModel):
    """Article after deduplication — guaranteed unique, dated, sourced."""
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    url: str
    title: str
    summary: str = ""
    full_text: str = ""
    source_name: str
    source_url: str
    published_at: datetime
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    language: str = "en"
    fetch_method: str = ""

    # Computed during filter step
    is_relevant: bool = True
    relevance_confidence: float = 1.0

    # Set by NER step
    embedding: List[float] = Field(default_factory=list)
    entities_raw: List[str] = Field(default_factory=list)

    # Industry classification (set by industry_classifier.py after NLI filter)
    industry_label: Optional[str] = None          # e.g. "healthcare_pharma"
    industry_order: Optional[int] = None          # 1 = direct player, 2 = adjacent/supply-chain
    # Index within run (set by pipeline for matrix indexing)
    run_index: int = -1


# ══════════════════════════════════════════════════════════════════════════════
# DEDUP (DedupAgent output)
# ══════════════════════════════════════════════════════════════════════════════

class DedupResult(BaseModel):
    """Output of DedupAgent — math gate 1."""
    articles: List[Article]
    removed_count: int
    dedup_pairs: List[Tuple[str, str]] = Field(default_factory=list)  # (kept_url, removed_url)

    # Math assertions (all must be True for gate to pass)
    assertion_count_non_increasing: bool = True
    assertion_threshold_respected: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# SALIENCE / FILTER (FilterAgent output)
# ══════════════════════════════════════════════════════════════════════════════

class FilterResult(BaseModel):
    """Output of FilterAgent — math gate 2."""
    articles: List[Article]
    dropped_articles: List[str] = Field(default_factory=list)    # article IDs
    gap4_dropped_companies: List[str] = Field(default_factory=list)  # company names
    llm_classified_count: int = 0        # How many went to LLM (ambiguous cases)
    auto_accepted_count: int = 0         # nli_entailment >= nli_auto_accept
    auto_rejected_count: int = 0         # nli_entailment <= nli_auto_reject

    # NLI filter diagnostics (used by SourceBandit reward signal)
    nli_mean_entailment: float = 0.0              # Mean entailment of kept articles
    nli_scores_by_source: Dict[str, float] = Field(default_factory=dict)
    hypothesis_version: str = "v0"

    # Math assertions
    assertion_target_companies_present: bool = True
    assertion_count_non_increasing: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# ENTITY EXTRACTION (NERAgent + NormAgent + ClassifierAgent)
# ══════════════════════════════════════════════════════════════════════════════

class Provenance(BaseModel):
    """Source attribution for any data point."""
    source_url: str
    source_name: str
    source_tier: str = "tier_2"
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = 0.5
    evidence: List[str] = Field(default_factory=list)
    corroborating_sources: int = 0


class IndustryClassification(BaseModel):
    """Industry classification for a company entity."""
    level_1: str
    level_2: str = ""
    order: IndustryOrder = IndustryOrder.FIRST
    confidence: float = 0.5
    evidence: List[str] = Field(default_factory=list)


class EntityGroup(BaseModel):
    """A validated, normalized entity with all article references.

    Built by: NERAgent → NormAgent → ClassifierAgent
    Each group = one real-world company/person/product.
    """
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    canonical_name: str
    variant_names: List[str] = Field(default_factory=list)
    entity_type: str = "ORG"              # ORG, PERSON, PRODUCT, GPE
    article_indices: List[int] = Field(default_factory=list)
    mention_count: int = 0
    avg_salience: float = 0.0

    # Validation (GLiNER B2B check)
    is_validated: bool = False
    is_b2b: bool = False
    validation_confidence: float = 0.0
    validation_evidence: List[str] = Field(default_factory=list)

    # Industry classification (ORG entities only)
    industry: Optional[IndustryClassification] = None

    # Target flag — set when entity matches user's requested companies
    is_target: bool = False

    provenance: List[Provenance] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTER MODELS (ClusterAgent + ValidationAgent)
# ══════════════════════════════════════════════════════════════════════════════

class OutlierRecord(BaseModel):
    """Record of an ejected outlier with full reasoning."""
    item_type: str                   # "article", "entity", "cluster"
    item_id: str
    reason: str
    evidence: str = ""
    confidence: float = 0.0


class DendrogramMetrics(BaseModel):
    """HAC dendrogram analysis metrics."""
    cophenetic_r: float = 0.0
    cut_threshold: float = 0.0
    n_subclusters: int = 1
    outlier_indices: List[int] = Field(default_factory=list)
    linkage_method: str = "average"


class ClusterResult(BaseModel):
    """A validated news event cluster.

    Each cluster = a specific event affecting specific entities.
    Not a vague topic grouping.
    """
    cluster_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    label: str = ""                  # Populated by SynthesisAgent

    # Membership
    article_indices: List[int] = Field(default_factory=list)
    article_count: int = 0

    # Entity provenance
    primary_entity: Optional[str] = None
    entity_names: List[str] = Field(default_factory=list)
    entity_groups: List[str] = Field(default_factory=list)  # EntityGroup IDs

    # Event classification
    event_type: str = ""
    event_granularity: EventGranularity = EventGranularity.MAJOR

    # Quality metrics
    confidence: float = 0.0

    # Dendrogram (HAC only)
    dendrogram_metrics: Optional[DendrogramMetrics] = None

    # Coherence
    coherence_score: float = 0.0
    entity_consistency: float = 0.0
    source_diversity: int = 0

    # Embedding (centroid of member articles)
    centroid_embedding: List[float] = Field(default_factory=list)

    # Clustering metadata
    algorithm: str = ""              # "hac", "hdbscan", "leiden"
    is_entity_seeded: bool = False
    parent_entity_group: Optional[str] = None

    # Industry classification
    industry: Optional[IndustryClassification] = None
    industries: List[str] = Field(default_factory=list)  # Human-readable industry names from article labels

    # Product matching
    matched_user_products: List[str] = Field(default_factory=list)

    # Synthesis
    summary: str = ""                # 2-3 sentence summary from SynthesisAgent
    representative_article_indices: List[int] = Field(default_factory=list)
    requires_review: bool = False    # Set if synthesis fails after 3 retries

    # Critic validation (AutoResearch quality gate)
    critic_score: float = 0.0        # 0-1, set by critic after synthesis
    critic_reasoning: str = ""       # Why critic accepted/rejected

    # Strategic business opportunity score. Range [0.0, 1.0].
    # Formula: 0.30×event_specificity + 0.25×entity_action + 0.20×industry + 0.15×temporal + 0.10×source_cred
    # Jim Cramer commentary = ~0.09 (LOW), Apollo acquisition = ~0.90 (HIGH)
    strategic_score: float = 0.0

    # Evidence chain (article → trend → lead citation)
    evidence_chain: Optional["EvidenceChain"] = None

    provenance: List[Provenance] = Field(default_factory=list)


class ValidationCheck(BaseModel):
    """Result of one of the 7 validation checks."""
    name: str                        # "coherence", "separation", "size", etc.
    passed: bool
    score: float = 0.0
    critique: str = ""               # SPECIFIC feedback for retry (not generic)
    action: ValidationFailureAction = ValidationFailureAction.RECLUSTER


class ValidationResult(BaseModel):
    """Result of the 7-check math gate on one cluster — math gate 6."""
    cluster_id: str
    passed: bool = False
    composite_score: float = 0.0

    checks: List[ValidationCheck] = Field(default_factory=list)
    rejection_reasons: List[str] = Field(default_factory=list)
    outliers: List[OutlierRecord] = Field(default_factory=list)
    ejected_article_indices: List[int] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHESIS (SynthesisAgent output — FIRST LLM CALL)
# ══════════════════════════════════════════════════════════════════════════════

class ClusterLabel(BaseModel):
    """LLM-generated label + summary for a cluster — math gate 7.

    Validated by: word count, proper noun presence, no HTML/URLs.
    Retried via Reflexion (Shinn 2023): max 3 retries with critique.
    """
    cluster_id: str
    label: str                       # "3-8 words, must name key company/topic"
    summary: str                     # "2-3 sentences, specific facts only"
    requires_review: bool = False
    attempt_count: int = 1
    last_critique: str = ""

    @field_validator("label")
    @classmethod
    def validate_label_format(cls, v: str) -> str:
        words = v.strip().split()
        if not (3 <= len(words) <= 8):
            raise ValueError(f"Label must be 3-8 words, got {len(words)}: '{v}'")
        if "<" in v or "http" in v or "**" in v:
            raise ValueError(f"Label must not contain HTML/URLs/markdown: '{v}'")
        return v


# ══════════════════════════════════════════════════════════════════════════════
# MATCH ENGINE (MatchAgent output)
# ══════════════════════════════════════════════════════════════════════════════

class Product(BaseModel):
    """A product the sales team sells — used for trigger matching."""
    name: str
    description: str
    target_roles: List[str] = Field(default_factory=list)   # ["CISO", "CTO"]
    target_industries: List[str] = Field(default_factory=list)
    buying_triggers: List[str] = Field(default_factory=list)  # ["data breach", "ransomware"]
    embedding: List[float] = Field(default_factory=list)     # computed at load time


class ProductCatalog(BaseModel):
    """Full catalog of products owned by the sales team."""
    products: List[Product] = Field(default_factory=list)
    owner_company: str = ""
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MatchResult(BaseModel):
    """Score for one (cluster, product) pair — math gate 8.

    fit_score = 0.50 * keyword_overlap
              + 0.30 * semantic_similarity
              + 0.20 * industry_match
    All math, no LLM.
    """
    cluster_id: str
    company: str
    product_name: str
    fit_score: float                 # 0.0 - 1.0

    # Breakdown
    keyword_overlap: float = 0.0
    semantic_similarity: float = 0.0
    industry_match: float = 0.0

    # Evidence
    evidence_quotes: List[str] = Field(default_factory=list)
    matched_triggers: List[str] = Field(default_factory=list)
    why_it_fits: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# INTER-AGENT COMMUNICATION
# ══════════════════════════════════════════════════════════════════════════════

class AgentRequest(BaseModel):
    """Request from one agent to another (Signal Bus message).

    Enables bidirectional agent communication without direct imports.
    """
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    from_agent: str
    to_agent: str
    request_type: AgentRequestType
    details: Dict[str, Any] = Field(default_factory=dict)
    priority: RequestPriority = RequestPriority.NORMAL
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False


class ThoughtEntry(BaseModel):
    """Chain-of-thought log entry (ReAct TAO loop)."""
    agent: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    thought: str
    action: str = ""
    observation: str = ""            # Tool result / math check result
    confidence: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE SHARED STATE (Blackboard pattern — Erman et al. 1980)
# ══════════════════════════════════════════════════════════════════════════════

class PipelineState(BaseModel):
    """Within-run shared state. All agents read/write via this object.

    NOT serialized via LangGraph state (contains non-serializable objects).
    Passed via dependency injection through IntelligenceDeps.
    """
    scope: Optional[DiscoveryScope] = None

    # Stepwise outputs (each agent reads previous, writes its own)
    raw_articles: List[RawArticle] = Field(default_factory=list)
    articles: List[Article] = Field(default_factory=list)           # post-dedup
    filtered_articles: List[Article] = Field(default_factory=list)  # post-filter
    entity_groups: List[EntityGroup] = Field(default_factory=list)
    clusters: List[ClusterResult] = Field(default_factory=list)
    labeled_clusters: List[ClusterResult] = Field(default_factory=list)  # post-synthesis
    match_results: List[MatchResult] = Field(default_factory=list)

    # Validation state
    passed_cluster_ids: List[str] = Field(default_factory=list)
    rejected_cluster_ids: List[str] = Field(default_factory=list)
    validation_results: List[ValidationResult] = Field(default_factory=list)

    # Gap 4 tracking
    gap4_dropped_companies: List[str] = Field(default_factory=list)

    # Inter-agent communication
    agent_requests: List[AgentRequest] = Field(default_factory=list)
    thought_log: List[ThoughtEntry] = Field(default_factory=list)
    noise_article_indices: List[int] = Field(default_factory=list)

    # Round tracking (supervisor negotiation)
    round_number: int = 0
    max_rounds: int = 3

    # Run metadata
    run_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def add_thought(self, agent: str, thought: str,
                    action: str = "", observation: str = "", confidence: float = 0.0) -> None:
        self.thought_log.append(ThoughtEntry(
            agent=agent, thought=thought, action=action,
            observation=observation, confidence=confidence,
        ))

    def add_request(self, from_agent: str, to_agent: str,
                    request_type: AgentRequestType, details: Dict[str, Any],
                    priority: RequestPriority = RequestPriority.NORMAL) -> None:
        self.agent_requests.append(AgentRequest(
            from_agent=from_agent, to_agent=to_agent,
            request_type=request_type, details=details, priority=priority,
        ))

    def pending_requests(self, for_agent: Optional[str] = None) -> List[AgentRequest]:
        reqs = [r for r in self.agent_requests if not r.resolved]
        if for_agent:
            reqs = [r for r in reqs if r.to_agent == for_agent]
        return sorted(reqs, key=lambda r: (
            0 if r.priority == RequestPriority.CRITICAL else
            1 if r.priority == RequestPriority.HIGH else 2
        ))

    def passed_clusters(self) -> List[ClusterResult]:
        return [c for c in self.clusters if c.cluster_id in self.passed_cluster_ids]

    def summary(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "round": self.round_number,
            "raw_articles": len(self.raw_articles),
            "articles_post_dedup": len(self.articles),
            "articles_post_filter": len(self.filtered_articles),
            "entity_groups": len(self.entity_groups),
            "clusters": len(self.clusters),
            "passed_clusters": len(self.passed_cluster_ids),
            "rejected_clusters": len(self.rejected_cluster_ids),
            "gap4_dropped": self.gap4_dropped_companies,
            "noise_articles": len(self.noise_article_indices),
            "match_results": len(self.match_results),
            "pending_requests": len(self.pending_requests()),
        }


# ══════════════════════════════════════════════════════════════════════════════
# FINAL OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

class IntelligenceResult(BaseModel):
    """Final output of an intelligence pipeline run."""
    run_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    scope: Optional[DiscoveryScope] = None
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Core results
    clusters: List[ClusterResult] = Field(default_factory=list)
    entity_groups: List[EntityGroup] = Field(default_factory=list)
    match_results: List[MatchResult] = Field(default_factory=list)
    filtered_articles: List[Article] = Field(default_factory=list)  # post-filter, for source bandit backward cascade

    # Quality metrics (all dynamically computed — no hardcoded thresholds)
    total_articles_fetched: int = 0
    total_articles_post_filter: int = 0
    total_clusters: int = 0
    noise_rate: float = 0.0
    mean_coherence: float = 0.0
    mean_fit_score: float = 0.0

    # Gap 4 report
    gap4_dropped_companies: List[str] = Field(default_factory=list)

    # Reasoning trace
    thought_log: List[ThoughtEntry] = Field(default_factory=list)
    rounds_completed: int = 0
    agent_requests_processed: int = 0

    # NLI filter diagnostics (flows to source bandit reward in orchestrator)
    nli_scores_by_source: Dict[str, float] = Field(default_factory=dict)



# ══════════════════════════════════════════════════════════════════════════════
# CRITIC VALIDATION (AutoResearch quality gate)
# ══════════════════════════════════════════════════════════════════════════════

class CriticResult(BaseModel):
    """Output of the trend critic — validates cluster quality post-synthesis."""
    score: float                     # 0-1 quality score
    passed: bool                     # score >= threshold (default 0.6)
    reasoning: str                   # Why the critic accepted/rejected
    refined_label: str = ""          # Improved label if critic found issues


# ══════════════════════════════════════════════════════════════════════════════
# EVIDENCE CHAIN (article → trend → lead citation path)
# ══════════════════════════════════════════════════════════════════════════════

class EvidenceChain(BaseModel):
    """Explicit citation path from articles to trend to lead.

    Used by email_agent to cite real evidence in outreach.
    """
    trend_id: str
    article_ids: List[str] = Field(default_factory=list)
    key_snippets: List[str] = Field(default_factory=list)  # 1-2 sentence evidence quotes
    companies_cited: List[str] = Field(default_factory=list)
    confidence: float = 0.0
