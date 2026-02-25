"""
Trend detection data models.

Defines the hierarchical trend tree structure (TrendNode, TrendTree) and
supporting types (SignalStrength, TrendDepth). Also contains the existing
MajorTrend model for backward compatibility.

Hierarchy:
  TrendTree
    └─ TrendNode (depth=1, MAJOR)
         └─ TrendNode (depth=2, SUB)
              └─ TrendNode (depth=3, MICRO)

V1 VALIDATION: TrendNode now validates lifecycle_stage (enum), affected_companies
(dedup/filter), event_5w1h (fill missing keys), buying_intent (structure check),
causal_chain (coerce string→list). Every coercion is logged.

REF: Architecture informed by BERTrend (Boutaleb et al. 2024) for signal
     classification and BERTopic (Grootendorst 2022) for hierarchical topics.
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator
from uuid import UUID, uuid4

from .base import (
    TrendType, Severity, Sector, ConfidenceScore, GeoLocation,
    LifecycleStage,
)

logger = logging.getLogger(__name__)

# Mapping for freeform lifecycle strings to the enum
_LIFECYCLE_ALIASES = {
    "emerging": LifecycleStage.EMERGING,
    "new": LifecycleStage.EMERGING,
    "early": LifecycleStage.EMERGING,
    "nascent": LifecycleStage.EMERGING,
    "growing": LifecycleStage.GROWING,
    "growth": LifecycleStage.GROWING,
    "expanding": LifecycleStage.GROWING,
    "accelerating": LifecycleStage.GROWING,
    "peak": LifecycleStage.PEAK,
    "mature": LifecycleStage.PEAK,
    "peaking": LifecycleStage.PEAK,
    "saturated": LifecycleStage.PEAK,
    "declining": LifecycleStage.DECLINING,
    "decline": LifecycleStage.DECLINING,
    "fading": LifecycleStage.DECLINING,
    "waning": LifecycleStage.DECLINING,
    "dying": LifecycleStage.DECLINING,
}

# Expected keys in event_5w1h dict
_5W1H_KEYS = {"who", "what", "whom", "when", "where", "why", "how"}


# ══════════════════════════════════════════════════════════════════════════════
# NEW ENUMS — Signal Classification
# ══════════════════════════════════════════════════════════════════════════════

class SignalStrength(str, Enum):
    """
    BERTrend-inspired signal classification.

    NOISE: popularity < P10 (10th percentile) — too small to matter
    WEAK:  P10 ≤ popularity ≤ P50 AND positive growth slope — emerging!
    STRONG: popularity > P50 (median) — confirmed trend

    REF: Boutaleb et al., "BERTrend: Neural Topic Modeling for Emerging
         Trends Detection," ACL FuturED Workshop 2024.
         https://arxiv.org/abs/2411.05930
    """
    NOISE = "noise"
    WEAK = "weak"
    STRONG = "strong"


class TrendCorrelation(BaseModel):
    """A correlation between two trend clusters (entity bridge + temporal lag)."""
    source_cluster_id: int = 0
    target_cluster_id: int = 0
    relationship: str = "co-occurs"  # "causes", "caused_by", "amplifies", "co-occurs"
    strength: float = 0.0
    lag_hours: float = 0.0
    bridge_entities: List[str] = Field(default_factory=list)
    sector_chain: Optional[str] = None
    evidence: str = ""  # human-readable explanation


class TrendDepth(str, Enum):
    """
    Depth level in the trend hierarchy.

    Maps to HDBSCAN condensed tree levels:
    - MEGA: entire corpus (depth 0, usually skipped)
    - MAJOR: top-level splits (depth 1) — broad themes
    - SUB: second-level splits (depth 2) — specific topics
    - MICRO: leaf clusters (depth 3) — individual events
    """
    MEGA = "mega"
    MAJOR = "major"
    SUB = "sub"
    MICRO = "micro"


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE LAYER CONTRACTS — Typed interfaces between pipeline layers
# ══════════════════════════════════════════════════════════════════════════════

class TopicCluster(BaseModel):
    """Output of Layer 2 (Cluster): a group of related articles.

    Represents a single cluster from Leiden community detection, with
    its centroid, keywords, and quality metrics. This is the input to
    Layer 3 (Relate) for causal graph construction.
    """
    cluster_id: int
    articles: List[Any] = Field(default_factory=list)  # List[NewsArticle]
    centroid: List[float] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    dominant_event: str = ""
    coherence: float = 0.0
    key_entities: List[str] = Field(default_factory=list)
    article_count: int = 0
    signals: Dict[str, Any] = Field(default_factory=dict)


class TrendEdge(BaseModel):
    """A causal or correlational relationship between two trends.

    Used by Layer 3 (Relate) to build the TrendGraph. Edges represent
    entity bridges, sector chains, or LLM-inferred causal links.

    Relationship types:
      - "causes": Trend A directly leads to Trend B
      - "amplifies": Trend A strengthens the effect of Trend B
      - "mitigates": Trend A reduces the effect of Trend B
      - "co-occurs": Trends share entities but no clear causal direction
    """
    source_trend_id: Any = None           # Cluster ID (int) or UUID
    target_trend_id: Any = None           # Cluster ID (int) or UUID
    relationship_type: str = "co-occurs"  # entity_bridge, sector_chain, causes, amplifies
    strength: float = 0.0                 # 0-1
    evidence: str = ""
    shared_entities: List[str] = Field(default_factory=list)
    detection_method: str = ""            # "entity_overlap", "sector_propagation", "llm_causal"


class TrendGraph(BaseModel):
    """Output of Layer 3+4: trends with relationships and temporal context.

    Extends the flat cluster list with causal edges and cascade paths.
    This is the primary data structure for the enrichment layer.
    """
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    clusters: Dict[int, TopicCluster] = Field(default_factory=dict)
    edges: List[TrendEdge] = Field(default_factory=list)
    cascades: List[List[int]] = Field(default_factory=list)  # Chain-reaction paths

    # Temporal context (populated by Layer 4)
    novelty_scores: Dict[int, float] = Field(default_factory=dict)  # cluster_id → novelty
    continuity_scores: Dict[int, float] = Field(default_factory=dict)

    @property
    def cluster_count(self) -> int:
        return len(self.clusters)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def get_bridge_entities(self) -> List[str]:
        """Entities appearing in 3+ clusters — interconnection hubs."""
        from collections import Counter
        entity_counts: Counter = Counter()
        for cluster in self.clusters.values():
            for entity in cluster.key_entities:
                entity_counts[entity] += 1
        return [e for e, c in entity_counts.most_common() if c >= 3]


# ══════════════════════════════════════════════════════════════════════════════
# TREND NODE — Single node in the trend tree
# ══════════════════════════════════════════════════════════════════════════════

class TrendNode(BaseModel):
    """
    A single node in the hierarchical trend tree.

    Each node represents a cluster of related articles at a specific depth.
    Depth 1 nodes are major themes, depth 2 are sub-topics, depth 3 are
    micro-signals or specific events.

    V1: All LLM-sourced fields have validators that coerce bad types,
    fill missing keys, deduplicate lists, and log every correction.
    """
    # ── Tree structure ──
    id: UUID = Field(default_factory=uuid4)
    parent_id: Optional[UUID] = None
    children_ids: List[UUID] = Field(default_factory=list)
    depth: int = 0
    depth_label: TrendDepth = TrendDepth.MAJOR
    tree_path: str = ""                      # "Theme > Topic > Sub-topic"

    # ── Trend content (from LLM synthesis) ──
    trend_title: str = ""
    trend_summary: str = ""
    actionable_insight: str = ""  # "WHY THIS MATTERS" - specific business action
    trend_type: TrendType = TrendType.EMERGING
    severity: Severity = Severity.MEDIUM
    primary_sectors: List[Sector] = Field(default_factory=list)
    confidence: ConfidenceScore = Field(default_factory=ConfidenceScore)

    # ── Cluster data (from HDBSCAN + c-TF-IDF) ──
    source_articles: List[UUID] = Field(default_factory=list)
    article_count: int = 0
    key_entities: List[str] = Field(default_factory=list)       # Top entities from NER
    key_keywords: List[str] = Field(default_factory=list)       # From c-TF-IDF
    source_diversity: float = 0.0                                # unique_publishers / total

    # ── Signal classification (BERTrend formula) ──
    signal_strength: SignalStrength = SignalStrength.STRONG
    trend_score: float = 0.0                 # Composite importance score
    actionability_score: float = 0.0         # Sales outreach value
    # ── Rich signals (computed by app/trends/signals/) ──
    signals: Dict[str, Any] = Field(default_factory=dict)

    # ── Structured event extraction (from enhanced LLM synthesis) ──
    event_5w1h: Dict[str, str] = Field(default_factory=dict)
    causal_chain: List[str] = Field(default_factory=list)

    # ── Buying intent signals (6sense/Bombora methodology) ──
    buying_intent: Dict[str, str] = Field(default_factory=dict)

    # ── Affected companies and regions ──
    affected_companies: List[str] = Field(default_factory=list)
    affected_regions: List[str] = Field(default_factory=list)
    lifecycle_stage: str = "emerging"

    # ── Article evidence (top-5 snippets for impact council first/second-order ID) ──
    article_snippets: List[str] = Field(default_factory=list)  # "Title: content[:500]"

    # ── Temporal evolution (BERTopic topics_over_time inspired) ──
    temporal_histogram: List[Dict[str, Any]] = Field(default_factory=list)
    velocity_history: List[float] = Field(default_factory=list)
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    momentum_label: str = ""  # "accelerating"|"steady"|"decelerating"|"spiking"

    # ── AI Council Validation (Stage A) ──
    validation_reasoning: str = ""           # Why this classification (from AI council)
    importance_score: float = 0.0            # AI-assessed business significance (0-1)
    validated_event_type: str = ""           # AI-validated event type
    event_type_reasoning: str = ""           # Why this event type
    should_subcluster: bool = False          # AI recommendation on sub-clustering
    subcluster_reason: str = ""              # Why or why not

    # ── V1 Validators ──

    @field_validator('lifecycle_stage', mode='before')
    @classmethod
    def validate_lifecycle_stage(cls, v):
        """Map freeform lifecycle strings to valid values."""
        if v is None:
            return "emerging"
        v_str = str(v).strip().lower()
        if not v_str:
            return "emerging"
        # Check exact match first
        valid = {e.value for e in LifecycleStage}
        if v_str in valid:
            return v_str
        # Check aliases
        mapped = _LIFECYCLE_ALIASES.get(v_str)
        if mapped:
            logger.debug(f"Mapped lifecycle_stage '{v}' → '{mapped.value}'")
            return mapped.value
        logger.warning(f"Invalid lifecycle_stage '{v}', defaulting to 'emerging'")
        return "emerging"

    @field_validator('affected_companies', mode='before')
    @classmethod
    def validate_affected_companies(cls, v):
        """Coerce to list, deduplicate, filter empties and generic names."""
        if v is None:
            return []
        if isinstance(v, str):
            v = [v] if v.strip() else []
        if not isinstance(v, list):
            v = [str(v)] if v else []
        # Filter empties, strip, deduplicate (preserving order)
        seen = set()
        result = []
        for item in v:
            if item is None:
                continue
            name = str(item).strip()
            if not name:
                continue
            name_lower = name.lower()
            if name_lower in seen:
                continue
            seen.add(name_lower)
            result.append(name)
        return result

    @field_validator('affected_regions', mode='before')
    @classmethod
    def validate_affected_regions(cls, v):
        """Coerce to list, filter empties."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v.strip()] if v.strip() else []
        if not isinstance(v, list):
            return [str(v)] if v else []
        return [str(item).strip() for item in v if item and str(item).strip()]

    @field_validator('event_5w1h', mode='before')
    @classmethod
    def validate_event_5w1h(cls, v):
        """Fill missing 5W1H keys with 'Not specified'."""
        if v is None or not isinstance(v, dict):
            return {}
        # Coerce ALL values to str (LLM may return lists for some keys)
        coerced = {}
        for key, val in v.items():
            if isinstance(val, list):
                coerced[key] = ", ".join(str(x) for x in val) if val else "Not specified"
            elif val is None or (isinstance(val, str) and not val.strip()):
                coerced[key] = "Not specified"
            else:
                coerced[key] = str(val).strip()
        # Ensure all expected keys exist
        for key in _5W1H_KEYS:
            if key not in coerced or not coerced[key]:
                coerced[key] = "Not specified"
        return coerced

    @field_validator('buying_intent', mode='before')
    @classmethod
    def validate_buying_intent(cls, v):
        """Validate buying_intent structure."""
        if v is None or not isinstance(v, dict):
            return {}
        # Ensure values are strings
        cleaned = {}
        for key, val in v.items():
            cleaned[str(key)] = str(val) if val is not None else ""
        return cleaned

    @field_validator('causal_chain', mode='before')
    @classmethod
    def validate_causal_chain(cls, v):
        """Coerce string→list, filter empties."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v.strip()] if v.strip() else []
        if not isinstance(v, list):
            return [str(v)] if v else []
        return [str(item).strip() for item in v if item and str(item).strip()]

    @field_validator('key_entities', mode='before')
    @classmethod
    def validate_key_entities(cls, v):
        """Coerce to list, deduplicate, limit to 10."""
        if v is None:
            return []
        if isinstance(v, str):
            v = [v] if v.strip() else []
        if not isinstance(v, list):
            v = [str(v)] if v else []
        seen = set()
        result = []
        for item in v:
            if item is None:
                continue
            name = str(item).strip()
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            result.append(name)
        return result[:10]

    class Config:
        use_enum_values = True


# ══════════════════════════════════════════════════════════════════════════════
# TREND TREE — The full hierarchical structure
# ══════════════════════════════════════════════════════════════════════════════

class TrendTree(BaseModel):
    """
    Complete hierarchical trend tree for a pipeline run.
    """
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Tree structure
    root_ids: List[UUID] = Field(default_factory=list)       # Top-level node IDs
    nodes: Dict[str, TrendNode] = Field(default_factory=dict) # All nodes by str(uuid)

    # Stats
    total_articles_processed: int = 0
    total_articles_after_dedup: int = 0
    total_clusters: int = 0
    max_depth_reached: int = 0
    noise_articles: int = 0              # HDBSCAN outliers (topic -1)

    # Pipeline metadata
    pipeline_elapsed_seconds: float = 0.0
    embedding_model: str = ""
    clustering_algorithm: str = ""

    # Causal reasoning (populated by Stage D Causal Council)
    causal_edges: List[Dict[str, Any]] = Field(default_factory=list)
    cascade_narratives: List[Dict[str, Any]] = Field(default_factory=list)
    causal_council_metrics: Dict[str, Any] = Field(default_factory=dict)

    def to_major_trends(self) -> List["MajorTrend"]:
        """
        Backward-compatible conversion: depth-1 nodes → flat MajorTrend list.
        """
        trends = []
        for root_id in self.root_ids:
            node = self.nodes.get(str(root_id))
            if node and node.depth <= 1:
                trend = MajorTrend(
                    id=node.id,
                    trend_title=node.trend_title,
                    trend_summary=node.trend_summary,
                    trend_type=node.trend_type,
                    severity=node.severity,
                    source_articles=node.source_articles,
                    article_count=node.article_count,
                    key_entities=node.key_entities,
                    key_keywords=node.key_keywords,
                    primary_sectors=node.primary_sectors,
                    confidence=node.confidence,
                    source_diversity_score=node.source_diversity,
                    trend_velocity=node.signals.get("velocity", 0.0),
                    lifecycle_stage=node.lifecycle_stage,
                    affected_regions=node.affected_regions,
                    first_reported=node.first_seen_at,
                    last_updated=node.last_seen_at,
                    event_5w1h=node.event_5w1h,
                    causal_chain=node.causal_chain,
                    buying_intent=node.buying_intent,
                    affected_companies=node.affected_companies,
                    trend_score=node.trend_score,
                    actionability_score=node.actionability_score,
                    actionable_insight=node.actionable_insight,
                    article_snippets=node.article_snippets,
                )
                trends.append(trend)
        trends.sort(
            key=lambda t: self.nodes.get(str(t.id), TrendNode()).trend_score,
            reverse=True,
        )
        return trends

    def get_by_signal(self, strength: SignalStrength) -> List[TrendNode]:
        """Filter nodes by signal strength."""
        return [n for n in self.nodes.values() if n.signal_strength == strength]

    def get_children(self, node_id: UUID) -> List[TrendNode]:
        """Get direct children of a node."""
        node = self.nodes.get(str(node_id))
        if not node:
            return []
        return [
            self.nodes[str(cid)]
            for cid in node.children_ids
            if str(cid) in self.nodes
        ]

    def get_weak_signals(self) -> List[TrendNode]:
        """All nodes classified as weak signals — potential emerging trends."""
        return self.get_by_signal(SignalStrength.WEAK)

    def print_tree(self) -> str:
        """Text-based tree visualization for debugging."""
        lines = []
        signal_icons = {"strong": "+", "weak": "~", "noise": "-"}

        def _walk(node_id: UUID, indent: int = 0):
            node = self.nodes.get(str(node_id))
            if not node:
                return
            icon = signal_icons.get(node.signal_strength, "?")
            prefix = "  " * indent
            lines.append(
                f"{prefix}[{icon}] {node.trend_title} "
                f"({node.article_count} articles, score={node.trend_score:.2f})"
            )
            for child_id in node.children_ids:
                _walk(child_id, indent + 1)

        for root_id in self.root_ids:
            _walk(root_id)
        return "\n".join(lines) if lines else "(empty tree)"


# ══════════════════════════════════════════════════════════════════════════════
# MAJOR TREND — Legacy flat model (backward compat)
# ══════════════════════════════════════════════════════════════════════════════

class MajorTrend(BaseModel):
    """
    Synthesized trend from clustered articles.

    LEGACY: This is the original flat trend model. New code should use
    TrendNode/TrendTree instead. This model is kept for backward compatibility
    with the existing Impact → Company → Contact → Email pipeline.
    """
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Trend identity
    trend_title: str
    trend_summary: str
    trend_type: TrendType = TrendType.GENERAL
    severity: Severity = Severity.MEDIUM

    # Source cluster
    cluster_id: Optional[UUID] = None
    source_articles: List[UUID] = Field(default_factory=list)
    article_count: int = 0

    # Source quality metrics
    source_diversity_score: float = 0.0

    # Extracted intelligence
    key_entities: List[str] = Field(default_factory=list)
    key_keywords: List[str] = Field(default_factory=list)

    # Geographic scope
    geography: GeoLocation = Field(default_factory=GeoLocation)
    is_national: bool = True
    affected_regions: List[str] = Field(default_factory=list)

    # Temporal analysis
    first_reported: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    trend_velocity: float = 0.0
    lifecycle_stage: str = "emerging"

    # Impact scope
    primary_sectors: List[Sector] = Field(default_factory=list)
    secondary_sectors: List[Sector] = Field(default_factory=list)

    # Confidence
    confidence: ConfidenceScore = Field(default_factory=ConfidenceScore)

    # Structured event extraction (carried from TrendNode)
    event_5w1h: Dict[str, str] = Field(default_factory=dict)
    causal_chain: List[str] = Field(default_factory=list)
    buying_intent: Dict[str, str] = Field(default_factory=dict)
    affected_companies: List[str] = Field(default_factory=list)

    # Scoring (carried from TrendNode — needed by downstream TrendData)
    trend_score: float = 0.0
    actionability_score: float = 0.0
    actionable_insight: str = ""

    # Source evidence snippets (carried from TrendNode — for impact council)
    article_snippets: List[str] = Field(default_factory=list)

    # LLM metadata
    llm_model_used: str = ""
    synthesis_tokens: int = 0

    class Config:
        use_enum_values = True
