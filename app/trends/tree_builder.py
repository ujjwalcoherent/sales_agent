"""
TrendTree assembly — converts cluster data into hierarchical TrendNode tree.

Extracted from engine.py for compaction. Shared by both top-level tree
building (Phase 9) and recursive sub-clustering.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.schemas.base import ConfidenceScore, Sector, Severity, TrendType
from app.schemas.news import NewsArticle
from app.schemas.trends import SignalStrength, TrendDepth, TrendNode, TrendTree

logger = logging.getLogger(__name__)

# Sector enum lookup (built once)
_SECTOR_MAP = {v.value.lower(): v for v in Sector}


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """Safely parse ISO datetime string from signals dict. Returns None on failure."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to parse datetime '{value}': {e}")
        return None


def _parse_sectors(raw_sectors: List[str]) -> List[Sector]:
    """Map LLM freeform sector text to closest Sector enum values."""
    valid = []
    for s in raw_sectors:
        s_lower = str(s).lower().strip()
        if s_lower in _SECTOR_MAP:
            valid.append(_SECTOR_MAP[s_lower])
        else:
            matched = next(
                (v for k, v in _SECTOR_MAP.items() if s_lower in k or k.startswith(s_lower)),
                None,
            )
            if matched:
                valid.append(matched)
    return valid


def _source_diversity(articles: List[NewsArticle]) -> float:
    """Compute source diversity ratio for a cluster."""
    unique = len({getattr(a, 'source_id', '') or getattr(a, 'source_name', '') for a in articles})
    return unique / max(len(articles), 1)


def create_trend_node(
    articles: List[NewsArticle],
    keywords: List[str],
    signals: Dict[str, Any],
    summary: Dict[str, Any],
    depth: int,
    parent_id: Optional[uuid.UUID] = None,
    parent_tree_path: str = "",
    parent_sectors: Optional[List[Sector]] = None,
    confidence: float = 0.7,
) -> TrendNode:
    """
    Create a single TrendNode from cluster data. Used by both top-level
    tree assembly and recursive sub-clustering.
    """
    node_id = uuid.uuid4()

    # Title with keyword fallback
    if summary.get("trend_title"):
        title = summary["trend_title"]
    elif keywords:
        clean_kw = [k.title() for k in keywords[:4] if len(k) > 2]
        title = ", ".join(clean_kw) if clean_kw else f"Cluster {node_id.hex[:6]}"
    else:
        title = f"Emerging Trend ({node_id.hex[:6]})"

    # Summary with fallback
    if summary.get("trend_summary"):
        trend_summary = summary["trend_summary"]
    elif articles:
        titles = [a.title for a in articles[:3]]
        trend_summary = f"Cluster of {len(articles)} articles: {'; '.join(titles)}."
    else:
        trend_summary = ""

    # V4: Parse enums with logged fallbacks
    raw_signal = signals.get("signal_strength", "noise")
    try:
        signal_strength = SignalStrength(raw_signal)
    except ValueError:
        logger.warning(f"Invalid signal_strength '{raw_signal}', falling back to NOISE")
        signal_strength = SignalStrength.NOISE

    raw_severity = summary.get("severity", "medium")
    try:
        severity = Severity(str(raw_severity).lower())
    except ValueError:
        logger.warning(f"Invalid severity '{raw_severity}', falling back to MEDIUM")
        severity = Severity.MEDIUM

    raw_trend_type = summary.get("trend_type", "general")
    try:
        trend_type = TrendType(str(raw_trend_type).lower())
    except ValueError:
        logger.warning(f"Invalid trend_type '{raw_trend_type}', falling back to GENERAL")
        trend_type = TrendType.GENERAL

    sectors = _parse_sectors(summary.get("primary_sectors", []))
    depth_label = {1: TrendDepth.MAJOR, 2: TrendDepth.SUB}.get(depth, TrendDepth.MICRO)
    tree_path = f"{parent_tree_path} > {title}" if parent_tree_path else title

    # ── Temporal evolution fields (from histogram computation) ──
    first_seen_at = _parse_iso_datetime(signals.get("first_seen_at"))
    last_seen_at = _parse_iso_datetime(signals.get("last_seen_at"))

    # ── V4: Confidence penalty for incomplete synthesis ──
    confidence_factors: List[str] = []
    adjusted_confidence = confidence

    if summary.get("trend_title"):
        confidence_factors.append("LLM synthesis title present")
    else:
        adjusted_confidence -= 0.15
        confidence_factors.append("Missing synthesis title (-0.15)")

    if summary.get("trend_summary") and len(str(summary["trend_summary"])) > 50:
        confidence_factors.append("Detailed synthesis summary")
    else:
        adjusted_confidence -= 0.10
        confidence_factors.append("Weak/missing synthesis summary (-0.10)")

    if summary.get("event_5w1h") and isinstance(summary["event_5w1h"], dict):
        filled_keys = sum(1 for v in summary["event_5w1h"].values() if v and str(v) != "Not specified")
        if filled_keys >= 5:
            confidence_factors.append(f"Rich 5W1H ({filled_keys}/7 filled)")
        elif filled_keys >= 3:
            confidence_factors.append(f"Partial 5W1H ({filled_keys}/7 filled)")
        else:
            adjusted_confidence -= 0.05
            confidence_factors.append(f"Sparse 5W1H ({filled_keys}/7 filled, -0.05)")
    else:
        adjusted_confidence -= 0.05
        confidence_factors.append("No 5W1H data (-0.05)")

    if summary.get("buying_intent") and isinstance(summary["buying_intent"], dict):
        confidence_factors.append("Buying intent signals present")
    else:
        adjusted_confidence -= 0.05
        confidence_factors.append("No buying intent signals (-0.05)")

    # Source diversity factor
    diversity = _source_diversity(articles)
    if diversity >= 0.5:
        confidence_factors.append(f"High source diversity ({diversity:.2f})")
    elif diversity >= 0.25:
        confidence_factors.append(f"Moderate source diversity ({diversity:.2f})")
    else:
        adjusted_confidence -= 0.05
        confidence_factors.append(f"Low source diversity ({diversity:.2f}, -0.05)")

    # Article count factor
    if len(articles) >= 5:
        confidence_factors.append(f"Strong cluster ({len(articles)} articles)")
    elif len(articles) >= 3:
        confidence_factors.append(f"Moderate cluster ({len(articles)} articles)")
    else:
        adjusted_confidence -= 0.05
        confidence_factors.append(f"Small cluster ({len(articles)} articles, -0.05)")

    adjusted_confidence = max(0.05, min(1.0, adjusted_confidence))

    # ── E1: Extract evidence fields from source articles ──
    source_titles: List[str] = []
    source_urls: List[str] = []
    source_names: List[str] = []
    for a in articles[:20]:  # Cap at 20 for UI display
        if hasattr(a, 'title') and a.title:
            source_titles.append(a.title)
        url = getattr(a, 'url', '') or getattr(a, 'link', '')
        source_urls.append(str(url) if url else "")
        name = getattr(a, 'source_name', '') or getattr(a, 'source_id', '')
        source_names.append(str(name) if name else "unknown")

    return TrendNode(
        id=node_id,
        parent_id=parent_id,
        children_ids=[],
        depth=depth,
        depth_label=depth_label,
        tree_path=tree_path,
        trend_title=title,
        trend_summary=trend_summary,
        actionable_insight=summary.get("actionable_insight", ""),
        trend_type=trend_type,
        severity=severity,
        primary_sectors=sectors or (parent_sectors or []),
        # V4: Confidence with penalty factors and human-readable breakdown
        confidence=ConfidenceScore(score=adjusted_confidence, factors=confidence_factors),
        source_articles=[a.id for a in articles],
        article_count=len(articles),
        key_entities=summary.get("key_entities", [])[:10],
        key_keywords=keywords[:10],
        source_diversity=diversity,
        signal_strength=signal_strength,
        trend_score=signals.get("trend_score", 0.0),
        actionability_score=signals.get("actionability_score", 0.0),
        popularity_score=signals.get("trend_score", 0.0),
        signals=signals,
        event_5w1h=summary.get("event_5w1h", {}),
        causal_chain=summary.get("causal_chain", []),
        buying_intent=summary.get("buying_intent", {}),
        affected_companies=summary.get("affected_companies", []),
        affected_regions=summary.get("affected_regions", []),
        lifecycle_stage=summary.get("lifecycle_stage", "emerging"),
        # E1: Evidence fields — source article details for transparency
        source_article_titles=source_titles,
        source_article_urls=source_urls,
        source_article_sources=source_names,
        # Temporal evolution (T1)
        temporal_histogram=signals.get("temporal_histogram", []),
        velocity_history=signals.get("velocity_history", []),
        first_seen_at=first_seen_at,
        last_seen_at=last_seen_at,
        momentum_label=signals.get("momentum_label", ""),
    )


def build_trend_tree(
    cluster_articles: Dict[int, List[NewsArticle]],
    cluster_keywords: Dict[int, List[str]],
    cluster_signals: Dict[int, Dict[str, Any]],
    cluster_summaries: Dict[int, Dict[str, Any]],
    noise_count: int,
    total_articles: int,
) -> TrendTree:
    """Phase 9: Assemble TrendTree from cluster data."""
    nodes: Dict[str, TrendNode] = {}
    root_ids: List[uuid.UUID] = []

    for cluster_id in sorted(cluster_articles.keys()):
        node = create_trend_node(
            articles=cluster_articles[cluster_id],
            keywords=cluster_keywords.get(cluster_id, []),
            signals=cluster_signals.get(cluster_id, {}),
            summary=cluster_summaries.get(cluster_id, {}),
            depth=1,
        )
        nodes[str(node.id)] = node
        root_ids.append(node.id)

    return TrendTree(
        root_ids=root_ids,
        nodes=nodes,
        total_articles_processed=total_articles,
        total_clusters=len(cluster_articles),
        max_depth_reached=1,
        noise_articles=noise_count,
    )
