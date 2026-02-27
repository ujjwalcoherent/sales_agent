"""
ValidatorAgent — cross-validates LLM synthesis against source articles.

Anti-hallucination gate between trend synthesis and the downstream pipeline
(impact agent, company agent, contact agent, email agent). Prevents fabricated
company names, invented causal chains, and ungrounded trend titles from
reaching sales outreach.

ALGORITHM (zero additional LLM calls for scoring):
    1. Extract ground-truth evidence from source articles:
       - NER entities (ORG, PERSON, GPE, MONEY, LAW, EVENT) via spaCy
       - Keywords from article titles + summaries
       - Content embedding (centroid of article embeddings)
    2. Score each synthesis field against this evidence:
       - Entity fields: Jaccard overlap of NER-extracted entities
       - Text fields: keyword overlap ratio + embedding cosine similarity
       - Structured fields: per-item keyword match
    3. Weighted average -> overall groundedness score
    4. Verdict: PASS / REVISE / REJECT based on configurable thresholds

BACK-AND-FORTH PROTOCOL:
    Round 1: Synthesizer produces output -> Validator scores it
      If PASS  -> done, output accepted
      If REJECT -> done, cluster skipped (score too low to salvage)
      If REVISE -> generate specific feedback listing ungrounded claims
    Round 2: Synthesizer regenerates with feedback injected into prompt
      -> Validator scores again -> final verdict (PASS or REJECT, no more rounds)

BUDGET: 0 LLM calls for validation itself. The only LLM cost is the
        optional regeneration call (max 1 extra call per cluster).

ENV VARS (all in app/config.py):
    VALIDATOR_ENABLED: bool = True
    VALIDATOR_MAX_ROUNDS: int = 2
    VALIDATOR_PASS_THRESHOLD: float = 0.6
    VALIDATOR_REJECT_THRESHOLD: float = 0.25
    VALIDATOR_ENTITY_OVERLAP_MIN: float = 0.4
    VALIDATOR_WEIGHT_ENTITY: float = 0.35
    VALIDATOR_WEIGHT_KEYWORD: float = 0.30
    VALIDATOR_WEIGHT_EMBEDDING: float = 0.35
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from app.config import get_settings
from app.schemas.validation import (
    FieldGroundedness,
    ValidationResult,
    ValidationRound,
    ValidationVerdict,
)
from app.tools.embeddings import EmbeddingTool

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# TEXT PROCESSING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

# Common stopwords to exclude from keyword extraction
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "about", "up",
    "that", "this", "these", "those", "it", "its", "he", "she", "they",
    "we", "you", "i", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their", "what", "which", "who", "whom", "also",
    "said", "says", "according", "per", "new", "like", "one", "two",
    "company", "companies", "india", "indian", "market", "business",
    "year", "years", "also", "including", "among", "across", "however",
})


def _extract_keywords(text: str, max_count: int = 50) -> Set[str]:
    """
    Extract meaningful keywords from text.

    Simple but effective: lowercase, remove punctuation, filter stopwords,
    keep tokens 3+ chars. Returns a set for O(1) lookup.
    """
    if not text:
        return set()
    # Lowercase and split on non-alphanumeric
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    # Filter stopwords and short tokens
    keywords = {t for t in tokens if t not in _STOPWORDS and len(t) >= 3}
    # Also extract bigrams for compound terms (e.g., "digital lending", "supply chain")
    words = [t for t in tokens if t not in _STOPWORDS and len(t) >= 3]
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        keywords.add(bigram)
    # Cap to avoid memory issues on very large texts
    if len(keywords) > max_count:
        # Keep the most frequent (approximation: just truncate set, order is arbitrary)
        return set(list(keywords)[:max_count])
    return keywords


def _normalize_entity(name: str) -> str:
    """Normalize entity name for comparison (lowercase + suffix stripping + aliases)."""
    from app.news.entity_normalizer import normalize_entity
    return normalize_entity(name).lower().strip()


def _entity_match(claimed: str, source_entities: Set[str]) -> bool:
    """
    Check if a claimed entity matches any source entity.

    Uses substring matching in both directions to handle cases like:
    - Claimed "RBI" matches source "Reserve Bank of India" (abbreviation)
    - Claimed "Tata Electronics" matches source "Tata" (partial)
    - Claimed "SEBI" matches source "SEBI" (exact)
    """
    claimed_norm = _normalize_entity(claimed)
    if not claimed_norm or len(claimed_norm) < 2:
        return False

    for source_ent in source_entities:
        # Exact match
        if claimed_norm == source_ent:
            return True
        # Claimed is substring of source (e.g., "Tata" in "Tata Electronics Pvt Ltd")
        if claimed_norm in source_ent:
            return True
        # Source is substring of claimed (e.g., "RBI" in "RBI regulation mandate")
        if source_ent in claimed_norm and len(source_ent) >= 3:
            return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE EVIDENCE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

class SourceEvidence:
    """
    Pre-computed evidence from source articles for fast validation.

    Built once per cluster, reused across validation rounds.
    """

    def __init__(
        self,
        articles: List,
        embedding_tool: Optional[EmbeddingTool] = None,
    ):
        self.article_count = len(articles)

        # 1. Collect all NER entities from articles (already extracted by pipeline)
        self.entities_raw: List[str] = []
        self.entities_by_type: Dict[str, Set[str]] = {}
        self.entity_set: Set[str] = set()  # Normalized for matching

        for article in articles:
            for entity in getattr(article, 'entities', []):
                ent_text = entity.text if hasattr(entity, 'text') else str(entity)
                ent_type = entity.type if hasattr(entity, 'type') else "UNKNOWN"
                self.entities_raw.append(ent_text)
                self.entities_by_type.setdefault(ent_type, set()).add(
                    _normalize_entity(ent_text)
                )
                self.entity_set.add(_normalize_entity(ent_text))

            # Also include entity_names (flat list on NewsArticle)
            for name in getattr(article, 'entity_names', []):
                self.entity_set.add(_normalize_entity(name))

            # Include mentioned_companies for ORG matching
            for name in getattr(article, 'mentioned_companies', []):
                self.entity_set.add(_normalize_entity(name))
                self.entities_by_type.setdefault("ORG", set()).add(
                    _normalize_entity(name)
                )

        # 2. Extract keywords from all article titles + summaries
        all_text_parts = []
        for article in articles:
            all_text_parts.append(getattr(article, 'title', ''))
            all_text_parts.append(getattr(article, 'summary', ''))
            content = getattr(article, 'content', None)
            if content:
                # Only first 500 chars of content (titles + summaries are more signal-dense)
                all_text_parts.append(content[:500])
        combined_text = " ".join(all_text_parts)
        self.keywords: Set[str] = _extract_keywords(combined_text, max_count=200)

        # Also collect article-level keywords
        for article in articles:
            for kw in getattr(article, 'keywords', []):
                self.keywords.add(kw.lower().strip())

        # 3. Compute centroid embedding (average of article embeddings)
        self.centroid_embedding: Optional[List[float]] = None
        if embedding_tool:
            embeddings = []
            for article in articles:
                emb = getattr(article, 'content_embedding', None) or getattr(
                    article, 'title_embedding', None
                )
                if emb is not None and len(emb) > 0 and np.any(np.array(emb) != 0):
                    embeddings.append(emb)
            if embeddings:
                arr = np.array(embeddings)
                centroid = arr.mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                self.centroid_embedding = centroid.tolist()

        logger.debug(
            f"SourceEvidence: {self.article_count} articles, "
            f"{len(self.entity_set)} unique entities, "
            f"{len(self.keywords)} keywords, "
            f"embedding={'yes' if self.centroid_embedding else 'no'}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# FIELD-LEVEL GROUNDEDNESS SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _score_entity_field(
    field_name: str,
    claimed_entities: List[str],
    evidence: SourceEvidence,
) -> FieldGroundedness:
    """
    Score an entity list field (key_entities, affected_companies) against
    source NER entities using fuzzy matching.

    Score = (matched entities) / (total claimed entities)
    """
    if not claimed_entities:
        # No claims = nothing to hallucinate = pass
        return FieldGroundedness(
            field_name=field_name,
            score=1.0,
            method="ner_overlap",
            detail="No entities claimed (vacuously grounded)",
        )

    matched = []
    ungrounded = []

    for entity in claimed_entities:
        if _entity_match(entity, evidence.entity_set):
            matched.append(entity)
        else:
            ungrounded.append(entity)

    score = len(matched) / len(claimed_entities) if claimed_entities else 0.0

    return FieldGroundedness(
        field_name=field_name,
        score=score,
        method="ner_overlap",
        evidence_found=matched[:10],
        ungrounded_items=ungrounded[:10],
        detail=(
            f"{len(matched)}/{len(claimed_entities)} entities found in sources. "
            f"Ungrounded: {', '.join(ungrounded[:5])}" if ungrounded else
            f"All {len(matched)} entities verified in sources."
        ),
    )


def _score_text_field(
    field_name: str,
    synthesis_text: str,
    evidence: SourceEvidence,
    embedding_tool: Optional[EmbeddingTool] = None,
) -> FieldGroundedness:
    """
    Score a text field (trend_title, trend_summary) using keyword overlap
    and optionally embedding similarity.

    Keyword score: |synthesis_keywords & source_keywords| / |synthesis_keywords|
    Embedding score: cosine_similarity(synthesis_embedding, source_centroid)

    Combined: 0.5 * keyword_score + 0.5 * embedding_score (if available)
    """
    if not synthesis_text or not synthesis_text.strip():
        return FieldGroundedness(
            field_name=field_name,
            score=0.0,
            method="keyword_overlap",
            detail="Empty text field",
        )

    synth_keywords = _extract_keywords(synthesis_text, max_count=100)

    if not synth_keywords:
        return FieldGroundedness(
            field_name=field_name,
            score=0.5,  # Benefit of the doubt for very short text
            method="keyword_overlap",
            detail="No extractable keywords from synthesis text",
        )

    # Keyword overlap
    overlap = synth_keywords & evidence.keywords
    keyword_score = len(overlap) / len(synth_keywords) if synth_keywords else 0.0

    # Embedding similarity (if available)
    embedding_score = None
    if embedding_tool and evidence.centroid_embedding:
        try:
            synth_embedding = embedding_tool.embed_text(synthesis_text[:500])
            if synth_embedding and any(v != 0 for v in synth_embedding):
                embedding_score = embedding_tool.compute_similarity(
                    synth_embedding, evidence.centroid_embedding
                )
                # Remap from [-1,1] to [0,1] (cosine sim can be negative)
                embedding_score = (embedding_score + 1.0) / 2.0
        except Exception as e:
            logger.debug(f"Embedding similarity failed for {field_name}: {e}")

    # Combine scores
    if embedding_score is not None:
        combined = 0.5 * keyword_score + 0.5 * embedding_score
        method = "combined"
    else:
        combined = keyword_score
        method = "keyword_overlap"

    # Find evidence snippets (keywords that matched)
    evidence_found = sorted(list(overlap))[:10]

    # Find ungrounded keywords (in synthesis but not in sources)
    ungrounded = sorted(list(synth_keywords - evidence.keywords))[:10]

    detail_parts = [f"keyword_overlap={keyword_score:.2f} ({len(overlap)}/{len(synth_keywords)})"]
    if embedding_score is not None:
        detail_parts.append(f"embedding_sim={embedding_score:.2f}")

    return FieldGroundedness(
        field_name=field_name,
        score=combined,
        method=method,
        evidence_found=evidence_found,
        ungrounded_items=ungrounded,
        detail=", ".join(detail_parts),
    )


def _score_list_field(
    field_name: str,
    items: List[str],
    evidence: SourceEvidence,
) -> FieldGroundedness:
    """
    Score a structured list field (causal_chain, key_facts) by checking
    keyword overlap for each item individually.

    Score = mean of per-item keyword overlap scores.
    """
    if not items:
        return FieldGroundedness(
            field_name=field_name,
            score=1.0,
            method="keyword_overlap",
            detail="No items claimed (vacuously grounded)",
        )

    item_scores = []
    evidence_found = []
    ungrounded = []

    for item in items:
        item_keywords = _extract_keywords(str(item), max_count=30)
        if not item_keywords:
            item_scores.append(0.5)  # Benefit of the doubt
            continue
        overlap = item_keywords & evidence.keywords
        ratio = len(overlap) / len(item_keywords) if item_keywords else 0.0
        item_scores.append(ratio)

        if ratio >= 0.3:
            evidence_found.append(str(item)[:80])
        else:
            ungrounded.append(str(item)[:80])

    avg_score = sum(item_scores) / len(item_scores) if item_scores else 0.0

    return FieldGroundedness(
        field_name=field_name,
        score=avg_score,
        method="keyword_overlap",
        evidence_found=evidence_found[:10],
        ungrounded_items=ungrounded[:10],
        detail=f"Per-item keyword overlap: avg={avg_score:.2f} over {len(items)} items",
    )


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATOR AGENT
# ══════════════════════════════════════════════════════════════════════════════

class ValidatorAgent:
    """
    Cross-validates LLM synthesis output against source article evidence.

    Usage:
        validator = ValidatorAgent()
        result = validator.validate(synthesis_dict, source_articles)
        if result.final_verdict == ValidationVerdict.REVISE:
            # Get feedback to inject into re-synthesis prompt
            feedback = result.rounds[-1].feedback
    """

    def __init__(self, embedding_tool: Optional[EmbeddingTool] = None):
        self.settings = get_settings()
        self.embedding_tool = embedding_tool or EmbeddingTool()

        # Load thresholds from config
        self.pass_threshold = self.settings.validator_pass_threshold
        self.reject_threshold = self.settings.validator_reject_threshold
        self.entity_overlap_min = self.settings.validator_entity_overlap_min

        # Scoring weights
        self.weight_entity = self.settings.validator_weight_entity
        self.weight_keyword = self.settings.validator_weight_keyword
        self.weight_embedding = self.settings.validator_weight_embedding

        # Normalize weights to sum to 1.0
        total = self.weight_entity + self.weight_keyword + self.weight_embedding
        if total > 0:
            self.weight_entity /= total
            self.weight_keyword /= total
            self.weight_embedding /= total

    def validate(
        self,
        synthesis: Dict[str, Any],
        articles: List,
        cluster_id: Optional[int] = None,
    ) -> ValidationResult:
        """
        Validate a single synthesis output against its source articles.

        This is the main entry point. Does NOT call the LLM.

        Args:
            synthesis: Dict from LLM synthesis (trend_title, key_entities, etc.)
            articles: Source NewsArticle objects for this cluster
            cluster_id: Optional HDBSCAN cluster label for tracking

        Returns:
            ValidationResult with verdict, scores, and feedback
        """
        t0 = time.time()

        # Build evidence once (reused if validate is called in a loop)
        evidence = SourceEvidence(articles, self.embedding_tool)

        # Score all fields
        round_result = self._score_synthesis(synthesis, evidence)

        elapsed_ms = int((time.time() - t0) * 1000)

        result = ValidationResult(
            cluster_id=cluster_id,
            rounds=[round_result],
            total_rounds=1,
            final_verdict=round_result.verdict,
            final_score=round_result.overall_score,
            source_entity_count=len(evidence.entity_set),
            source_keyword_count=len(evidence.keywords),
            synthesis_entity_count=len(
                synthesis.get("key_entities", []) +
                synthesis.get("affected_companies", [])
            ),
            entity_overlap_ratio=self._get_entity_overlap_ratio(
                synthesis, evidence
            ),
            elapsed_ms=elapsed_ms,
        )

        logger.info(
            f"Validation [{result.final_verdict.value.upper()}]: "
            f"score={result.final_score:.2f}, "
            f"entities={result.entity_overlap_ratio:.0%}, "
            f"elapsed={elapsed_ms}ms"
        )

        return result

    def validate_with_revision(
        self,
        synthesis: Dict[str, Any],
        articles: List,
        revised_synthesis: Optional[Dict[str, Any]] = None,
        previous_result: Optional[ValidationResult] = None,
        cluster_id: Optional[int] = None,
    ) -> ValidationResult:
        """
        Validate a revised synthesis (round 2+).

        Called after the synthesizer has regenerated with feedback.
        Adds a new round to the existing ValidationResult.

        Args:
            synthesis: The revised synthesis dict
            articles: Same source articles
            revised_synthesis: (unused, kept for interface clarity)
            previous_result: ValidationResult from previous round
            cluster_id: Optional cluster ID

        Returns:
            Updated ValidationResult with new round appended
        """
        t0 = time.time()
        evidence = SourceEvidence(articles, self.embedding_tool)
        round_result = self._score_synthesis(synthesis, evidence)

        elapsed_ms = int((time.time() - t0) * 1000)

        if previous_result:
            result = previous_result
            result.rounds.append(round_result)
            result.total_rounds = len(result.rounds)
            result.final_verdict = round_result.verdict
            result.final_score = round_result.overall_score
            result.elapsed_ms += elapsed_ms
        else:
            result = ValidationResult(
                cluster_id=cluster_id,
                rounds=[round_result],
                total_rounds=1,
                final_verdict=round_result.verdict,
                final_score=round_result.overall_score,
                source_entity_count=len(evidence.entity_set),
                source_keyword_count=len(evidence.keywords),
                synthesis_entity_count=len(
                    synthesis.get("key_entities", []) +
                    synthesis.get("affected_companies", [])
                ),
                entity_overlap_ratio=self._get_entity_overlap_ratio(
                    synthesis, evidence
                ),
                elapsed_ms=elapsed_ms,
            )

        logger.info(
            f"Validation round {result.total_rounds} "
            f"[{result.final_verdict.value.upper()}]: "
            f"score={result.final_score:.2f}"
        )

        return result

    def build_revision_feedback(self, validation_round: ValidationRound) -> str:
        """
        Build a revision prompt from validation feedback.

        This string is injected into the synthesis prompt on retry so the LLM
        knows exactly what to fix. Specific and actionable.

        Returns:
            Feedback string to append to the synthesis prompt.
        """
        parts = [
            "IMPORTANT: Your previous synthesis had groundedness issues. "
            "Fix the following problems using ONLY information from the source articles:"
        ]

        for fb in validation_round.feedback:
            parts.append(f"- {fb}")

        parts.append(
            "\nDo NOT invent company names, statistics, or causal relationships "
            "that are not supported by the articles above. If information is not "
            "available in the articles, say 'Not specified' instead of fabricating."
        )

        return "\n".join(parts)

    # ── INTERNAL METHODS ──────────────────────────────────────────────────────

    def _score_synthesis(
        self,
        synthesis: Dict[str, Any],
        evidence: SourceEvidence,
    ) -> ValidationRound:
        """
        Score all fields and compute overall groundedness + verdict.
        """
        field_scores: List[FieldGroundedness] = []

        # 1. Entity fields (NER overlap)
        key_entities = synthesis.get("key_entities", [])
        if key_entities:
            field_scores.append(
                _score_entity_field("key_entities", key_entities, evidence)
            )

        affected_companies = synthesis.get("affected_companies", [])
        if affected_companies:
            field_scores.append(
                _score_entity_field("affected_companies", affected_companies, evidence)
            )

        # 2. Text fields (keyword + embedding)
        trend_title = synthesis.get("trend_title", "")
        if trend_title:
            field_scores.append(
                _score_text_field("trend_title", trend_title, evidence, self.embedding_tool)
            )

        trend_summary = synthesis.get("trend_summary", "")
        if trend_summary:
            field_scores.append(
                _score_text_field("trend_summary", trend_summary, evidence, self.embedding_tool)
            )

        actionable_insight = synthesis.get("actionable_insight", "")
        if actionable_insight:
            field_scores.append(
                _score_text_field("actionable_insight", actionable_insight, evidence, self.embedding_tool)
            )

        # 3. Structured list fields (per-item keyword overlap)
        causal_chain = synthesis.get("causal_chain", [])
        if causal_chain and isinstance(causal_chain, list):
            field_scores.append(
                _score_list_field("causal_chain", causal_chain, evidence)
            )

        key_facts = synthesis.get("key_facts", [])
        if key_facts and isinstance(key_facts, list):
            field_scores.append(
                _score_list_field("key_facts", key_facts, evidence)
            )

        # 4. Compute weighted overall score
        # Group scores by category for weighting
        entity_scores = [
            fs.score for fs in field_scores
            if fs.method == "ner_overlap"
        ]
        keyword_scores = [
            fs.score for fs in field_scores
            if fs.method == "keyword_overlap"
        ]
        embedding_scores = [
            fs.score for fs in field_scores
            if fs.method == "combined"
        ]

        # Fallback: if no embedding scores, redistribute weight to keyword
        effective_weight_entity = self.weight_entity
        effective_weight_keyword = self.weight_keyword
        effective_weight_embedding = self.weight_embedding

        if not embedding_scores:
            effective_weight_keyword += effective_weight_embedding
            effective_weight_embedding = 0.0

        if not entity_scores:
            effective_weight_keyword += effective_weight_entity
            effective_weight_entity = 0.0

        overall = 0.0
        if entity_scores:
            overall += effective_weight_entity * (sum(entity_scores) / len(entity_scores))
        if keyword_scores:
            overall += effective_weight_keyword * (sum(keyword_scores) / len(keyword_scores))
        if embedding_scores:
            overall += effective_weight_embedding * (sum(embedding_scores) / len(embedding_scores))

        # If we had no scores at all, give a neutral score
        if not field_scores:
            overall = 0.5

        overall = max(0.0, min(1.0, overall))

        # 5. Determine verdict
        verdict, feedback = self._determine_verdict(overall, field_scores, evidence, synthesis)

        return ValidationRound(
            round_number=1,  # Caller updates for round 2+
            verdict=verdict,
            overall_score=overall,
            field_scores=field_scores,
            feedback=feedback,
        )

    def _determine_verdict(
        self,
        overall_score: float,
        field_scores: List[FieldGroundedness],
        evidence: SourceEvidence,
        synthesis: Dict[str, Any],
    ) -> Tuple[ValidationVerdict, List[str]]:
        """
        Determine PASS / REVISE / REJECT and generate specific feedback.
        """
        feedback: List[str] = []

        # Hard reject if score is too low
        if overall_score < self.reject_threshold:
            feedback.append(
                f"Overall groundedness score ({overall_score:.2f}) is below "
                f"reject threshold ({self.reject_threshold}). "
                f"Synthesis is too fabricated to salvage."
            )
            return ValidationVerdict.REJECT, feedback

        # Check entity overlap specifically
        entity_overlap = self._get_entity_overlap_ratio(synthesis, evidence)
        if entity_overlap < self.entity_overlap_min:
            feedback.append(
                f"Entity overlap is only {entity_overlap:.0%} "
                f"(need {self.entity_overlap_min:.0%}). "
                f"Many claimed entities are not found in source articles."
            )

        # Build specific feedback for failing fields
        for fs in field_scores:
            if fs.score < 0.5 and fs.ungrounded_items:
                if fs.field_name == "affected_companies":
                    feedback.append(
                        f"These company names were NOT found in source articles and may be fabricated: "
                        f"{', '.join(fs.ungrounded_items[:5])}. "
                        f"Only include companies explicitly mentioned in the articles."
                    )
                elif fs.field_name == "key_entities":
                    feedback.append(
                        f"These entities have no source evidence: "
                        f"{', '.join(fs.ungrounded_items[:5])}. "
                        f"Stick to entities actually mentioned in the articles."
                    )
                elif fs.field_name == "causal_chain":
                    feedback.append(
                        f"Parts of the causal chain lack source evidence: "
                        f"{'; '.join(fs.ungrounded_items[:3])}. "
                        f"Base each step on facts from the articles."
                    )
                elif fs.field_name in ("trend_title", "trend_summary"):
                    feedback.append(
                        f"The {fs.field_name} uses terms not found in source articles. "
                        f"Ensure the title/summary directly reflects article content."
                    )

        # Pass or revise
        if overall_score >= self.pass_threshold and not feedback:
            return ValidationVerdict.PASS, []

        if overall_score >= self.pass_threshold and feedback:
            # Score is above threshold but there are specific issues.
            # If entity issues are minor, still pass with a warning logged.
            critical_feedback = [
                f for f in feedback
                if "fabricated" in f.lower() or "NOT found" in f
            ]
            if not critical_feedback:
                logger.info(
                    f"Score {overall_score:.2f} >= {self.pass_threshold} "
                    f"with minor feedback, passing with warnings"
                )
                return ValidationVerdict.PASS, feedback

        # Default: REVISE with feedback
        if not feedback:
            feedback.append(
                f"Overall groundedness score ({overall_score:.2f}) is below "
                f"the pass threshold ({self.pass_threshold}). "
                f"Ensure all claims are grounded in the source articles."
            )

        return ValidationVerdict.REVISE, feedback

    def _get_entity_overlap_ratio(
        self,
        synthesis: Dict[str, Any],
        evidence: "SourceEvidence",
    ) -> float:
        """
        Compute what fraction of synthesis-claimed entities appear in sources.
        """
        all_claimed = (
            synthesis.get("key_entities", []) +
            synthesis.get("affected_companies", [])
        )
        if not all_claimed:
            return 1.0  # Nothing claimed = nothing hallucinated

        matched = sum(
            1 for entity in all_claimed
            if _entity_match(entity, evidence.entity_set)
        )
        return matched / len(all_claimed)
