"""
Industry Classification — 2-Pass NLI (1st Order + 2nd Order).

After the base NLI filter passes articles as "B2B relevant", this module
applies a SECOND NLI pass to classify them by industry and order:

  1st Order:  The company IN the article IS in the target industry.
              "Sun Pharma launches oncology drug" → 1st order healthcare_pharma.

  2nd Order:  The company services/supplies 1st-order companies.
              "Delhivery signs cold-chain contract with GSK" → 2nd order healthcare_pharma.

Research basis:
  Same model as NLI filter: cross-encoder/nli-deberta-v3-small (arXiv:1909.00161).
  NLI generalizes globally — "company named X" pattern recognition is region-agnostic.
  Pattern: [COMPANY] → action → industry-specific entity = 1st order entailment.

Design:
  - Hypotheses are AUTO-GENERATED from user profile descriptions (not hardcoded).
  - Same NLI model singleton is reused (no second model load).
  - Articles are tagged with industry_label, industry_order, first_order_score,
    second_order_score — downstream agents use these for cluster grouping + pitching.

Usage:
    from app.intelligence.industry_classifier import classify_articles, IndustrySpec

    spec = IndustrySpec(
        industry_id="healthcare_pharma",
        first_order_description="pharmaceutical companies, hospital chains, biotech firms",
        second_order_description="CROs, cold-chain logistics, healthcare IT vendors",
        region="India",
    )
    articles = classify_articles(articles, [spec])
    # Each article now has .industry_label, .industry_order, .first_order_score
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# NLI threshold to assign industry label — must be meaningful, not just any match
_CLASSIFY_THRESHOLD = 0.55
# Minimum 1st-order score to allow 2nd-order classification (prevents noise from getting labeled)
_MIN_FIRST_ORDER_FOR_SECOND = 0.15
# Minimum margin between 1st and 2nd order scores to prefer 1st order
_ORDER_MARGIN = 0.10


@dataclass
class IndustrySpec:
    """Specification for one industry's 1st/2nd order NLI hypotheses.

    Hypotheses are generated from human-readable descriptions, not hardcoded strings.
    This makes the system work for ANY industry without code changes.

    NLI hypothesis design principles (from H_entity_action benchmark, B2B_mean=0.859):
    1. SHORT (< 30 words) — DeBERTa NLI performs best with brief, declarative hypotheses
    2. "named in the text" anchor — factual check that rejects cricket/geopolitics without manual lists
    3. SHORT LABEL (not full list) — "healthcare company" beats "pharmaceutical companies, hospital chains..."
       because NLI grammar: "a specific pharmaceutical companies" confuses the model
    4. Region suffix — narrows hypothesis geography without affecting NLI inference quality
    """
    industry_id: str                # "healthcare_pharma", "fintech", etc.
    short_label: str                # "healthcare or pharmaceutical" — used in hypothesis (SINGULAR, SHORT)
    first_order_description: str    # "pharma manufacturers, hospital chains, biotech firms" — for logging/UI
    second_order_description: str   # "CROs, cold-chain logistics, healthcare IT vendors"
    supply_chain_label: str         # "pharmaceutical or healthcare" — used in 2nd order hypothesis (SHORT)
    region: str = "global"          # ISO code or "global" — wired into hypothesis

    @property
    def first_order_hypothesis(self) -> str:
        """Short, grammatically correct 1st order hypothesis.

        Uses short_label (singular) to avoid grammar confusion in NLI:
        "a specific healthcare company" works; "a specific pharmaceutical companies,
        hospital chains" does not (plural after article confuses model).
        """
        region_str = f" in {self.region}" if self.region.lower() != "global" else ""
        return (
            f"This article reports on a specific {self.short_label} company "
            f"named in the text{region_str} that is growing, raising capital, "
            f"expanding, or making a strategic business move."
        )

    @property
    def second_order_hypothesis(self) -> str:
        """Short, grammatically correct 2nd order hypothesis.

        Stricter than v1: requires a NAMED COMPANY + a SPECIFIC COMMERCIAL RELATIONSHIP
        with the supply chain. Prevents crime/politics/war articles from matching.
        """
        region_str = f" in {self.region}" if self.region.lower() != "global" else ""
        return (
            f"This article is a business news report about a specific company "
            f"named in the text that sells, delivers, or contracts products or services "
            f"to {self.supply_chain_label} companies{region_str} as a vendor or supplier."
        )


# ── Built-in industry specs (users can extend via profile API) ────────────────

BUILT_IN_SPECS: Dict[str, IndustrySpec] = {
    "healthcare_pharma": IndustrySpec(
        industry_id="healthcare_pharma",
        short_label="healthcare or pharmaceutical",
        supply_chain_label="pharmaceutical or healthcare",
        first_order_description="pharmaceutical companies, hospital chains, biotech firms, medical device manufacturers",
        second_order_description="CROs, cold-chain logistics providers, healthcare IT vendors, lab equipment suppliers, pharma distributors",
    ),
    "fintech_bfsi": IndustrySpec(
        industry_id="fintech_bfsi",
        short_label="fintech or financial services",
        supply_chain_label="fintech or banking",
        first_order_description="fintech companies, banks, insurance firms, NBFC lenders, payment processors",
        second_order_description="core banking software vendors, RegTech providers, fraud detection SaaS, payment infrastructure companies",
    ),
    "it_technology": IndustrySpec(
        industry_id="it_technology",
        short_label="enterprise software or B2B technology",  # "B2B technology" avoids cricket/consumer
        supply_chain_label="enterprise software or B2B tech",
        first_order_description="SaaS companies, cloud providers, AI startups, enterprise software firms, IT services companies",
        second_order_description="data center operators, developer tool vendors, cloud infrastructure providers, system integrators",
    ),
    "manufacturing": IndustrySpec(
        industry_id="manufacturing",
        short_label="manufacturing or industrial",
        supply_chain_label="manufacturing or industrial",
        first_order_description="industrial manufacturers, automotive OEMs, electronics manufacturers, heavy equipment companies",
        second_order_description="industrial automation vendors, ERP software for manufacturing, supply chain management platforms, MRO distributors",
    ),
    "logistics_supply_chain": IndustrySpec(
        industry_id="logistics_supply_chain",
        short_label="logistics or supply chain",
        supply_chain_label="logistics or freight",
        first_order_description="logistics companies, freight carriers, warehouse operators, last-mile delivery firms",
        second_order_description="fleet management software vendors, warehouse management system providers, logistics SaaS platforms",
    ),
    "retail_fmcg": IndustrySpec(
        industry_id="retail_fmcg",
        short_label="retail or consumer goods",
        supply_chain_label="retail or consumer goods",
        first_order_description="retail chains, FMCG companies, e-commerce platforms, consumer goods manufacturers",
        second_order_description="retail tech vendors, inventory management SaaS, loyalty platform providers, omnichannel solutions",
    ),
}


def classify_articles(
    articles: List[Any],
    specs: Optional[List[IndustrySpec]] = None,
    batch_size: int = 32,
) -> List[Any]:
    """Classify articles by industry using 2-pass NLI.

    For each industry spec, scores articles against 1st-order and 2nd-order hypotheses.
    Articles are tagged with the best-matching industry label and order.

    Args:
        articles: List of Article objects (must have .title and .summary).
        specs: List of IndustrySpec to classify against. If None, uses all BUILT_IN_SPECS.
        batch_size: NLI batch size (32 = optimal for CPU DeBERTa).

    Returns:
        Same article list with .industry_label, .industry_order,
        .first_order_score, .second_order_score populated.
    """
    if not articles:
        return articles

    if specs is None:
        specs = list(BUILT_IN_SPECS.values())

    if not specs:
        return articles

    from app.intelligence.engine.nli_filter import score_articles as nli_score

    # For each spec: score 1st order, score 2nd order, assign best match
    # Track best match per article across all specs
    best_match: Dict[str, Dict] = {}  # article_id → {label, order, score}

    for spec in specs:
        logger.info(
            f"[industry_clf] Classifying {len(articles)} articles "
            f"for '{spec.industry_id}' (region={spec.region})"
        )

        # 1st order pass
        first_scores = nli_score(
            articles,
            batch_size=batch_size,
            hypothesis=spec.first_order_hypothesis,
        )

        # 2nd order pass
        second_scores = nli_score(
            articles,
            batch_size=batch_size,
            hypothesis=spec.second_order_hypothesis,
        )

        for article, fs, ss in zip(articles, first_scores, second_scores):
            art_id = getattr(article, "id", id(article))

            # Guard: 2nd-order classification requires minimum 1st-order plausibility.
            # If the article has near-zero 1st-order score, the 2nd-order match is
            # almost certainly a false positive (crime/politics/war misclassified as
            # supply chain). The 1st-order hypothesis acts as a semantic gatekeeper.
            if ss >= _CLASSIFY_THRESHOLD and fs < _MIN_FIRST_ORDER_FOR_SECOND:
                continue  # 2nd-order with zero 1st-order context = false positive

            best_score = max(fs, ss)
            if best_score < _CLASSIFY_THRESHOLD:
                continue  # Article doesn't belong to this industry

            # Determine order: prefer 1st unless 2nd is clearly better
            order = 1 if (fs >= ss or (ss - fs) < _ORDER_MARGIN) else 2
            order_score = fs if order == 1 else ss

            prev = best_match.get(art_id)
            if prev is None or order_score > prev["score"]:
                best_match[art_id] = {
                    "label": spec.industry_id,
                    "order": order,
                    "score": order_score,
                    "first_order_score": fs,
                    "second_order_score": ss,
                }

    # Apply best match to articles
    labeled = 0
    first_order_count = 0
    second_order_count = 0

    for article in articles:
        art_id = getattr(article, "id", id(article))
        match = best_match.get(art_id)
        if match:
            article.industry_label = match["label"]
            article.industry_order = match["order"]
            article.first_order_score = match["first_order_score"]
            article.second_order_score = match["second_order_score"]
            labeled += 1
            if match["order"] == 1:
                first_order_count += 1
            else:
                second_order_count += 1

    logger.info(
        f"[industry_clf] Labeled {labeled}/{len(articles)} articles: "
        f"1st_order={first_order_count}, 2nd_order={second_order_count}, "
        f"unlabeled={len(articles) - labeled}"
    )

    return articles


def build_spec_from_profile(
    industry_id: str,
    first_order_description: str,
    second_order_description: str,
    short_label: str = "",
    supply_chain_label: str = "",
    region: str = "global",
) -> IndustrySpec:
    """Create an IndustrySpec from user profile inputs.

    Called by the profile API when a user defines a custom industry target.
    The resulting spec is fed into classify_articles() on each pipeline run.
    """
    # Derive short_label from first item in description if not provided
    _short = short_label or first_order_description.split(",")[0].strip()
    _supply = supply_chain_label or _short
    return IndustrySpec(
        industry_id=industry_id,
        short_label=_short,
        supply_chain_label=_supply,
        first_order_description=first_order_description,
        second_order_description=second_order_description,
        region=region,
    )


def get_spec(industry_id: str) -> Optional[IndustrySpec]:
    """Retrieve a built-in spec by ID, or None if not found."""
    return BUILT_IN_SPECS.get(industry_id)
