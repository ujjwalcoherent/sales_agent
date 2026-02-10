"""
Embedding-based event classification for news articles.

Instead of hardcoded regex patterns, this uses the EMBEDDING MODEL to
semantically match articles against event type descriptions. Just like
AlphaSense's "Smart Synonyms" — the AI understands meaning, not words.

HOW IT WORKS:
  1. Each event type has MULTIPLE semantic description variants (V6)
  2. Article titles+summaries are embedded using the same model as clustering
  3. Cosine similarity finds the closest event type (max across variants)
  4. Disambiguation logic handles close scores (V6)
  5. Articles get tagged with the best-matching event type

V6 IMPROVEMENTS:
  - Multiple description variants per event type (averaged embedding)
  - Configurable threshold from env (EVENT_CLASSIFIER_THRESHOLD)
  - Disambiguation: if top-2 scores within margin, check keyword boost
  - Higher default threshold (0.35 vs 0.20) to reduce misclassification
  - Structured logging with per-article confidence

PERFORMANCE: ~1-2s for 500 articles (batched embedding + vector math)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# V6: Event types with urgency scores, MULTIPLE description variants, and keyword boosters.
# Multiple variants per event type improve embedding coverage — the final
# event embedding is the average of all variant embeddings.
# Keyword boosters give a small score bonus when exact keywords appear in the text.
EVENTS = {
    "regulation": {
        "urgency": 0.95,
        "descriptions": [
            "Government regulation, compliance mandate, regulatory change, central bank policy, financial authority guidelines, penalties, KYC AML requirements, licensing rules, industry standards enforcement",
            "SEBI regulation, RBI circular, TRAI directive, FSSAI compliance, government mandate on industry, new compliance requirements for companies",
            "Regulatory crackdown, policy tightening, mandatory disclosure, data protection law, environmental regulation, labor law reform",
        ],
        "keyword_boost": ["regulation", "regulatory", "compliance", "mandate", "sebi", "rbi", "trai", "fssai", "norms", "guidelines", "circular", "directive"],
    },
    "funding": {
        "urgency": 0.70,
        "descriptions": [
            "Startup funding round, venture capital investment, Series A B C D, company valuation, unicorn status, fundraising, seed round, private equity deal, growth capital",
            "Company raises million dollars in funding, investor led round, pre-IPO placement, bridge round, convertible note funding announcement",
            "Venture capital firm invests in startup, growth equity round, late stage funding, angel investment deal closed",
        ],
        "keyword_boost": ["funding", "raises", "raised", "investment", "series", "venture", "capital", "investor", "valuation", "unicorn", "seed"],
    },
    "expansion": {
        "urgency": 0.75,
        "descriptions": [
            "Business expansion, opening new office plant factory, entering new market, scaling operations, global expansion, geographic growth, new product line",
            "Company opens new manufacturing facility, expands operations to new city state country, scales workforce and infrastructure",
            "New branch office opening, retail store expansion, capacity expansion, production ramp-up, market penetration strategy",
        ],
        "keyword_boost": ["expansion", "expands", "opens", "new office", "new plant", "new facility", "scaling", "entering"],
    },
    "acquisition": {
        "urgency": 0.80,
        "descriptions": [
            "Company acquisition, corporate merger, buyout, takeover bid, M&A deal, consolidation, asset purchase, strategic acquisition",
            "Company acquires rival firm, merger agreement signed, takeover offer accepted, asset acquisition completed",
            "Corporate consolidation deal, hostile takeover bid, friendly merger, acqui-hire talent acquisition",
        ],
        "keyword_boost": ["acquisition", "acquires", "merger", "buyout", "takeover", "M&A", "merged", "acquire"],
    },
    "leadership_change": {
        "urgency": 0.65,
        "descriptions": [
            "New CEO CTO CFO COO appointed, executive resignation, leadership transition, board reshuffle, management change, founder stepping down",
            "Company appoints new chief executive officer, senior leadership change announced, board of directors reconstituted",
            "CEO resignation, founder exits company, new managing director appointed, CXO level hiring announcement",
        ],
        "keyword_boost": ["appoints", "appointed", "CEO", "CTO", "CFO", "COO", "resignation", "steps down", "new head"],
    },
    "layoffs": {
        "urgency": 0.85,
        "descriptions": [
            "Company layoffs, job cuts, workforce reduction, restructuring, downsizing, mass termination, cost cutting through headcount",
            "Tech company fires hundreds of employees, mass layoffs announced, workforce reduction plan, hiring freeze",
            "Restructuring leads to job losses, company shuts down division, employee termination drive, headcount reduction",
        ],
        "keyword_boost": ["layoffs", "layoff", "fires", "fired", "job cuts", "workforce reduction", "downsizing", "restructuring"],
    },
    "ipo": {
        "urgency": 0.80,
        "descriptions": [
            "IPO initial public offering, stock market listing, DRHP filing, company going public, secondary offering, stock market debut",
            "Company files draft red herring prospectus, IPO subscription opens, stock exchange listing date announced",
            "Startup plans IPO, company valued at billion before listing, post-IPO trading, grey market premium",
        ],
        "keyword_boost": ["IPO", "listing", "DRHP", "going public", "public offering", "stock market debut", "listed"],
    },
    "technology": {
        "urgency": 0.60,
        "descriptions": [
            "AI artificial intelligence adoption, digital transformation, cloud migration, cybersecurity threat, generative AI, technology disruption, automation, semiconductor, SaaS platform, tech innovation",
            "Company launches AI-powered product, enterprise digital transformation initiative, cloud-first strategy, robotic process automation",
            "New technology platform launched, open source release, API-first product, machine learning model deployment, tech stack modernization",
        ],
        "keyword_boost": ["AI", "artificial intelligence", "digital", "cloud", "cybersecurity", "automation", "SaaS", "machine learning", "generative"],
    },
    "partnership": {
        "urgency": 0.50,
        "descriptions": [
            "Strategic business partnership, corporate alliance, industry collaboration, joint venture, memorandum of understanding, distribution agreement",
            "Two companies sign partnership deal, strategic alliance formed, collaboration agreement for joint product development",
            "MOU signed between firms, distribution partnership, technology licensing deal, co-development agreement",
        ],
        "keyword_boost": ["partnership", "partners", "alliance", "joint venture", "MOU", "collaboration", "tie-up"],
    },
    "crisis": {
        "urgency": 0.90,
        "descriptions": [
            "Corporate fraud, financial scam, accounting scandal, data breach, bankruptcy filing, loan default, regulatory violation, corporate crisis",
            "Company faces investigation for financial irregularities, data leak affects millions, corporate governance failure",
            "Bankruptcy proceedings initiated, company defaults on debt, management fraud exposed, stock price crashes on scandal",
        ],
        "keyword_boost": ["fraud", "scam", "scandal", "breach", "bankruptcy", "default", "violation", "crisis", "investigation"],
    },
    "supply_chain": {
        "urgency": 0.75,
        "descriptions": [
            "Supply chain disruption, procurement challenge, raw material shortage, logistics bottleneck, commodity price volatility, sourcing difficulty, warehouse and distribution issues",
            "Global supply chain crisis affects Indian manufacturers, semiconductor shortage delays production, shipping container shortage",
            "Raw material costs surge, supply bottleneck, inventory shortage, port congestion, freight rate increase",
        ],
        "keyword_boost": ["supply chain", "procurement", "shortage", "logistics", "bottleneck", "raw material", "sourcing"],
    },
    "price_change": {
        "urgency": 0.65,
        "descriptions": [
            "Price hike, inflation impact, commodity price surge, margin pressure, cost increase passed to consumers, price war, deflation, tariff impact",
            "Company raises product prices citing inflation, input cost increase leads to price revision, competitive pricing pressure",
            "Fuel price hike, steel price increase, food inflation, consumer goods price revision, tariff duty change",
        ],
        "keyword_boost": ["price", "hike", "inflation", "cost increase", "tariff", "price war", "margin"],
    },
    "consumer_shift": {
        "urgency": 0.55,
        "descriptions": [
            "Consumer behavior change, D2C direct-to-consumer growth, e-commerce boom, quick commerce, changing spending patterns, brand switching, digital adoption",
            "Shift in consumer preferences, online shopping growth, tier-2 tier-3 market demand, premium segment growth",
            "Digital payments adoption surge, consumer confidence index, rural demand revival, festival season spending trends",
        ],
        "keyword_boost": ["consumer", "D2C", "e-commerce", "spending", "demand", "shopping", "brand", "quick commerce"],
    },
    "market_entry": {
        "urgency": 0.70,
        "descriptions": [
            "Cross-border market entry, foreign direct investment, government incentive scheme, international expansion, trade agreement, export opportunity, new geography launch",
            "Foreign company enters Indian market, Make in India initiative, PLI scheme beneficiary, FDI inflow announcement",
            "Company launches in new country, trade deal signed, export promotion, bilateral agreement, market access",
        ],
        "keyword_boost": ["market entry", "FDI", "PLI", "Make in India", "export", "trade agreement", "international", "cross-border"],
    },
}

# Pre-extracted for quick access
EVENT_URGENCY = {etype: info["urgency"] for etype, info in EVENTS.items()}

# V6: Flatten all descriptions for embedding (used by classifier)
# Each event type → list of description strings
EVENT_DESCRIPTION_VARIANTS = {
    etype: info["descriptions"] for etype, info in EVENTS.items()
}

# V6: Keyword boost sets (lowercased for matching)
EVENT_KEYWORD_BOOST = {
    etype: {kw.lower() for kw in info.get("keyword_boost", [])}
    for etype, info in EVENTS.items()
}


class EmbeddingEventClassifier:
    """
    Semantic event classifier using embedding cosine similarity.

    V6: Uses averaged multi-variant embeddings per event type, configurable
    threshold from env, and disambiguation logic for close scores.

    Instead of regex patterns, this embeds event type descriptions and
    compares them against article titles. The embedding model understands
    meaning — "RBI tightens norms" matches "regulation" without any
    hardcoded pattern for "RBI" or "norms."
    """

    def __init__(self, embedding_tool):
        self.embedding_tool = embedding_tool
        self._event_embeddings_norm = None
        self._event_types = None
        # V6: Load configurable thresholds from env
        self._threshold: Optional[float] = None
        self._ambiguity_margin: Optional[float] = None

    def _get_config(self) -> Tuple[float, float]:
        """Load classifier config from env (cached after first call)."""
        if self._threshold is not None:
            return self._threshold, self._ambiguity_margin
        try:
            from app.config import get_settings
            s = get_settings()
            self._threshold = s.event_classifier_threshold
            self._ambiguity_margin = s.event_classifier_ambiguity_margin
        except Exception:
            self._threshold = 0.35
            self._ambiguity_margin = 0.05
        return self._threshold, self._ambiguity_margin

    def _ensure_event_embeddings(self):
        """
        Compute and cache event type embeddings (once).

        V6: Embeds ALL description variants per event type, then averages
        them into a single representative embedding. This gives broader
        semantic coverage than a single long description.
        """
        if self._event_embeddings_norm is not None:
            return

        self._event_types = list(EVENT_DESCRIPTION_VARIANTS.keys())

        # Embed all variants for all event types
        all_descs: List[str] = []
        variant_counts: List[int] = []  # How many variants per event type
        for etype in self._event_types:
            variants = EVENT_DESCRIPTION_VARIANTS[etype]
            all_descs.extend(variants)
            variant_counts.append(len(variants))

        raw_all = np.array(self.embedding_tool.embed_batch(all_descs))

        # Average variants per event type
        averaged = []
        idx = 0
        for count in variant_counts:
            chunk = raw_all[idx:idx + count]
            avg = np.mean(chunk, axis=0)
            averaged.append(avg)
            idx += count

        event_embs = np.array(averaged)
        norms = np.linalg.norm(event_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self._event_embeddings_norm = event_embs / norms

        total_variants = sum(variant_counts)
        logger.debug(
            f"Event embeddings computed: {len(self._event_types)} types, "
            f"{total_variants} total variants, dim={event_embs.shape[1]}"
        )

    def _keyword_boost_score(self, text_lower: str, event_type: str) -> float:
        """
        V6: Check if article text contains keywords for an event type.

        Returns a small boost (0.0 to 0.08) based on keyword matches.
        This helps disambiguate when embedding scores are close.
        """
        keywords = EVENT_KEYWORD_BOOST.get(event_type, set())
        if not keywords:
            return 0.0
        matches = sum(1 for kw in keywords if kw in text_lower)
        # Cap at 0.08 boost (2+ keyword matches = max boost)
        return min(matches * 0.04, 0.08)

    def classify_batch(
        self, articles: list, threshold: float = None,
    ) -> Dict[str, int]:
        """
        Classify all articles by embedding similarity. Returns event distribution.

        V6 improvements over original:
        - Uses averaged multi-variant embeddings for better coverage
        - Configurable threshold from env (default 0.35)
        - Disambiguation: when top-2 scores within margin, keyword boost breaks tie
        - Structured logging with confidence stats

        Args:
            articles: List of NewsArticle objects.
            threshold: Minimum cosine similarity to assign a specific event type.
                       If None, uses env EVENT_CLASSIFIER_THRESHOLD (default 0.35).
        """
        self._ensure_event_embeddings()
        env_threshold, ambiguity_margin = self._get_config()
        if threshold is None:
            threshold = env_threshold

        # Embed title + summary for richer context (titles alone are too short)
        texts = [
            f"{a.title}. {a.summary or ''}"[:500] for a in articles
        ]
        title_embs = np.array(self.embedding_tool.embed_batch(texts))
        norms = np.linalg.norm(title_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        title_embs_norm = title_embs / norms

        # Cosine similarity: (n_articles, n_event_types)
        sims = np.dot(title_embs_norm, self._event_embeddings_norm.T)

        distribution: Dict[str, int] = {}
        confidences: List[float] = []
        disambiguated_count = 0

        for i, article in enumerate(articles):
            # V6: Get top-2 scores for disambiguation
            sorted_indices = np.argsort(sims[i])[::-1]
            best_idx = int(sorted_indices[0])
            best_score = float(sims[i][best_idx])
            second_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else -1
            second_score = float(sims[i][second_idx]) if second_idx >= 0 else 0.0

            text_lower = f"{article.title}. {getattr(article, 'summary', '') or ''}".lower()

            if best_score >= threshold:
                etype = self._event_types[best_idx]

                # V6: Disambiguation — if top-2 are within margin, use keyword boost
                if second_idx >= 0 and (best_score - second_score) < ambiguity_margin:
                    second_etype = self._event_types[second_idx]
                    boost_best = self._keyword_boost_score(text_lower, etype)
                    boost_second = self._keyword_boost_score(text_lower, second_etype)

                    if boost_second > boost_best:
                        logger.debug(
                            f"Disambiguated '{article.title[:50]}': "
                            f"{etype}({best_score:.3f}+{boost_best:.3f}) → "
                            f"{second_etype}({second_score:.3f}+{boost_second:.3f})"
                        )
                        etype = second_etype
                        best_score = second_score + boost_second
                        disambiguated_count += 1
                    else:
                        best_score += boost_best

                urgency = EVENT_URGENCY[etype]
            else:
                etype = "general"
                urgency = 0.30

            confidences.append(best_score)

            # Store classification confidence for downstream use
            article._trigger_event = etype
            article._trigger_urgency = urgency
            article._trigger_confidence = best_score
            article._trigger_intent = (
                f"{etype} event detected (confidence: {best_score:.2f})"
                if etype != "general"
                else "General business news"
            )

            if hasattr(article, 'trend_types') and isinstance(article.trend_types, list):
                if etype not in [str(t) for t in article.trend_types]:
                    article.trend_types.append(etype)

            distribution[etype] = distribution.get(etype, 0) + 1

        # V6: Structured logging with stats
        avg_conf = np.mean(confidences) if confidences else 0.0
        general_pct = distribution.get("general", 0) / max(len(articles), 1) * 100
        logger.info(
            f"Event classification: {distribution} | "
            f"threshold={threshold:.2f}, avg_confidence={avg_conf:.3f}, "
            f"general={general_pct:.0f}%, disambiguated={disambiguated_count}"
        )
        return distribution
