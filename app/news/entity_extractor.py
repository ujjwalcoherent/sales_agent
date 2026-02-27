"""
Named Entity Recognition (NER) for news articles using spaCy.

Extracts people, organizations, locations, dates, events, and monetary
amounts from article text. These entities feed into:
- Trend signal computation (key_person_flag, company_count, entity_density)
- Cluster quality assessment (entity overlap between articles)
- Sales intelligence (which companies/people are involved in each trend)

MODEL CHOICE:
  en_core_web_sm (15MB, CPU): Fast, adequate for well-written news. Default.
  en_core_web_trf (440MB, GPU optional): 3-5% better accuracy, 10x slower.
  Use trf when entity accuracy is critical (e.g., financial compliance).

PERFORMANCE:
  500 articles (sm):  ~1.5 sec (using pipe() batch processing)
  500 articles (trf): ~15 sec
  5000 articles (sm): ~12 sec

REQUIRES: pip install spacy && python -m spacy download en_core_web_sm
NOTE: spaCy requires Python <=3.13 as of spaCy 3.8. Python 3.14 support
      is expected in spaCy 3.9+.

REF: spaCy v3 NER benchmarks: 90-96% F1 on OntoNotes.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

# Attempt to import spaCy — will fail on unsupported Python versions
try:
    import spacy
    _SPACY_AVAILABLE = True
except Exception as e:
    _SPACY_AVAILABLE = False
    logger.warning(f"spaCy not available ({e}). NER extraction will be skipped.")


# Entity types we care about for news trend analysis
RELEVANT_ENTITY_TYPES = {
    "PERSON",    # People (Narendra Modi, Elon Musk)
    "ORG",       # Organizations (RBI, Tata, Google)
    "GPE",       # Geopolitical entities (India, Mumbai, USA)
    "DATE",      # Dates (January 15, 2025)
    "EVENT",     # Named events (Union Budget, Olympics)
    "MONEY",     # Monetary values (Rs 1.26 lakh crore, $200M)
    "PRODUCT",   # Products (iPhone, Boeing 737)
    "LAW",       # Laws and regulations (GDPR, RBI circular)
    "NORP",      # Nationalities/groups (Indian, Republican)
    "FAC",       # Facilities (Mumbai Stock Exchange)
    "LOC",       # Non-GPE locations (Asia Pacific, Western Ghats)
    "QUANTITY",  # Quantities (500 employees, 200 stores, 10GW capacity)
    "PERCENT",   # Percentages (30% tariff reduction, 15% margin compression)
    "CARDINAL",  # Numeric values (42 companies, 3 million users)
}

# Identity entity types — named things useful for overlap, dedup, cluster coherence.
# These go into article.entity_names (the "quick overlap" list).
# Numeric types (DATE, MONEY, QUANTITY, PERCENT, CARDINAL) are excluded because
# "January 2026" never overlaps meaningfully with "February 2026" and "$500M"
# never overlaps with "$200M". They still appear in article.entities for signals.
IDENTITY_ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "EVENT", "PRODUCT", "LAW", "NORP", "FAC", "LOC",
}


class EntityExtractor:
    """
    Batch entity extraction from news articles using spaCy.

    Uses spaCy's pipe() for efficient batch processing (much faster than
    processing articles one-by-one).

    EXTENSIBILITY: To swap to a different NER system:
    1. Create a new class with the same extract_batch() interface
    2. Register with StrategyRegistry: register("entity", "your_ner", YourClass)
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Args:
            model_name: spaCy model to use. Options:
                - "en_core_web_sm" (15MB, fast, default)
                - "en_core_web_md" (44MB, includes word vectors)
                - "en_core_web_lg" (560MB, best non-transformer)
                - "en_core_web_trf" (440MB, transformer-based, highest accuracy)
        """
        self.model_name = model_name
        self._nlp = None

    @property
    def nlp(self):
        """Lazy-load spaCy model (only when first used)."""
        if self._nlp is None:
            if not _SPACY_AVAILABLE:
                raise ImportError(
                    "spaCy is not available in this Python environment. "
                    "spaCy requires Python <=3.13. Upgrade spaCy or use Python 3.13."
                )
            try:
                self._nlp = spacy.load(self.model_name)
                logger.info(f"Loaded spaCy model: {self.model_name}")
            except OSError:
                raise OSError(
                    f"spaCy model '{self.model_name}' not found. Run:\n"
                    f"  python -m spacy download {self.model_name}"
                )
        return self._nlp

    def extract_batch(self, articles: list) -> list:
        """
        Extract entities from a batch of articles. Enriches articles in-place.

        Uses spaCy's pipe() for batch processing which is significantly faster
        than calling nlp() on each article individually.

        If spaCy is not available (e.g., unsupported Python version), returns
        articles unchanged — the pipeline continues without NER data.
        Entity-based signals will return zero values.

        Args:
            articles: List of NewsArticle objects

        Returns:
            Same list with entities, entity_names, and legacy fields populated.
        """
        if not articles:
            return articles

        if not _SPACY_AVAILABLE:
            logger.warning(
                "spaCy not available — skipping NER. Entity signals will be zero. "
                "Install spaCy with a compatible Python version for full functionality."
            )
            return articles

        # Prepare texts for batch processing
        # Use title + summary + first 1200 chars of content to match synthesis window.
        # NER must see the SAME text window as the LLM synthesizer, otherwise the
        # validator flags entities from content[500:1200] as "hallucinated" when the
        # LLM correctly extracted them from article text.
        # Performance: ~2s for 500 articles (sm model), acceptable overhead.
        texts = []
        for article in articles:
            content_lead = (getattr(article, "content", None) or "")[:1200]
            parts = [article.title, article.summary]
            if content_lead:
                parts.append(content_lead)
            text = ". ".join(p for p in parts if p)
            texts.append(text)

        # Process all texts in batch using pipe()
        # n_process=1 for safety (multiprocessing can cause issues on Windows)
        # batch_size=50 is optimal for the sm model
        docs = list(self.nlp.pipe(texts, batch_size=50, n_process=1))

        from app.schemas.news import Entity as EntityModel
        from app.news.entity_normalizer import normalize_entities_batch

        for article, doc in zip(articles, docs):
            entities = []
            entity_names = []
            companies = []
            people = []
            locations = []

            for ent in doc.ents:
                if ent.label_ not in RELEVANT_ENTITY_TYPES:
                    continue

                # Compute salience: position-based heuristic
                # Entities mentioned earlier in the text are more salient
                # (inverted pyramid — most important info comes first)
                position_ratio = ent.start_char / max(len(doc.text), 1)
                salience = max(0.1, 1.0 - position_ratio)

                entity = EntityModel(
                    text=ent.text,
                    type=ent.label_,
                    salience=round(salience, 2),
                )
                entities.append(entity)

                # Only identity entities go into entity_names (for overlap/coherence).
                # Numeric types (DATE, MONEY, QUANTITY, PERCENT, CARDINAL) are kept
                # in article.entities but excluded from the overlap list.
                if ent.label_ in IDENTITY_ENTITY_TYPES:
                    entity_names.append(ent.text)

                # Populate legacy fields for backward compat with downstream pipeline
                if ent.label_ == "ORG":
                    companies.append(ent.text)
                elif ent.label_ == "PERSON":
                    people.append(ent.text)
                elif ent.label_ in ("GPE", "LOC", "FAC"):
                    locations.append(ent.text)

            # Normalize + deduplicate entity names (identity entities only)
            unique_names = normalize_entities_batch(entity_names, deduplicate=True)

            article.entities = entities
            article.entity_names = unique_names
            # Normalize legacy fields too — prevents "Tata Motors Ltd" and
            # "Tata Motors" from counting as 2 companies in signals.
            article.mentioned_companies = normalize_entities_batch(
                article.mentioned_companies + companies, deduplicate=True
            )
            article.mentioned_people = normalize_entities_batch(
                article.mentioned_people + people, deduplicate=True
            )
            article.mentioned_locations = normalize_entities_batch(
                article.mentioned_locations + locations, deduplicate=True
            )

        logger.info(
            f"Entity extraction: {len(articles)} articles processed, "
            f"avg {sum(len(a.entities) for a in articles) / max(len(articles), 1):.1f} "
            f"entities per article"
        )
        return articles

