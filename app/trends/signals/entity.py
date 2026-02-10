"""
Entity-based signal computation for news trend analysis.

Measures the importance and specificity of named entities in a trend.
No hardcoded VIP lists or regulatory entity lists — uses NER output
and the embedding-based event classifier instead.

SIGNALS:
  entity_momentum:       Is an entity dominating this cluster? (concentration)
  key_person_flag:       Does a PERSON entity appear in article titles? (prominence)
  company_count:         How many unique companies are mentioned?
  entity_density:        Entities per article (higher = more specific/actionable).
  regulatory_entity_flag: Uses event classifier output (no hardcoded list).
"""

import logging
from collections import Counter
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def compute_entity_signals(articles: list) -> Dict[str, Any]:
    """
    Compute all entity-based signals from NER output.
    No hardcoded keyword lists — uses NER entities and event classifier.
    """
    if not articles:
        return _empty_signals()

    return {
        "entity_momentum": _compute_entity_momentum(articles),
        "key_person_flag": _check_key_person(articles),
        "key_persons_found": _find_key_persons(articles),
        "company_count": _count_companies(articles),
        "entity_density": _compute_entity_density(articles),
        "regulatory_entity_flag": _check_regulatory_entities(articles),
        "top_entities": _get_top_entities(articles, top_n=10),
    }


def _compute_entity_momentum(articles: list) -> float:
    """
    How concentrated are entity mentions? Higher = more focused topic.

    Formula: top_entity_count / total_entity_count
    >0.3: Highly focused (one entity dominates)
    <0.1: Diffuse (many entities, generic industry news)
    """
    all_entities = []
    for a in articles:
        names = getattr(a, 'entity_names', [])
        all_entities.extend(n.lower() for n in names)

    if not all_entities:
        return 0.0

    counts = Counter(all_entities)
    top_count = counts.most_common(1)[0][1]
    return top_count / len(all_entities)


def _check_key_person(articles: list) -> bool:
    """
    Does a PERSON entity appear in article titles?

    Title prominence = importance. If a person is named in the headline,
    they're a key figure in the story. No hardcoded VIP list needed —
    NER + title prominence handles this dynamically across any country.
    """
    for a in articles:
        title = (getattr(a, 'title', '') or '').lower()

        # Check NER-extracted people against the title
        people = getattr(a, 'mentioned_people', [])
        for person in people:
            # Person name appears in the title = prominent figure
            if person.lower().strip() in title:
                return True

        # Also check entity_names with PERSON type
        entities = getattr(a, 'entities', [])
        for ent in entities:
            if getattr(ent, 'type', '') == 'PERSON':
                name = getattr(ent, 'text', '').lower().strip()
                if name and name in title:
                    return True

    return False


def _find_key_persons(articles: list) -> List[str]:
    """Return PERSON entities that appear in article titles (prominent figures)."""
    found = set()
    for a in articles:
        title = (getattr(a, 'title', '') or '').lower()

        people = getattr(a, 'mentioned_people', [])
        for person in people:
            if person.lower().strip() in title:
                found.add(person.strip())

        entities = getattr(a, 'entities', [])
        for ent in entities:
            if getattr(ent, 'type', '') == 'PERSON':
                name = getattr(ent, 'text', '').strip()
                if name and name.lower() in title:
                    found.add(name)

    return sorted(found)


def _count_companies(articles: list) -> int:
    """Count unique companies mentioned across all articles (from NER)."""
    companies = set()
    for a in articles:
        mentioned = getattr(a, 'mentioned_companies', [])
        companies.update(c.lower().strip() for c in mentioned if c.strip())
    return len(companies)


def _compute_entity_density(articles: list) -> float:
    """
    Average number of entities per article.
    Higher density = more specific, factual reporting.
    """
    if not articles:
        return 0.0

    total_entities = sum(
        len(getattr(a, 'entity_names', []))
        for a in articles
    )
    return total_entities / len(articles)


def _check_regulatory_entities(articles: list) -> bool:
    """
    Does this cluster involve regulation?
    Uses the embedding-based event classifier output — no hardcoded entity list.
    """
    for a in articles:
        event = getattr(a, '_trigger_event', '')
        if event in ('regulation', 'crisis'):
            return True
    return False


def _get_top_entities(articles: list, top_n: int = 10) -> List[Dict[str, Any]]:
    """Get the most frequently mentioned entities with their counts and types."""
    entity_counts: Counter = Counter()
    entity_types: Dict[str, str] = {}

    for a in articles:
        entities = getattr(a, 'entities', [])
        for ent in entities:
            text = getattr(ent, 'text', '').strip()
            ent_type = getattr(ent, 'type', 'UNKNOWN')
            if text:
                key = text.lower()
                entity_counts[key] += 1
                entity_types[key] = ent_type

    return [
        {"text": text, "count": count, "type": entity_types.get(text, "UNKNOWN")}
        for text, count in entity_counts.most_common(top_n)
    ]


def _empty_signals() -> Dict[str, Any]:
    """Return zero signals when no articles are available."""
    return {
        "entity_momentum": 0.0,
        "key_person_flag": False,
        "key_persons_found": [],
        "company_count": 0,
        "entity_density": 0.0,
        "regulatory_entity_flag": False,
        "top_entities": [],
    }
