"""
Entity co-occurrence tracking — Recorded Future-inspired knowledge graph.

WHY CO-OCCURRENCE:
  Entities that frequently appear together in news articles often have hidden
  business relationships. Tracking these reveals:
  - Company-to-regulation links (who's affected by new laws)
  - Company-to-company relationships (supply chain, competition, partnership)
  - Person-to-company movements (leadership changes across industry)
  - Entity-to-event patterns (which companies face which trigger events)

  This is the core technique behind Recorded Future's intelligence platform
  and is used by Bloomberg Terminal for relationship mapping.

APPROACH:
  Lightweight adjacency-based co-occurrence (no networkx dependency needed).
  For each article, all entity pairs that appear together get a +1 weight.
  Result: a weighted graph where edge weight = co-occurrence frequency.

  O(N * E^2) where N = articles, E = avg entities per article (~5-10).
  For 500 articles with ~8 entities each: ~500 * 28 = 14,000 pair updates.
  Takes <0.1 sec. Runs after NER extraction.

REF: Recorded Future entity graph methodology
     Bloomberg relationship mapping
     Google Knowledge Graph entity linking
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


class EntityCooccurrenceTracker:
    """
    Tracks which entities appear together across articles.

    Builds a lightweight co-occurrence graph without external dependencies.
    Used to identify hidden connections between trends and companies.
    """

    def __init__(self):
        # edge: (entity_a, entity_b) → count
        # Always store with sorted names so (A,B) == (B,A)
        self.edges: Dict[Tuple[str, str], int] = defaultdict(int)

        # entity → set of article IDs where it appears
        self.entity_articles: Dict[str, Set[str]] = defaultdict(set)

        # entity → set of trigger events it's associated with
        self.entity_events: Dict[str, Set[str]] = defaultdict(set)

    def process_articles(self, articles: list) -> Dict[str, Any]:
        """
        Build co-occurrence graph from articles with extracted entities.

        Args:
            articles: List of NewsArticle objects (must have entity_names attribute)

        Returns:
            Dict with co-occurrence stats and top relationships.
        """
        for article in articles:
            entity_names = getattr(article, 'entity_names', [])
            if not entity_names:
                continue

            article_id = str(getattr(article, 'id', ''))
            trigger_event = getattr(article, '_trigger_event', 'general')

            # Normalize entity names
            normalized = list(set(e.strip().title() for e in entity_names if len(e.strip()) > 1))

            # Track entity → articles
            for entity in normalized:
                self.entity_articles[entity].add(article_id)
                self.entity_events[entity].add(trigger_event)

            # Track co-occurrence pairs
            for i in range(len(normalized)):
                for j in range(i + 1, len(normalized)):
                    pair = tuple(sorted([normalized[i], normalized[j]]))
                    self.edges[pair] += 1

        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the co-occurrence graph."""
        if not self.edges:
            return {
                "total_entities": 0,
                "total_edges": 0,
                "top_connections": [],
                "most_connected_entities": [],
                "cross_event_entities": [],
            }

        # Top co-occurring pairs (strongest connections)
        top_connections = sorted(
            self.edges.items(), key=lambda x: x[1], reverse=True
        )[:20]

        # Most connected entities (highest degree)
        degree: Dict[str, int] = defaultdict(int)
        for (a, b), weight in self.edges.items():
            degree[a] += weight
            degree[b] += weight
        most_connected = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:15]

        # Entities that appear across multiple trigger event types
        # These are "cross-cutting" entities that connect different trends
        cross_event = [
            (entity, len(events), events)
            for entity, events in self.entity_events.items()
            if len(events) >= 2
        ]
        cross_event.sort(key=lambda x: x[1], reverse=True)

        return {
            "total_entities": len(self.entity_articles),
            "total_edges": len(self.edges),
            "top_connections": [
                {"entity_a": a, "entity_b": b, "strength": w}
                for (a, b), w in top_connections
            ],
            "most_connected_entities": [
                {"entity": e, "connections": c, "article_count": len(self.entity_articles.get(e, set()))}
                for e, c in most_connected
            ],
            "cross_event_entities": [
                {"entity": e, "event_count": c, "events": list(evts)}
                for e, c, evts in cross_event[:10]
            ],
        }

    def get_entity_neighbors(self, entity: str) -> List[Dict[str, Any]]:
        """Get all entities connected to a given entity, with strength."""
        entity_norm = entity.strip().title()
        neighbors = []
        for (a, b), weight in self.edges.items():
            if a == entity_norm:
                neighbors.append({"entity": b, "strength": weight})
            elif b == entity_norm:
                neighbors.append({"entity": a, "strength": weight})
        neighbors.sort(key=lambda x: x["strength"], reverse=True)
        return neighbors

    def get_company_event_matrix(self) -> List[Dict[str, Any]]:
        """
        Get which companies are associated with which trigger events.

        This is the key sales intelligence output: "Company X is facing
        regulation + funding events → compliance + growth consulting need."
        """
        results = []
        for entity, events in self.entity_events.items():
            if len(events) > 0:
                article_count = len(self.entity_articles.get(entity, set()))
                if article_count >= 2:  # Only entities mentioned in 2+ articles
                    results.append({
                        "entity": entity,
                        "events": list(events),
                        "article_count": article_count,
                        "multi_event": len(events) > 1,
                    })
        results.sort(key=lambda x: x["article_count"], reverse=True)
        return results[:30]
