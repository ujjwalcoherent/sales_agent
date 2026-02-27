"""
Layer 2: Trend detection engine.

Pipeline (TrendPipeline, formerly RecursiveTrendEngine):
  Layer 1 (Ingest):  articles → scrape → dedup → NER → embed → filter
  Layer 2 (Cluster): Leiden → coherence → keywords → signals
  Layer 3 (Relate):  causal graph (entity bridges, sector chains)
  Layer 4 (Temporalize): trend memory (novelty vs. continuity)
  Layer 5 (Enrich):  LLM synthesis → quality gate → TrendTree

Modules:
  - engine.py: TrendPipeline (the core layered pipeline)
  - clustering.py: k-NN graph + Leiden community detection
  - keywords.py: c-TF-IDF topic labeling
  - signals/: Signal computation (temporal, source, content, entity, market, composite)
  - coherence.py: Post-clustering validation in original embedding space
"""

from app.trends.engine import TrendPipeline, RecursiveTrendEngine, detect_trend_tree
from app.tools.embeddings import EmbeddingTool, embed, embed_batch, cosine_similarity
