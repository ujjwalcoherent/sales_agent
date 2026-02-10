"""
Layer 2: Trend detection engine.

Pipeline (RecursiveTrendEngine):
  articles → dedup → NER → embed → UMAP → HDBSCAN → c-TF-IDF → signals → LLM → TrendTree

Modules:
  - engine.py: RecursiveTrendEngine (the core pipeline)
  - reduction.py: UMAP dimensionality reduction
  - keywords.py: c-TF-IDF topic labeling
  - signals/: Signal computation (temporal, source, content, entity, market, composite)
"""

from app.trends.engine import RecursiveTrendEngine, detect_trend_tree
from app.tools.embeddings import EmbeddingTool, embed, embed_batch, cosine_similarity
from app.tools.trend_synthesizer import TrendSynthesizer, synthesize_trends
