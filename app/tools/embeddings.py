"""
Embeddings tool for generating vector representations of text.

Supports (in priority order):
1. Local Sentence Transformers (fastest, no API calls)
2. Ollama nomic-embed-text (local fallback)
3. Hugging Face Inference API (cloud fallback)

Models:
- Local: paraphrase-multilingual-MiniLM-L12-v2 (384-dim, multilingual)
- Ollama: nomic-embed-text (768-dim)
- HuggingFace API: BAAI/bge-small-en-v1.5 (384-dim)

IMPORTANT: All fallback providers MUST produce the same dimension as the primary
provider. If dimension changes mid-pipeline, downstream components (trend memory,
semantic dedup thresholds) break silently. The _locked_dim mechanism prevents this.
"""

import logging
from typing import Any, Dict, List, Optional

import httpx
import numpy as np

from ..config import get_settings

logger = logging.getLogger(__name__)

# Default embedding dimension (updated on first successful embedding)
_DEFAULT_DIM = 384

# Singleton for local model (avoids reloading on every EmbeddingTool instantiation)
_local_model = None
_local_model_name = None


def _get_device() -> str:
    """Detect best available device: CUDA GPU > CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            logger.info(f"GPU detected: {gpu_name} ({vram_mb}MB VRAM) — using CUDA")
            return "cuda"
    except Exception:
        pass
    logger.info("No CUDA GPU detected — using CPU")
    return "cpu"


def _get_local_model(model_name: str):
    """Load local sentence-transformers model (singleton, lazy-loaded)."""
    global _local_model, _local_model_name
    if _local_model is not None and _local_model_name == model_name:
        return _local_model
    try:
        import os
        # Set HF_TOKEN so sentence-transformers/huggingface_hub authenticates downloads
        settings = get_settings()
        if settings.huggingface_api_key and not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = settings.huggingface_api_key

        from sentence_transformers import SentenceTransformer
        device = _get_device()
        logger.info(f"Loading local embedding model: {model_name} on {device}...")
        _local_model = SentenceTransformer(model_name, device=device)
        _local_model_name = model_name
        dim = _local_model.get_sentence_embedding_dimension()
        logger.info(f"Local embedding model loaded: {model_name} (dim={dim}, device={device})")
        return _local_model
    except Exception as e:
        logger.error(
            f"CRITICAL: Failed to load local embedding model '{model_name}': {type(e).__name__}: {e}. "
            f"Fallback providers may produce different dimensions, causing pipeline errors. "
            f"Install: pip install sentence-transformers"
        )
        return None


class EmbeddingTool:
    """
    Generate embeddings with local-first strategy.

    Priority:
    1. Local Sentence Transformers (no API, fastest)
    2. Ollama nomic-embed-text (local, no API key needed)
    3. Hugging Face API (cloud fallback if HF_API_KEY set)
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._hf_client = None
        self._ollama_available: bool | None = None
        self._local_model_checked = False
        self._local_available = False
        self._embedding_dim = _DEFAULT_DIM
        # Once the first embedding is produced, lock the dimension.
        # Reject fallback providers that produce a different dimension.
        self._dim_locked = False
        self._active_provider: Optional[str] = None

    @property
    def local_model(self):
        """Lazy-load local sentence-transformers model."""
        if not self._local_model_checked:
            self._local_model_checked = True
            model_name = getattr(self.settings, 'local_embedding_model', 'paraphrase-multilingual-MiniLM-L12-v2')
            model = _get_local_model(model_name)
            self._local_available = model is not None
            if model is not None:
                self._embedding_dim = model.get_sentence_embedding_dimension()
                self._dim_locked = True
                self._active_provider = f"local:{model_name}"
                logger.info(f"Embedding provider: local ({model_name}, dim={self._embedding_dim})")
        return _local_model if self._local_available else None

    @property
    def hf_client(self):
        """Lazy-load Hugging Face client."""
        if self._hf_client is not None:
            return self._hf_client

        if not self.settings.huggingface_api_key:
            logger.debug("HF_API_KEY not set, Hugging Face embeddings unavailable")
            return None

        try:
            from huggingface_hub import InferenceClient
            self._hf_client = InferenceClient(token=self.settings.huggingface_api_key)
            logger.info("Hugging Face embeddings initialized")
            return self._hf_client
        except ImportError:
            logger.warning("huggingface-hub not installed")
            return None

    @property
    def ollama_available(self) -> bool:
        """Check if Ollama is available for embeddings (cached after first check)."""
        if self._ollama_available is not None:
            return self._ollama_available

        if not self.settings.use_ollama:
            self._ollama_available = False
            return False

        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.settings.ollama_base_url}/api/tags")
                self._ollama_available = response.status_code == 200
                if self._ollama_available:
                    logger.info("Ollama available for embeddings")
        except Exception:
            self._ollama_available = False
            logger.debug("Ollama not available for embeddings")

        return self._ollama_available

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _zero_vector(self) -> List[float]:
        return [0.0] * self._embedding_dim

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text. Returns a zero vector on failure."""
        if not text or not text.strip():
            return self._zero_vector()

        embedding = self._try_embed(text)
        if embedding is not None:
            return embedding

        logger.error("No embedding backend available (set HF_API_KEY or enable Ollama)")
        return self._zero_vector()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using batched API calls.

        Empty/blank texts receive zero vectors. Sends texts in batches
        to the HF API (or Ollama) instead of one-by-one.
        """
        if not texts:
            return []

        # Build index map: track which texts are non-empty
        results: List[List[float] | None] = [None] * len(texts)
        non_empty_indices = []
        non_empty_texts = []
        for i, t in enumerate(texts):
            if t and t.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(t)
            else:
                results[i] = self._zero_vector()

        if not non_empty_texts:
            return [self._zero_vector() for _ in texts]

        # Try batch embed via HF first, then Ollama fallback
        batch_embeddings = self._try_embed_batch(non_empty_texts, batch_size)

        if batch_embeddings and len(batch_embeddings) == len(non_empty_texts):
            for idx, emb in zip(non_empty_indices, batch_embeddings):
                results[idx] = emb
        else:
            # Fallback: sequential (shouldn't normally reach here)
            logger.warning("Batch embed failed, falling back to sequential")
            for idx, text in zip(non_empty_indices, non_empty_texts):
                results[idx] = self.embed_text(text)

        # Fill any remaining None with zero vectors
        for i in range(len(results)):
            if results[i] is None:
                results[i] = self._zero_vector()

        # DEBUG: Log embedding stats
        valid_embeddings = [e for e in results if e and any(v != 0 for v in e)]
        if valid_embeddings:
            emb_array = np.array(valid_embeddings)
            logger.debug(f"Embedding batch stats: shape={emb_array.shape}, "
                        f"min={emb_array.min():.4f}, max={emb_array.max():.4f}, "
                        f"mean={emb_array.mean():.4f}, std={emb_array.std():.4f}")
            # Check for near-identical embeddings (would indicate a bug)
            if len(valid_embeddings) > 1:
                from sklearn.metrics.pairwise import cosine_similarity as cosine_sim
                sample_size = min(10, len(valid_embeddings))
                sample_sims = cosine_sim(emb_array[:sample_size])
                upper_tri = sample_sims[np.triu_indices(sample_size, k=1)]
                if len(upper_tri) > 0:
                    avg_sim = upper_tri.mean()
                    logger.debug(f"Sample pairwise cosine similarities: avg={avg_sim:.4f}, "
                                f"min={upper_tri.min():.4f}, max={upper_tri.max():.4f}")
                    if avg_sim > 0.99:
                        logger.warning(f"!!! SUSPICIOUSLY HIGH EMBEDDING SIMILARITY ({avg_sim:.4f}) - embeddings may be identical !!!")

        logger.info(f"Generated {len(non_empty_texts)} embeddings (batch), dim={self._embedding_dim}")
        return results

    def _try_embed_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[float]] | None:
        """Try batch embedding: Local → Ollama → HF API."""
        # 1. Local sentence-transformers (fastest, no API)
        if self.local_model is not None:
            try:
                return self._embed_local_batch(texts, batch_size)
            except Exception as e:
                logger.warning(f"Local batch embedding failed: {e}, trying Ollama")

        # 2. Ollama nomic-embed-text (local, no API key)
        if self.ollama_available:
            try:
                return self._embed_ollama_batch(texts)
            except Exception as e:
                logger.warning(f"Ollama batch embedding failed: {e}, trying HF API")

        # 3. HuggingFace API (cloud fallback)
        if self.hf_client:
            try:
                return self._embed_hf_batch(texts, batch_size)
            except Exception as e:
                logger.error(f"HF batch embedding failed: {e}")

        return None

    def _embed_local_batch(
        self, texts: List[str], batch_size: int = 64
    ) -> List[List[float]]:
        """Embed using local sentence-transformers model. Uses GPU (CUDA) if available."""
        import time
        start = time.time()
        model = self.local_model
        # sentence-transformers handles batching internally, but we can set batch_size
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        result = embeddings.tolist()
        elapsed = time.time() - start
        rate = len(texts) / elapsed if elapsed > 0 else 0
        logger.info(f"Local embeddings: {len(texts)} texts in {elapsed:.2f}s ({rate:.0f} texts/sec)")
        if result:
            self._update_dim(result[0])
        return result

    def _embed_hf_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[float]]:
        """Send texts to HF API in batches (one API call per batch)."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = self.hf_client.feature_extraction(
                batch,
                model=self.settings.embedding_model,
            )
            # Parse response: list of embeddings
            batch_embs = []
            if isinstance(result, np.ndarray):
                batch_embs = result.tolist()
            elif isinstance(result, list):
                for item in result:
                    if isinstance(item, list) and item and isinstance(item[0], list):
                        batch_embs.append(item[0])
                    elif isinstance(item, list):
                        batch_embs.append(item)
                    else:
                        batch_embs.append(list(item))
            else:
                batch_embs = [list(r) for r in result]

            if batch_embs:
                self._update_dim(batch_embs[0])
            all_embeddings.extend(batch_embs)
            logger.info(f"HF batch {i // batch_size + 1}: {len(batch)} texts embedded")

        return all_embeddings

    def _embed_ollama_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed via Ollama using a persistent client connection."""
        url = f"{self.settings.ollama_base_url}/api/embeddings"
        embeddings = []
        with httpx.Client(timeout=120.0) as client:
            for text in texts:
                response = client.post(
                    url, json={"model": "nomic-embed-text", "prompt": text}
                )
                response.raise_for_status()
                emb = response.json().get("embedding", [])
                if not emb and self._embedding_dim:
                    emb = self._zero_vector()  # Replace empty with zero vector
                embeddings.append(emb)
        if embeddings:
            # Check dimension compatibility before committing
            ollama_dim = len(embeddings[0])
            if self._dim_locked and ollama_dim != self._embedding_dim:
                raise ValueError(
                    f"Ollama nomic-embed-text produces {ollama_dim}-dim embeddings "
                    f"but pipeline expects {self._embedding_dim}-dim. Cannot mix dimensions."
                )
            if not self._active_provider:
                self._active_provider = "ollama:nomic-embed-text"
            self._update_dim(embeddings[0])
        return embeddings

    def _try_embed(self, text: str) -> List[float] | None:
        """Try each backend in priority order: Local → Ollama → HF API."""
        # 1. Local sentence-transformers
        if self.local_model is not None:
            try:
                return self._embed_local(text)
            except Exception as e:
                logger.warning(f"Local embedding failed: {e}, trying Ollama")

        # 2. Ollama
        if self.ollama_available:
            try:
                return self._embed_ollama(text)
            except Exception as e:
                logger.warning(f"Ollama embedding failed: {e}, trying HF API")

        # 3. HuggingFace API
        if self.hf_client:
            try:
                return self._embed_hf(text)
            except Exception as e:
                logger.error(f"HuggingFace embedding failed: {e}")

        return None

    def _embed_local(self, text: str) -> List[float]:
        """Embed a single text using local sentence-transformers."""
        model = self.local_model
        embedding = model.encode(text, normalize_embeddings=True)
        result = embedding.tolist()
        self._update_dim(result)
        return result

    def _embed_hf(self, text: str) -> List[float]:
        """Generate embedding using Hugging Face API."""
        result = self.hf_client.feature_extraction(
            text,
            model=self.settings.embedding_model,
        )

        if isinstance(result, list):
            # Some models return nested [[...]] instead of flat [...]
            if result and isinstance(result[0], list):
                embedding = result[0]
            else:
                embedding = result
        else:
            embedding = list(result)

        self._update_dim(embedding)
        # Reject mismatched dimensions — return zero vector instead of corrupt data
        if self._dim_locked and len(embedding) != self._embedding_dim:
            logger.error(f"HF returned {len(embedding)}-dim, expected {self._embedding_dim}. Returning zero vector.")
            return self._zero_vector()
        return embedding

    def _embed_ollama(self, text: str) -> List[float]:
        """Generate embedding using Ollama API."""
        url = f"{self.settings.ollama_base_url}/api/embeddings"
        payload = {"model": "nomic-embed-text", "prompt": text}

        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            embedding = response.json().get("embedding", [])

        # Check dimension compatibility
        if embedding and self._dim_locked and len(embedding) != self._embedding_dim:
            raise ValueError(
                f"Ollama produces {len(embedding)}-dim but pipeline expects {self._embedding_dim}-dim"
            )
        self._update_dim(embedding)
        return embedding

    def _update_dim(self, embedding: List[float]) -> None:
        """Update embedding dimension from a successful result.

        If dimension was already locked by the primary provider, reject
        embeddings with mismatched dimensions to prevent silent corruption.
        """
        if not embedding:
            return
        new_dim = len(embedding)
        if self._dim_locked and new_dim != self._embedding_dim:
            logger.error(
                f"DIMENSION MISMATCH: fallback provider returned {new_dim}-dim "
                f"but pipeline locked to {self._embedding_dim}-dim. "
                f"This embedding will cause downstream errors."
            )
            # Don't update - keep the locked dimension so zero vectors are correct size
            return
        self._embedding_dim = new_dim
        if not self._dim_locked:
            self._dim_locked = True

    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """Compute cosine similarity between two embeddings (0-1)."""
        if not embedding1 or not embedding2:
            return 0.0

        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def compute_similarity_matrix(
        self,
        embeddings: List[List[float]],
    ) -> np.ndarray:
        """Compute pairwise cosine similarity matrix for all embeddings."""
        if not embeddings:
            return np.array([])

        matrix = np.array(embeddings)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = matrix / norms
        return np.dot(normalized, normalized.T)

    def find_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Find the top-k most similar embeddings above the given threshold."""
        if not query_embedding or not candidate_embeddings:
            return []

        query = np.array(query_embedding)
        candidates = np.array(candidate_embeddings)

        query_norm = query / (np.linalg.norm(query) + 1e-10)
        cand_norms = np.linalg.norm(candidates, axis=1, keepdims=True)
        cand_norms[cand_norms == 0] = 1
        candidates_norm = candidates / cand_norms

        similarities = np.dot(candidates_norm, query_norm)
        ranked_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in ranked_indices[:top_k]:
            sim = float(similarities[idx])
            if sim >= threshold:
                results.append({"index": int(idx), "similarity": sim})
        return results


# ── Convenience functions ──────────────────────────────────────────────────────

def embed(text: str) -> List[float]:
    """Quick single-text embedding."""
    return EmbeddingTool().embed_text(text)


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Quick batch embedding."""
    return EmbeddingTool().embed_batch(texts)


def cosine_similarity(emb1: List[float], emb2: List[float]) -> float:
    """Quick cosine similarity."""
    return EmbeddingTool().compute_similarity(emb1, emb2)
