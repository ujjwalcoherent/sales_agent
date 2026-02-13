"""
Diagnostic test script for the India Trend Lead Agent pipeline.
Tests each component independently to isolate failures.
"""
import asyncio
import sys
import os
import time
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix Windows console encoding for Hindi/Unicode text
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass  # Python < 3.7 fallback

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def safe_print(*args, **kwargs):
    """Print that handles Unicode on Windows without crashing."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode('ascii', errors='replace').decode('ascii'), **kwargs)


async def test_1_settings():
    """Test 1: Verify settings load correctly."""
    print("\n" + "=" * 60)
    print("TEST 1: Settings & Configuration")
    print("=" * 60)

    from app.config import get_settings
    settings = get_settings()

    print(f"  NVIDIA API Key: {'SET' if settings.nvidia_api_key else 'MISSING'}")
    print(f"  NVIDIA Model: {settings.nvidia_model}")
    print(f"  Ollama enabled: {settings.use_ollama}")
    print(f"  Ollama Model: {settings.ollama_model}")
    print(f"  OpenRouter Key: {'SET' if settings.openrouter_api_key else 'MISSING'}")
    print(f"  Gemini Key: {'SET' if settings.gemini_api_key else 'MISSING'}")
    print(f"  HF Key: {'SET' if settings.huggingface_api_key else 'MISSING'}")
    print(f"  Local Embedding Model: {settings.local_embedding_model}")
    print(f"  HF Embedding Model: {settings.embedding_model}")
    print(f"  Mock Mode: {settings.mock_mode}")
    return True


async def test_2_embedding_model():
    """Test 2: Verify embedding model loads and produces correct output."""
    print("\n" + "=" * 60)
    print("TEST 2: Embedding Model (Multilingual)")
    print("=" * 60)

    from app.tools.embeddings import EmbeddingTool

    tool = EmbeddingTool()

    # Test with English text
    english_texts = [
        "RBI announces new KYC norms for digital lending platforms",
        "Zepto raises $200 million in Series F funding round",
        "Government approves semiconductor fabrication plants",
    ]

    # Test with Hindi text
    hindi_texts = [
        "भारतीय रिज़र्व बैंक ने डिजिटल ऋण के लिए नए KYC मानदंड जारी किए",
        "सरकार ने सेमीकंडक्टर फैब्रिकेशन प्लांट को मंजूरी दी",
    ]

    # Test with mixed
    mixed_texts = english_texts + hindi_texts

    print(f"\n  Testing {len(mixed_texts)} texts (English + Hindi)...")
    start = time.time()
    embeddings = tool.embed_batch(mixed_texts)
    elapsed = time.time() - start

    print(f"  Embedding time: {elapsed:.2f}s")
    print(f"  Embedding dim: {len(embeddings[0]) if embeddings else 'N/A'}")
    print(f"  Model used: {tool._local_model_checked}, local_available: {tool._local_available}")

    # Check embedding quality
    import numpy as np
    emb_array = np.array(embeddings)
    print(f"  Shape: {emb_array.shape}")
    print(f"  Stats: min={emb_array.min():.4f}, max={emb_array.max():.4f}, mean={emb_array.mean():.4f}")

    # Check pairwise similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(emb_array)

    print(f"\n  Pairwise cosine similarities:")
    for i in range(len(mixed_texts)):
        for j in range(i + 1, len(mixed_texts)):
            label_i = "EN" if i < len(english_texts) else "HI"
            label_j = "EN" if j < len(english_texts) else "HI"
            sim = sim_matrix[i][j]
            safe_print(f"    [{label_i}] '{mixed_texts[i][:40]}...' vs [{label_j}] '{mixed_texts[j][:40]}...' = {sim:.4f}")

    # KEY CHECK: Hindi embeddings should NOT be near-identical
    if len(hindi_texts) >= 2:
        hindi_start = len(english_texts)
        hindi_sim = sim_matrix[hindi_start][hindi_start + 1]
        if hindi_sim > 0.95:
            print(f"\n  *** WARNING: Hindi embeddings too similar ({hindi_sim:.4f}) - model may not handle Hindi ***")
        else:
            print(f"\n  GOOD: Hindi embeddings have reasonable differentiation ({hindi_sim:.4f})")

    # Check if English embeddings are distinct
    en_sims = []
    for i in range(len(english_texts)):
        for j in range(i + 1, len(english_texts)):
            en_sims.append(sim_matrix[i][j])
    avg_en_sim = sum(en_sims) / len(en_sims) if en_sims else 0
    print(f"  Avg English pairwise similarity: {avg_en_sim:.4f}")

    if avg_en_sim > 0.90:
        print(f"  *** WARNING: English embeddings too similar ({avg_en_sim:.4f}) ***")

    return True


async def test_3_rss_fetch():
    """Test 3: Fetch articles from RSS sources."""
    print("\n" + "=" * 60)
    print("TEST 3: RSS Feed Fetching")
    print("=" * 60)

    from app.tools.rss_tool import RSSTool

    tool = RSSTool()
    start = time.time()
    articles = await tool.fetch_all_sources(max_per_source=5, hours_ago=72)
    elapsed = time.time() - start

    print(f"  Fetched {len(articles)} articles in {elapsed:.2f}s")

    # Show source distribution
    source_counts = {}
    for a in articles:
        source_counts[a.source_name] = source_counts.get(a.source_name, 0) + 1

    print(f"  Sources: {len(source_counts)}")
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {src}: {cnt}")

    # Check for content availability
    with_content = sum(1 for a in articles if a.content)
    with_summary = sum(1 for a in articles if a.summary)
    print(f"\n  Articles with content: {with_content}/{len(articles)}")
    print(f"  Articles with summary: {with_summary}/{len(articles)}")

    # Check for non-English articles
    import re
    hindi_pattern = re.compile(r'[\u0900-\u097F]')  # Devanagari script
    hindi_articles = [a for a in articles if hindi_pattern.search(a.title)]
    print(f"  Articles with Hindi title: {len(hindi_articles)}/{len(articles)}")
    if hindi_articles:
        for a in hindi_articles[:3]:
            safe_print(f"    - [{a.source_name}] {a.title[:60]}")

    return articles


async def test_4_llm_providers():
    """Test 4: Test each LLM provider individually."""
    print("\n" + "=" * 60)
    print("TEST 4: LLM Provider Chain")
    print("=" * 60)

    from app.tools.llm_service import LLMService

    tool = LLMService()

    # Check provider status
    status = await tool.get_provider_status()
    print(f"  Provider status: {status}")

    test_prompt = 'Respond with exactly this JSON: {"status": "ok", "provider": "test"}'
    system_prompt = "You are a test assistant. Respond with valid JSON only."

    # Test NVIDIA (with 60s timeout to avoid blocking)
    if tool._nvidia_configured:
        print("\n  Testing NVIDIA...")
        try:
            result = await asyncio.wait_for(
                tool._call_nvidia(test_prompt, system_prompt, 0.1, 100),
                timeout=60
            )
            print(f"    NVIDIA: OK ({len(result)} chars)")
        except asyncio.TimeoutError:
            print(f"    NVIDIA: TIMED OUT (60s)")
        except Exception as e:
            print(f"    NVIDIA: FAILED - {e}")

    # Test Ollama
    if tool.settings.use_ollama:
        print("\n  Testing Ollama...")
        try:
            result = await tool._call_ollama(test_prompt, system_prompt, 0.1, 100)
            print(f"    Ollama: OK ({len(result)} chars)")
        except Exception as e:
            print(f"    Ollama: FAILED - {e}")

    # Test OpenRouter
    if tool._openrouter_configured:
        print("\n  Testing OpenRouter...")
        try:
            result = await tool._call_openrouter(test_prompt, system_prompt, 0.1, 100)
            print(f"    OpenRouter: OK ({len(result)} chars)")
        except Exception as e:
            print(f"    OpenRouter: FAILED - {e}")

    # Test Gemini
    if tool._gemini_configured:
        print("\n  Testing Gemini...")
        try:
            result = await tool._call_gemini(test_prompt, system_prompt, 0.1, 100)
            print(f"    Gemini: OK ({len(result)} chars)")
        except Exception as e:
            print(f"    Gemini: FAILED - {e}")

    # Test generate_json (full pipeline)
    print("\n  Testing generate_json (full provider chain)...")
    try:
        result = await tool.generate_json(
            prompt=test_prompt,
            system_prompt=system_prompt,
        )
        print(f"    generate_json: OK - {result}")
        print(f"    Used provider: {tool.last_provider}")
    except Exception as e:
        print(f"    generate_json: FAILED - {e}")

    return True


async def test_5_dedup():
    """Test 5: Test deduplication pipeline."""
    print("\n" + "=" * 60)
    print("TEST 5: Deduplication")
    print("=" * 60)

    from app.news.dedup import ArticleDeduplicator
    from app.schemas import NewsArticle, SourceType, SourceTier
    from datetime import datetime

    dedup = ArticleDeduplicator(threshold=0.25, shingle_size=2)

    # Create test articles with duplicates
    articles = [
        NewsArticle(
            title="RBI Mandates New KYC Norms for Digital Lending",
            summary="The Reserve Bank of India announced stricter KYC requirements",
            url="https://example.com/1",
            source_id="et", source_name="Economic Times",
            source_type=SourceType.RSS, source_tier=SourceTier.TIER_1,
            source_credibility=0.95, published_at=datetime.utcnow()
        ),
        NewsArticle(
            title="RBI announces new KYC rules for digital lenders",
            summary="RBI has mandated new Know Your Customer requirements",
            url="https://example.com/2",
            source_id="mint", source_name="Mint",
            source_type=SourceType.RSS, source_tier=SourceTier.TIER_1,
            source_credibility=0.95, published_at=datetime.utcnow()
        ),
        NewsArticle(
            title="Zepto Raises $200M at $5B Valuation",
            summary="Quick commerce startup Zepto closed a $200 million funding round",
            url="https://example.com/3",
            source_id="inc42", source_name="Inc42",
            source_type=SourceType.RSS, source_tier=SourceTier.TIER_2,
            source_credibility=0.88, published_at=datetime.utcnow()
        ),
    ]

    result = dedup.deduplicate(articles)
    print(f"  Input: {len(articles)} articles")
    print(f"  Output: {len(result)} articles")
    print(f"  Dedup rate: {(len(articles) - len(result)) / len(articles) * 100:.1f}%")

    for a in result:
        print(f"    Kept: {a.title}")

    return True


async def test_6_mini_pipeline():
    """Test 6: Run a mini pipeline (fetch → dedup → embed → cluster)."""
    print("\n" + "=" * 60)
    print("TEST 6: Mini Pipeline (Fetch → Dedup → Embed → Cluster)")
    print("=" * 60)

    from app.config import get_settings
    from app.tools.rss_tool import RSSTool
    from app.trends.engine import RecursiveTrendEngine

    settings = get_settings()

    # Fetch a small set
    print("  Step 1: Fetching articles...")
    rss = RSSTool()
    articles = await rss.fetch_all_sources(max_per_source=5, hours_ago=72)
    print(f"    Got {len(articles)} articles")

    if len(articles) < 5:
        print("    Not enough articles to test pipeline")
        return False

    # Create engine
    engine = RecursiveTrendEngine(
        dedup_threshold=settings.dedup_threshold,
        dedup_shingle_size=settings.dedup_shingle_size,
        semantic_dedup_threshold=settings.semantic_dedup_threshold,
        spacy_model=settings.spacy_model,
        umap_n_components=settings.umap_n_components,
        umap_n_neighbors=settings.umap_n_neighbors,
        umap_min_dist=settings.umap_min_dist,
        umap_metric=settings.umap_metric,
        min_cluster_size=settings.hdbscan_min_cluster_size,
        min_samples=settings.hdbscan_min_samples,
        cluster_selection_method=settings.hdbscan_cluster_selection,
        max_depth=1,  # Only 1 level for speed
        max_concurrent_llm=settings.engine_max_concurrent_llm,
        mock_mode=False,  # Test with REAL LLM
    )

    try:
        # Phase 0: Content scraping
        print("\n  Step 1.5: Content scraping...")
        start = time.time()
        articles = await engine._phase_scrape(articles)
        has_content = sum(1 for a in articles if a.content)
        print(f"    Scraped: {has_content}/{len(articles)} have full content ({time.time() - start:.2f}s)")

        # Phase 0.5: Event classification
        print("  Step 1.6: Event classification...")
        start = time.time()
        engine._phase_classify_events(articles)
        event_dist = engine.metrics.get("event_distribution", {})
        print(f"    Events: {event_dist} ({time.time() - start:.3f}s)")

        # Phase 0.7: Business relevance filter
        print("  Step 1.7: Relevance filter...")
        before = len(articles)
        articles = engine._phase_relevance_filter(articles)
        print(f"    Relevance filter: {before} → {len(articles)} ({before - len(articles)} non-business removed)")

        # Phase 1: Dedup
        print("\n  Step 2: Dedup...")
        start = time.time()
        articles = engine._phase_dedup(articles)
        print(f"    After dedup: {len(articles)} ({time.time() - start:.2f}s)")

        # Phase 2: NER
        print("  Step 3: NER...")
        start = time.time()
        articles = engine._phase_ner(articles)
        print(f"    NER done: {len(articles)} ({time.time() - start:.2f}s)")

        # Phase 2.5: Entity co-occurrence
        print("  Step 3.5: Entity co-occurrence...")
        start = time.time()
        engine._phase_entity_cooccurrence(articles)
        graph = engine.metrics.get("entity_graph", {})
        print(f"    Entities: {graph.get('total_entities', 0)}, Edges: {graph.get('total_edges', 0)} ({time.time() - start:.3f}s)")
        for conn in graph.get("top_connections", [])[:3]:
            safe_print(f"      {conn['entity_a']} <-> {conn['entity_b']}: {conn['strength']}x")
        for ce in graph.get("cross_event_entities", [])[:3]:
            safe_print(f"      Cross-event: {ce['entity']} -> {ce['events']}")

        # Phase 3: Embed
        print("  Step 4: Embed...")
        start = time.time()
        embeddings = engine._phase_embed(articles)
        print(f"    Embedded: {len(embeddings)} ({time.time() - start:.2f}s)")

        # Phase 3.5: Semantic dedup
        print("  Step 5: Semantic dedup...")
        start = time.time()
        articles, embeddings = engine._phase_semantic_dedup(
            articles, embeddings, threshold=settings.semantic_dedup_threshold
        )
        print(f"    After semantic dedup: {len(articles)} ({time.time() - start:.2f}s)")

        if len(articles) < 5:
            print("    Not enough articles after dedup for clustering")
            return True

        # Phase 4: UMAP
        print("  Step 6: UMAP reduction...")
        start = time.time()
        reduced = engine._phase_reduce(embeddings)
        print(f"    Reduced: {reduced.shape} ({time.time() - start:.2f}s)")

        # Phase 5: Cluster
        print("  Step 7: HDBSCAN clustering...")
        start = time.time()
        labels, noise_count = engine._phase_cluster(reduced)
        import numpy as np
        from collections import Counter
        label_counts = Counter(labels)
        n_clusters = len([l for l in label_counts if l >= 0])
        print(f"    Clusters: {n_clusters}, Noise: {noise_count} ({time.time() - start:.2f}s)")
        print(f"    Label distribution: {dict(sorted(label_counts.items()))}")

        # Phase 6: Keywords
        cluster_articles = engine._group_by_cluster(articles, labels)
        cluster_keywords = engine._phase_keywords(cluster_articles)
        for cid, kws in cluster_keywords.items():
            safe_print(f"    Cluster {cid} keywords: {kws[:5]}")

        # Phase 7: Signals
        cluster_signals = engine._phase_signals(cluster_articles)
        for cid, sigs in cluster_signals.items():
            print(f"    Cluster {cid} signal: {sigs.get('signal_strength', '?')}, score={sigs.get('trend_score', 0):.2f}")

        # Phase 8: LLM Synthesis (the critical test)
        print("\n  Step 8: LLM Synthesis (testing real LLM calls)...")
        start = time.time()
        cluster_summaries = await engine._phase_synthesize(cluster_articles, cluster_keywords)
        elapsed = time.time() - start
        print(f"    Synthesis done in {elapsed:.2f}s")

        success_count = 0
        fail_count = 0
        for cid, summary in cluster_summaries.items():
            if summary and summary.get("trend_title"):
                safe_print(f"    Cluster {cid}: '{summary['trend_title']}' ({summary.get('severity', '?')})")
                # Show enhanced fields
                buying = summary.get("buying_intent", {})
                if buying:
                    safe_print(f"      Buying intent: {buying.get('signal_type', 'N/A')} | urgency: {buying.get('urgency', 'N/A')}")
                    safe_print(f"      Who needs help: {buying.get('who_needs_help', 'N/A')[:80]}")
                    safe_print(f"      Pitch hook: {buying.get('pitch_hook', 'N/A')[:80]}")
                w5h1 = summary.get("event_5w1h", {})
                if w5h1:
                    safe_print(f"      5W1H: WHO={w5h1.get('who', '?')[:40]} | WHAT={w5h1.get('what', '?')[:40]}")
                chain = summary.get("causal_chain", [])
                if chain:
                    safe_print(f"      Causal chain: {' -> '.join(c[:30] for c in chain[:3])}")
                companies = summary.get("affected_companies", [])
                if companies:
                    safe_print(f"      Companies: {', '.join(companies[:5])}")
                # Show recommended services from our portfolio
                services = summary.get("recommended_services", [])
                if services:
                    for svc in services[:2]:
                        safe_print(f"      Service: {svc.get('service', '?')} → {svc.get('specific_offering', '?')}")
                        safe_print(f"        Target: {str(svc.get('target_companies', '?'))[:80]}")
                success_count += 1
            else:
                safe_print(f"    Cluster {cid}: FAILED - {summary}")
                fail_count += 1

        print(f"\n    LLM Synthesis: {success_count} succeeded, {fail_count} failed")

        return True

    except Exception as e:
        import traceback
        print(f"  Pipeline failed: {e}")
        traceback.print_exc()
        return False


async def main():
    """Run all diagnostic tests."""
    print("=" * 60)
    print("PIPELINE DIAGNOSTIC TEST SUITE")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    tests = [
        ("Settings", test_1_settings),
        ("Embedding Model", test_2_embedding_model),
        ("RSS Fetch", test_3_rss_fetch),
        ("LLM Providers", test_4_llm_providers),
        ("Deduplication", test_5_dedup),
        ("Mini Pipeline", test_6_mini_pipeline),
    ]

    results = {}
    for name, test_func in tests:
        try:
            result = await test_func()
            results[name] = "PASS" if result else "FAIL"
        except Exception as e:
            print(f"\n  *** TEST EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results[name] = f"ERROR: {e}"

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        status = "PASS" if result == "PASS" else "FAIL"
        print(f"  [{status}] {name}: {result}")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
