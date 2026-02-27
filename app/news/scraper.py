"""
Article content scraper using trafilatura.

WHY SCRAPE FULL CONTENT:
  RSS feeds only provide title + summary (~50-100 words).
  Full articles give 500-2000 words → 10-20x richer embeddings → better clustering.

APPROACH:
  - Scrape top N articles by source credibility (not all — RSS summary is fallback)
  - Async parallel scraping with concurrency limit (avoid hammering sources)
  - trafilatura handles: paywall detection, boilerplate removal, encoding issues
  - Fallback to BeautifulSoup if trafilatura fails

PERFORMANCE:
  50 articles:  ~8-12 sec (parallel, 10 concurrent)
  150 articles: ~20-30 sec (default cap)
"""

import asyncio
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Silence trafilatura completely — "empty HTML tree: None" and "parsed tree length: 1"
# are expected noise from paywalled/JS-heavy sites, not actionable errors for us.
for _noisy in ("trafilatura", "trafilatura.core", "trafilatura.utils", "trafilatura.htmlprocessing",
               "trafilatura.settings", "trafilatura.feeds"):
    _tlog = logging.getLogger(_noisy)
    _tlog.setLevel(logging.CRITICAL + 1)  # Above CRITICAL — suppress everything
    _tlog.propagate = False               # Don't bubble up to root logger

# Browser-like headers to avoid being blocked
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

# Max concurrent HTTP requests to avoid overwhelming sources
_MAX_CONCURRENT = 10

# Skip domains with hard paywalls that return ZERO useful content.
# NOTE: ET and Mint removed — they have soft paywalls where trafilatura
# still extracts 300-500 words (far richer than the 50-word RSS summary).
_SKIP_DOMAINS = frozenset({
    "wsj.com",
    "ft.com",
    "bloomberg.com",
})


def _extract_text(html_content: str, url: str) -> Optional[str]:
    """Extract article text from HTML using trafilatura."""
    try:
        import trafilatura
        text = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=False,
            no_fallback=False,  # Use fallback extractors if main fails
            favor_precision=True,
            deduplicate=True,
        )
        if text and len(text) > 100:
            return text
    except Exception as e:
        logger.debug(f"trafilatura failed for {url}: {e}")

    # Fallback: simple BeautifulSoup extraction
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, "lxml")
        # Remove nav, header, footer, scripts
        for tag in soup.find_all(["nav", "header", "footer", "script", "style", "aside"]):
            tag.decompose()
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30)
        if len(text) > 100:
            return text
    except Exception as e:
        logger.debug(f"BeautifulSoup fallback failed for {url}: {e}")

    return None


async def scrape_article(url: str, client: httpx.AsyncClient) -> Optional[str]:
    """Scrape full text content from a single article URL."""
    if not url or url == "#":
        return None

    from urllib.parse import urlparse
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")

    # Skip paywalled domains
    if domain in _SKIP_DOMAINS:
        return None

    # Google News redirect URLs: follow the redirect to get the actual publisher URL.
    # These redirect to the real article (ET, Mint, BS, etc.) — don't skip them.
    try:
        response = await client.get(url, headers=_HEADERS, follow_redirects=True, timeout=15.0)
        if response.status_code != 200:
            logger.debug(f"HTTP {response.status_code} for {domain}")
            return None

        # Check final URL after redirects — might have landed on a blocked domain
        final_domain = urlparse(str(response.url)).netloc.replace("www.", "")
        if final_domain in _SKIP_DOMAINS:
            return None

        return _extract_text(response.text, url)
    except Exception as e:
        logger.debug(f"Scrape failed for {domain}: {e}")
        return None


async def scrape_articles(
    articles: list,
    max_concurrent: int = _MAX_CONCURRENT,
    max_articles: int = 150,
) -> int:
    """
    Scrape full content for the top N articles by source credibility.

    Modifies articles in-place (sets article.content).
    Returns count of articles successfully enriched.
    Articles beyond max_articles keep their RSS summary as embedding fallback.

    Args:
        articles: List of NewsArticle objects
        max_concurrent: Max parallel HTTP connections
        max_articles: Cap on how many articles to scrape (highest credibility first)
    """
    # Only scrape articles without content — sorted by source credibility descending
    candidates = sorted(
        [a for a in articles if not a.content and a.url],
        key=lambda a: getattr(a, "source_credibility", 0.5),
        reverse=True,
    )
    to_scrape = candidates[:max_articles]

    if not candidates:
        logger.info("All articles already have content, skipping scrape")
        return 0

    skipped = len(candidates) - len(to_scrape)
    skip_note = f", {skipped} skipped (RSS summary used)" if skipped else ""
    logger.info(f"Scraping full content for {len(to_scrape)}/{len(articles)} articles{skip_note}...")

    semaphore = asyncio.Semaphore(max_concurrent)
    enriched = 0

    # Share a single client for connection pooling — much faster than one per article
    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
        async def _scrape_one(article):
            nonlocal enriched
            async with semaphore:
                try:
                    # Wrap with overall timeout of 20 seconds (httpx has 15s, but this ensures fast failure)
                    text = await asyncio.wait_for(
                        scrape_article(article.url, client),
                        timeout=20.0
                    )
                    if text:
                        article.content = text[:5000]  # Cap at 5000 chars
                        enriched += 1
                except asyncio.TimeoutError:
                    logger.debug(f"⏱️ Scrape timeout for {article.url}")
                except Exception as e:
                    logger.debug(f"Scrape error for {article.url}: {e}")

        await asyncio.gather(*[_scrape_one(a) for a in to_scrape], return_exceptions=True)

    logger.info(f"Scraped content: {enriched}/{len(to_scrape)} articles enriched")
    return enriched
