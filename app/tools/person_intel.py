"""
Person intelligence -- scrape context about a contact for hyper-personalized outreach.

Sources (in priority order):
1. Web search for "{person_name} {company_name}" -- extract public profile snippets
2. News search for recent mentions
3. LLM synthesis of talking points from scraped data + trend context

Output: PersonContext with talking points for email personalization.

Usage:
    from app.tools.person_intel import gather_person_context, PersonContext

    ctx = await gather_person_context(
        person_name="Jensen Huang",
        company_name="NVIDIA",
        role="CEO",
        trend_context="AI chip demand surge in 2026",
    )
    print(ctx.talking_points)
"""

import asyncio
import logging
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Module-level concurrency control ─────────────────────────────
# Limit concurrent person lookups to avoid hammering search APIs
_PERSON_SEM = asyncio.Semaphore(3)

# Total timeout for the entire gather_person_context call (seconds)
_GATHER_TIMEOUT = 25.0

# Max snippets to keep per search type
_MAX_PROFILE_SNIPPETS = 5
_MAX_NEWS_SNIPPETS = 5


# ══════════════════════════════════════════════════════════════════
# PersonContext Model
# ══════════════════════════════════════════════════════════════════


class PersonContext(BaseModel):
    """Scraped context about a person for email personalization."""

    person_name: str
    company_name: str
    role: str

    # Scraped intelligence
    background_summary: str = ""  # 2-3 sentences about their background
    recent_focus: str = ""  # What they've been working on recently
    notable_achievements: list[str] = Field(default_factory=list)
    shared_interests: list[str] = Field(default_factory=list)  # Topics they care about
    talking_points: list[str] = Field(default_factory=list)  # Points to reference in outreach

    # Sources
    linkedin_headline: str = ""
    recent_posts: list[str] = Field(default_factory=list)  # Titles/summaries of content
    news_mentions: list[str] = Field(default_factory=list)  # Recent news mentioning them
    sources: list[str] = Field(default_factory=list)  # URLs where context was found

    # Deep intel fields (populated by Stage 2 — background ScrapeGraphAI)
    career_history: list[str] = Field(default_factory=list)  # Previous roles/companies
    speaking_topics: list[str] = Field(default_factory=list)  # Conference/podcast topics
    github_profile: str = ""  # Repos, languages, contribution summary
    content_themes: list[str] = Field(default_factory=list)  # Recurring themes across content
    deep_enriched: bool = False  # Flag: has Stage 2 completed?
    deep_enriched_at: Optional[str] = None  # ISO timestamp for staleness check

    @property
    def has_context(self) -> bool:
        """True if we found any meaningful intel beyond the basics."""
        return bool(
            self.background_summary
            or self.talking_points
            or self.news_mentions
            or self.notable_achievements
            or self.recent_posts
            or self.content_themes
        )


# ══════════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════════


async def gather_person_context(
    person_name: str,
    company_name: str,
    role: str,
    trend_context: str = "",
    deep: bool = False,
    linkedin_url: str = "",
) -> PersonContext:
    """Scrape all available context about a person for outreach personalization.

    Two stages:
      Stage 1 (fast, 2-5s): Web search + news search + LLM synthesis
      Stage 2 (background, 10-60s): ScrapeGraphAI on discovered URLs (optional)

    Always returns a PersonContext (never raises). On failure, returns
    a bare context with only name/company/role filled in.

    Args:
        person_name: Full name of the contact.
        company_name: Company they work at.
        role: Their job title/role.
        trend_context: The trend this outreach is about (for relevance).
        deep: If True, fire Stage 2 background enrichment (ScrapeGraphAI).
        linkedin_url: LinkedIn URL from Apollo (used for DDG enrichment, not direct scraping).

    Returns:
        PersonContext with whatever intel was gathered.
    """
    bare = PersonContext(
        person_name=person_name,
        company_name=company_name,
        role=role,
    )

    try:
        async with _PERSON_SEM:
            ctx = await asyncio.wait_for(
                _gather_impl(person_name, company_name, role, trend_context),
                timeout=_GATHER_TIMEOUT,
            )

        # Stage 2: Fire background deep enrichment if enabled
        if deep:
            from app.config import get_settings
            settings = get_settings()
            if settings.person_deep_intel_enabled:
                asyncio.create_task(
                    _deep_person_scrape(ctx, linkedin_url=linkedin_url)
                )

        return ctx
    except asyncio.TimeoutError:
        logger.debug(f"person_intel timed out for {person_name} @ {company_name}")
        return bare
    except Exception as e:
        logger.debug(f"person_intel failed for {person_name} @ {company_name}: {e}")
        return bare


# ══════════════════════════════════════════════════════════════════
# Internal Implementation
# ══════════════════════════════════════════════════════════════════


async def _gather_impl(
    person_name: str,
    company_name: str,
    role: str,
    trend_context: str,
) -> PersonContext:
    """Core implementation -- runs inside semaphore + timeout."""
    ctx = PersonContext(
        person_name=person_name,
        company_name=company_name,
        role=role,
    )

    # ── Step 1 + 2: Parallel web search + news search ─────────
    profile_query = f'"{person_name}" "{company_name}" {role}'
    news_query = f'"{person_name}" {company_name}'

    profile_results, news_results = await asyncio.gather(
        _safe_search(profile_query, max_results=_MAX_PROFILE_SNIPPETS),
        _safe_search(news_query, max_results=_MAX_NEWS_SNIPPETS, news_mode=True),
    )

    # Collect snippets and sources from profile search
    profile_snippets: list[str] = []
    for r in profile_results:
        if r.snippet:
            profile_snippets.append(r.snippet)
        if r.url:
            ctx.sources.append(r.url)
        # Try to detect LinkedIn headline from search results
        if not ctx.linkedin_headline and r.url and "linkedin.com" in r.url:
            ctx.linkedin_headline = r.snippet[:200] if r.snippet else ""

    # Collect news mentions
    for r in news_results:
        if r.title:
            ctx.news_mentions.append(r.title)
        if r.url and r.url not in ctx.sources:
            ctx.sources.append(r.url)

    # ── Step 3: LLM synthesis of talking points ───────────────
    # Only call LLM if we have something to synthesize
    raw_intel = _build_raw_intel(profile_snippets, ctx.news_mentions)
    if raw_intel:
        synthesis = await _synthesize_talking_points(
            person_name=person_name,
            company_name=company_name,
            role=role,
            raw_intel=raw_intel,
            trend_context=trend_context,
        )
        if synthesis:
            ctx.background_summary = synthesis.get("background_summary", "")
            ctx.recent_focus = synthesis.get("recent_focus", "")
            ctx.notable_achievements = _ensure_list(synthesis.get("notable_achievements"))
            ctx.shared_interests = _ensure_list(synthesis.get("shared_interests"))
            ctx.talking_points = _ensure_list(synthesis.get("talking_points"))

    logger.info(
        f"person_intel: {person_name} @ {company_name} -- "
        f"{len(ctx.talking_points)} talking points, "
        f"{len(ctx.news_mentions)} news mentions, "
        f"{len(ctx.sources)} sources"
    )
    return ctx


async def _safe_search(
    query: str,
    max_results: int = 5,
    news_mode: bool = False,
) -> list:
    """Wrapper around web_intel.search that never raises."""
    try:
        from app.tools.web.web_intel import search
        return await search(query, max_results=max_results, news_mode=news_mode)
    except Exception as e:
        logger.debug(f"person_intel search failed for '{query[:60]}': {e}")
        return []


def _build_raw_intel(profile_snippets: list[str], news_mentions: list[str]) -> str:
    """Combine all scraped text into a single block for LLM synthesis."""
    parts: list[str] = []

    if profile_snippets:
        parts.append("PROFILE SEARCH RESULTS:")
        for i, s in enumerate(profile_snippets[:_MAX_PROFILE_SNIPPETS], 1):
            parts.append(f"  {i}. {s[:300]}")

    if news_mentions:
        parts.append("\nRECENT NEWS MENTIONS:")
        for i, n in enumerate(news_mentions[:_MAX_NEWS_SNIPPETS], 1):
            parts.append(f"  {i}. {n[:200]}")

    return "\n".join(parts)


async def _synthesize_talking_points(
    person_name: str,
    company_name: str,
    role: str,
    raw_intel: str,
    trend_context: str,
) -> Optional[dict]:
    """Use LLM to synthesize scraped intel into structured talking points.

    Returns dict with keys: background_summary, recent_focus,
    notable_achievements, shared_interests, talking_points.
    Returns None on failure.
    """
    try:
        from app.tools.llm.llm_service import LLMService

        llm = LLMService(lite=True)  # Use cheap model for synthesis

        prompt = f"""Analyze the following web intelligence about a person and produce personalized outreach talking points.

PERSON: {person_name}
ROLE: {role} at {company_name}
{f'TREND CONTEXT (what our outreach is about): {trend_context}' if trend_context else ''}

RAW INTELLIGENCE:
{raw_intel}

Based on this information, produce a JSON object with these fields:
- background_summary: 2-3 sentences about their professional background (what they're known for)
- recent_focus: 1-2 sentences about what they've been working on or talking about recently
- notable_achievements: list of 1-3 specific achievements, promotions, awards, or initiatives
- shared_interests: list of 1-3 professional topics they clearly care about
- talking_points: list of 2-3 SPECIFIC things to reference in a cold outreach email to build rapport (e.g., "their recent keynote on AI safety at CES 2026", "the company's Q4 expansion into APAC markets")

RULES:
- Only include facts supported by the raw intelligence above
- Do NOT fabricate achievements or events
- If the raw intelligence is thin, return shorter lists -- do not pad with generic filler
- talking_points should be SPECIFIC and VERIFIABLE, not generic flattery
- If the intelligence doesn't support any talking points, return empty lists"""

        system_prompt = (
            "You are a sales intelligence analyst. Extract factual, specific "
            "insights from web search results for outreach personalization. "
            "Never fabricate information. Respond with valid JSON only."
        )

        result = await llm.generate_json(
            prompt=prompt,
            system_prompt=system_prompt,
        )

        if isinstance(result, dict) and "error" not in result:
            return result
        return None

    except Exception as e:
        logger.debug(f"person_intel LLM synthesis failed for {person_name}: {e}")
        return None


def _ensure_list(val) -> list[str]:
    """Coerce a value to list[str] safely."""
    if val is None:
        return []
    if isinstance(val, str):
        return [val] if val.strip() else []
    if isinstance(val, list):
        return [str(v).strip() for v in val if v and str(v).strip()]
    return []


# ══════════════════════════════════════════════════════════════════
# Stage 2: Deep Person Enrichment (Background, ScrapeGraphAI)
# ══════════════════════════════════════════════════════════════════

# Semaphore for deep scraping (separate from Stage 1 — heavier operations)
_DEEP_SEM = asyncio.Semaphore(2)


def _categorize_urls(urls: list[str]) -> dict[str, list[str]]:
    """Categorize discovered URLs by source type for targeted scraping."""
    from urllib.parse import urlparse

    categories: dict[str, list[str]] = {
        "medium": [],
        "substack": [],
        "github": [],
        "company_bio": [],
        "personal": [],
    }

    for url in urls:
        try:
            parsed = urlparse(url)
            host = parsed.hostname or ""
            path = parsed.path.lower()

            if "medium.com" in host:
                categories["medium"].append(url)
            elif "substack.com" in host:
                categories["substack"].append(url)
            elif "github.com" in host and path.count("/") <= 2:
                # Only profile pages, not deep repo links
                categories["github"].append(url)
            elif any(kw in path for kw in ("/team", "/about", "/people", "/leadership")):
                categories["company_bio"].append(url)
            elif "linkedin.com" not in host and "twitter.com" not in host and "x.com" not in host:
                categories["personal"].append(url)
        except Exception:
            continue

    return categories


async def _scrape_url_with_scrapegraph(url: str, prompt: str, timeout: int = 60) -> dict:
    """Scrape a single URL using SmartScraperGraph. Runs in a thread (sync graph).

    Returns extracted dict or empty dict on failure.
    """
    try:
        from app.config import get_settings
        settings = get_settings()

        openai_key = settings.openai_api_key
        if not openai_key:
            return {}

        from scrapegraphai.graphs import SmartScraperGraph

        config = {
            "llm": {
                "api_key": openai_key,
                "model": settings.scrapegraph_model,
                "temperature": 0.1,
            },
            "verbose": False,
            "headless": True,
        }

        scraper = SmartScraperGraph(
            prompt=prompt,
            source=url,
            config=config,
        )

        result = await asyncio.wait_for(
            asyncio.to_thread(scraper.run),
            timeout=timeout,
        )
        return result if isinstance(result, dict) else {}

    except Exception as e:
        logger.debug(f"SmartScraper failed for {url}: {e}")
        return {}


async def _scrape_blog_posts(urls: list[str], person_name: str) -> list[str]:
    """Scrape Medium/Substack/personal blog for recent posts."""
    from app.config import get_settings
    settings = get_settings()

    posts: list[str] = []
    max_urls = min(len(urls), settings.person_intel_max_urls)

    for url in urls[:max_urls]:
        result = await _scrape_url_with_scrapegraph(
            url,
            prompt=(
                f"Extract article titles and key topics from {person_name}'s blog or publication page. "
                f"Return JSON with: articles (list of objects with title and topic), "
                f"themes (list of recurring professional themes)."
            ),
            timeout=settings.scrapegraph_timeout,
        )
        articles = result.get("articles", [])
        if isinstance(articles, list):
            for a in articles[:5]:
                if isinstance(a, dict) and a.get("title"):
                    posts.append(f"{a['title']}: {a.get('topic', '')}")
                elif isinstance(a, str):
                    posts.append(a)

    return posts


async def _scrape_github_profile(urls: list[str], person_name: str) -> str:
    """Scrape GitHub profile for repos/contributions summary."""
    if not urls:
        return ""

    result = await _scrape_url_with_scrapegraph(
        urls[0],
        prompt=(
            f"Extract {person_name}'s GitHub profile information: "
            f"bio, popular repositories (name + description), programming languages, "
            f"and contribution activity. Return JSON with: bio, repos (list), languages (list)."
        ),
        timeout=45,
    )

    parts = []
    if result.get("bio"):
        parts.append(result["bio"])
    repos = result.get("repos", [])
    if isinstance(repos, list) and repos:
        repo_names = [r.get("name", str(r)) if isinstance(r, dict) else str(r) for r in repos[:3]]
        parts.append(f"Key repos: {', '.join(repo_names)}")
    langs = result.get("languages", [])
    if isinstance(langs, list) and langs:
        parts.append(f"Languages: {', '.join(str(l) for l in langs[:5])}")

    return ". ".join(parts)


async def _search_speaking_topics(person_name: str, company_name: str) -> list[str]:
    """Search for conference talks, podcasts, and keynotes."""
    try:
        from app.tools.web.web_intel import search
        query = f'"{person_name}" {company_name} conference OR podcast OR keynote OR webinar'
        results = await search(query, max_results=5, news_mode=False)

        topics = []
        for r in results:
            if r.title and person_name.split()[0].lower() in r.title.lower():
                topics.append(r.title[:150])
            elif r.snippet and person_name.split()[0].lower() in r.snippet.lower():
                topics.append(r.snippet[:150])

        return topics[:5]
    except Exception as e:
        logger.debug(f"Speaking topic search failed for {person_name}: {e}")
        return []


async def _deep_person_scrape(
    ctx: PersonContext,
    linkedin_url: str = "",
) -> None:
    """Stage 2: Background deep enrichment using ScrapeGraphAI.

    Runs AFTER Stage 1 completes. Enriches the ctx object in-place
    with data from Medium/Substack, GitHub, company bio pages, and
    conference/podcast mentions.

    This function never raises — all failures are logged and silently ignored.
    """
    async with _DEEP_SEM:
        try:
            from app.config import get_settings
            from datetime import datetime, timezone
            settings = get_settings()

            enabled_sources = set(
                s.strip() for s in settings.person_intel_sources.split(",") if s.strip()
            )

            # Categorize URLs discovered in Stage 1
            url_categories = _categorize_urls(ctx.sources)

            tasks = []

            # Blog posts (Medium + Substack)
            blog_urls = []
            if "medium" in enabled_sources:
                blog_urls.extend(url_categories["medium"])
            if "substack" in enabled_sources:
                blog_urls.extend(url_categories["substack"])
            if blog_urls:
                tasks.append(("blog", _scrape_blog_posts(blog_urls, ctx.person_name)))

            # GitHub profile
            if "github" in enabled_sources and url_categories["github"]:
                tasks.append(("github", _scrape_github_profile(
                    url_categories["github"], ctx.person_name
                )))

            # Company bio page
            if "company_bio" in enabled_sources and url_categories["company_bio"]:
                tasks.append(("bio", _scrape_url_with_scrapegraph(
                    url_categories["company_bio"][0],
                    prompt=(
                        f"Extract {ctx.person_name}'s professional bio, achievements, "
                        f"publications, and speaking engagements from this page. "
                        f"Return JSON with: bio, achievements (list), publications (list)."
                    ),
                    timeout=settings.scrapegraph_timeout,
                )))

            # Conference/podcast topics (search-based, no ScrapeGraphAI)
            if "conferences" in enabled_sources:
                tasks.append(("speaking", _search_speaking_topics(
                    ctx.person_name, ctx.company_name
                )))

            if not tasks:
                return

            # Run all scraping tasks in parallel
            labels = [t[0] for t in tasks]
            results = await asyncio.gather(
                *[t[1] for t in tasks],
                return_exceptions=True,
            )

            for label, result in zip(labels, results):
                if isinstance(result, Exception):
                    logger.debug(f"Deep scrape {label} failed for {ctx.person_name}: {result}")
                    continue

                if label == "blog" and isinstance(result, list):
                    ctx.recent_posts.extend(result[:10])
                elif label == "github" and isinstance(result, str) and result:
                    ctx.github_profile = result
                elif label == "bio" and isinstance(result, dict):
                    if result.get("achievements"):
                        achievements = _ensure_list(result["achievements"])
                        ctx.notable_achievements.extend(achievements[:5])
                    if result.get("bio") and not ctx.background_summary:
                        ctx.background_summary = str(result["bio"])[:500]
                elif label == "speaking" and isinstance(result, list):
                    ctx.speaking_topics.extend(result[:5])

            # Synthesize content themes from all deep intel
            if ctx.recent_posts or ctx.speaking_topics:
                themes = await _synthesize_content_themes(ctx)
                if themes:
                    ctx.content_themes = themes

            ctx.deep_enriched = True
            ctx.deep_enriched_at = datetime.now(timezone.utc).isoformat()

            logger.info(
                f"Deep person intel: {ctx.person_name} -- "
                f"{len(ctx.recent_posts)} posts, "
                f"{len(ctx.speaking_topics)} talks, "
                f"github={'yes' if ctx.github_profile else 'no'}"
            )

        except Exception as e:
            logger.debug(f"Deep person scrape failed for {ctx.person_name}: {e}")


async def _synthesize_content_themes(ctx: PersonContext) -> list[str]:
    """Use LLM to extract recurring themes across all person content."""
    try:
        from app.tools.llm.llm_service import LLMService
        llm = LLMService(lite=True)

        content_items = []
        for post in ctx.recent_posts[:5]:
            content_items.append(f"Blog/article: {post}")
        for topic in ctx.speaking_topics[:3]:
            content_items.append(f"Talk/conference: {topic}")
        if ctx.github_profile:
            content_items.append(f"GitHub: {ctx.github_profile[:200]}")

        if not content_items:
            return []

        prompt = (
            f"Given {ctx.person_name}'s content:\n"
            + "\n".join(f"- {item}" for item in content_items)
            + "\n\nExtract 3-5 recurring professional themes or interests. "
            "Return as a JSON array of short theme strings."
        )

        result = await asyncio.wait_for(
            llm.generate_json(prompt=prompt, system_prompt="Return valid JSON only."),
            timeout=6.0,
        )

        if isinstance(result, list):
            return _ensure_list(result)[:5]
        elif isinstance(result, dict):
            for v in result.values():
                if isinstance(v, list):
                    return _ensure_list(v)[:5]

    except Exception as e:
        logger.debug(f"Content theme synthesis failed for {ctx.person_name}: {e}")

    return []
