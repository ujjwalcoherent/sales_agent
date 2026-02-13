"""
Trend Synthesizer - Generates MajorTrend objects from ArticleCluster.

Uses LLM to synthesize coherent trends from clustered articles.
Supports Ollama (local/free), Gemini, and Groq.
"""

import logging
import re
from typing import List, Dict, Optional, Any

from ..schemas import (
    NewsArticle, ArticleCluster, MajorTrend, SectorImpact,
    Sector, TrendType, Severity, ImpactType, ServiceType,
    ConfidenceScore, GeoLocation
)
from .llm_service import LLMService

logger = logging.getLogger(__name__)

# Mapping from LLM-returned sector keywords to Sector enum values.
# Multiple keywords can map to the same sector for fuzzy matching.
SECTOR_KEYWORD_MAP: Dict[str, Sector] = {
    "fintech": Sector.BFSI,
    "banking": Sector.BFSI,
    "finance": Sector.BFSI,
    "bfsi": Sector.BFSI,
    "insurance": Sector.BFSI,
    "nbfc": Sector.BFSI,
    "healthcare": Sector.HEALTHCARE_PHARMA,
    "pharma": Sector.HEALTHCARE_PHARMA,
    "health": Sector.HEALTHCARE_PHARMA,
    "manufacturing": Sector.MANUFACTURING,
    "retail": Sector.RETAIL_FMCG,
    "fmcg": Sector.RETAIL_FMCG,
    "ecommerce": Sector.RETAIL_FMCG,
    "consumer": Sector.RETAIL_FMCG,
    "logistics": Sector.LOGISTICS,
    "supply chain": Sector.LOGISTICS,
    "real_estate": Sector.REAL_ESTATE,
    "real estate": Sector.REAL_ESTATE,
    "construction": Sector.REAL_ESTATE,
    "education": Sector.EDUCATION,
    "edtech": Sector.EDUCATION,
    "agriculture": Sector.AGRICULTURE,
    "agritech": Sector.AGRICULTURE,
    "energy": Sector.ENERGY_UTILITIES,
    "utilities": Sector.ENERGY_UTILITIES,
    "oil": Sector.ENERGY_UTILITIES,
    "power": Sector.ENERGY_UTILITIES,
    "telecom": Sector.TELECOM,
    "it": Sector.IT_TECHNOLOGY,
    "technology": Sector.IT_TECHNOLOGY,
    "software": Sector.IT_TECHNOLOGY,
    "tech": Sector.IT_TECHNOLOGY,
    "automotive": Sector.AUTOMOTIVE,
    "ev": Sector.AUTOMOTIVE,
    "auto": Sector.AUTOMOTIVE,
}

# Mapping from LLM-returned service keywords to ServiceType enum values.
SERVICE_KEYWORD_MAP: Dict[str, ServiceType] = {
    "procurement": ServiceType.PROCUREMENT_INTELLIGENCE,
    "market intelligence": ServiceType.MARKET_INTELLIGENCE,
    "competitive": ServiceType.COMPETITIVE_INTELLIGENCE,
    "monitoring": ServiceType.MARKET_MONITORING,
    "industry": ServiceType.INDUSTRY_ANALYSIS,
    "technology": ServiceType.TECHNOLOGY_RESEARCH,
    "cross border": ServiceType.CROSS_BORDER_EXPANSION,
    "consumer": ServiceType.CONSUMER_INSIGHTS,
    "consulting": ServiceType.CONSULTING_ADVISORY,
    "advisory": ServiceType.CONSULTING_ADVISORY,
}


def _parse_enum_value(raw: str, enum_class: type, default):
    """Parse a string into an enum value, returning the default on failure."""
    try:
        return enum_class(raw.lower())
    except ValueError:
        return default


def _match_keywords(names: List[str], keyword_map: Dict[str, Any], limit: int = 0) -> list:
    """Match a list of raw names against a keyword map, returning unique matched values.

    Args:
        names: Raw strings from LLM output to match against keywords.
        keyword_map: Mapping of keywords to enum/object values.
        limit: Maximum results to return. 0 means no limit.
    """
    matched = []
    for name in names:
        name_lower = name.lower().strip()
        for keyword, value in keyword_map.items():
            if keyword in name_lower and value not in matched:
                matched.append(value)
                break
    if limit:
        return matched[:limit]
    return matched


class TrendSynthesizer:
    """
    Synthesize major trends from article clusters using LLM.

    Process:
    1. Extract key facts from cluster articles
    2. Generate trend title and summary
    3. Classify trend type and severity
    4. Identify affected sectors
    5. Compute confidence score
    """

    SYSTEM_PROMPT = """You are a business intelligence analyst specializing in Indian markets.
Your task is to synthesize news articles into coherent market trends.
Focus on:
- What is the core event/development?
- Which industries/sectors are affected?
- What is the business impact?
- How urgent/significant is this?

Always respond with valid JSON."""

    def __init__(self, llm: Optional[LLMService] = None, mock_mode: bool = False):
        """
        Initialize trend synthesizer.

        Args:
            llm: LLM service instance (creates one if not provided)
            mock_mode: Use mock responses
        """
        self.llm = llm or LLMService(mock_mode=mock_mode)
        self.mock_mode = mock_mode

    async def synthesize_trend(
        self,
        cluster: ArticleCluster,
        articles: List[NewsArticle]
    ) -> Optional[MajorTrend]:
        """
        Synthesize a MajorTrend from a cluster of articles.

        Args:
            cluster: ArticleCluster object
            articles: List of all articles (to look up cluster members)

        Returns:
            MajorTrend object or None if synthesis fails
        """
        # Get articles in this cluster
        cluster_articles = [a for a in articles if a.id in cluster.article_ids]

        if not cluster_articles:
            logger.warning(f"No articles found for cluster {cluster.id}")
            return None

        # Build context from articles
        context = self._build_context(cluster_articles, cluster)

        # Generate trend via LLM
        trend_data = await self._generate_trend(context)

        if not trend_data or "error" in trend_data:
            logger.error(f"Failed to generate trend: {trend_data}")
            return None

        # Build MajorTrend object
        trend = self._build_trend(trend_data, cluster, cluster_articles)

        logger.info(f"Synthesized trend: {trend.trend_title}")
        return trend

    async def synthesize_all(
        self,
        clusters: List[ArticleCluster],
        articles: List[NewsArticle]
    ) -> List[MajorTrend]:
        """
        Synthesize trends from all clusters.

        Args:
            clusters: List of ArticleCluster objects
            articles: List of all articles

        Returns:
            List of MajorTrend objects
        """
        trends = []

        for cluster in clusters:
            try:
                trend = await self.synthesize_trend(cluster, articles)
                if trend:
                    trends.append(trend)
            except Exception as e:
                logger.error(f"Failed to synthesize cluster {cluster.id}: {e}")

        logger.info(f"Synthesized {len(trends)} trends from {len(clusters)} clusters")
        return trends

    def _build_context(
        self,
        articles: List[NewsArticle],
        cluster: ArticleCluster
    ) -> str:
        """Build context string from articles for LLM."""
        context_parts = []

        # Add cluster metadata
        context_parts.append(f"CLUSTER INFO:")
        context_parts.append(f"- Articles: {len(articles)}")
        context_parts.append(f"- Sources: {cluster.unique_sources}")
        context_parts.append(f"- Common entities: {', '.join(cluster.common_entities[:5])}")
        context_parts.append(f"- Common keywords: {', '.join(cluster.common_keywords[:10])}")
        context_parts.append("")

        # Add article summaries
        context_parts.append("ARTICLES:")
        for i, article in enumerate(articles[:10], 1):  # Limit to 10 articles
            context_parts.append(f"\n[Article {i}]")
            context_parts.append(f"Source: {article.source_name} (credibility: {article.source_credibility:.2f})")
            context_parts.append(f"Title: {article.title}")
            context_parts.append(f"Summary: {article.summary[:300]}...")
            context_parts.append(f"Published: {article.published_at.strftime('%Y-%m-%d %H:%M')}")

        return "\n".join(context_parts)

    async def _generate_trend(self, context: str) -> Dict[str, Any]:
        """Generate trend data via LLM."""
        prompt = f"""Analyze these related news articles and synthesize them into a single coherent market trend.

{context}

Respond with JSON:
{{
    "trend_title": "Concise title (max 15 words)",
    "trend_summary": "2-3 paragraph summary of what's happening and why it matters",
    "trend_type": "One of: regulation, policy, funding, acquisition, partnership, expansion, layoffs, hiring, product_launch, ipo, bankruptcy, technology, supply_chain, price_change, general",
    "severity": "One of: critical, high, medium, low",
    "key_entities": ["List of companies, people, policies mentioned"],
    "key_facts": ["List of verified facts from the articles"],
    "key_numbers": ["Any statistics, amounts, percentages mentioned"],
    "primary_sectors": ["List of 1-3 most affected sectors"],
    "secondary_sectors": ["List of 1-3 indirectly affected sectors"],
    "affected_regions": ["List of affected states/cities in India"],
    "is_national": true or false,
    "lifecycle_stage": "One of: emerging, growing, peak, declining",
    "confidence_explanation": "Why you are confident in this synthesis"
}}"""

        try:
            response = await self.llm.generate_json(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT
            )
            return response
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {"error": str(e)}

    def _build_trend(
        self,
        data: Dict[str, Any],
        cluster: ArticleCluster,
        articles: List[NewsArticle]
    ) -> MajorTrend:
        """Build MajorTrend object from LLM response."""

        trend_type = _parse_enum_value(data.get("trend_type", "general"), TrendType, TrendType.GENERAL)
        severity = _parse_enum_value(data.get("severity", "medium"), Severity, Severity.MEDIUM)
        primary_sectors = self._parse_sectors(data.get("primary_sectors", []))
        secondary_sectors = self._parse_sectors(data.get("secondary_sectors", []))

        # Compute confidence score
        confidence = self._compute_confidence(cluster, articles, data)

        # Calculate source quality metrics
        source_credibilities = [a.source_credibility for a in articles]
        avg_credibility = sum(source_credibilities) / len(source_credibilities) if source_credibilities else 0.5

        # Build trend object
        trend = MajorTrend(
            trend_title=data.get("trend_title", "Unknown Trend"),
            trend_slug=self._slugify(data.get("trend_title", "")),
            trend_summary=data.get("trend_summary", ""),
            trend_type=trend_type,
            severity=severity,
            cluster_id=cluster.id,
            source_articles=[a.id for a in articles],
            article_count=len(articles),
            source_diversity_score=cluster.source_diversity_score,
            avg_source_credibility=avg_credibility,
            agreement_score=cluster.coherence_score,
            key_entities=data.get("key_entities", [])[:10],
            key_keywords=cluster.common_keywords[:10],
            key_facts=data.get("key_facts", [])[:5],
            key_numbers=self._parse_numbers(data.get("key_numbers", [])),
            geography=GeoLocation(
                country="India",
                state=data.get("affected_regions", [None])[0] if data.get("affected_regions") else None
            ),
            is_national=data.get("is_national", True),
            affected_regions=data.get("affected_regions", []),
            first_reported=cluster.earliest_article,
            last_updated=cluster.latest_article,
            lifecycle_stage=data.get("lifecycle_stage", "emerging"),
            primary_sectors=primary_sectors,
            secondary_sectors=secondary_sectors,
            confidence=confidence,
            llm_model_used=self.llm.settings.ollama_model if self.llm.settings.use_ollama else "gemini"
        )

        return trend

    def _parse_sectors(self, sector_names: List[str]) -> List[Sector]:
        """Parse sector names to Sector enums using the shared keyword map."""
        return _match_keywords(sector_names, SECTOR_KEYWORD_MAP, limit=3)

    def _parse_numbers(self, numbers: List) -> List[Dict[str, Any]]:
        """Parse key numbers into structured format."""
        return [
            num if isinstance(num, dict) else {"value": num, "context": ""}
            for num in numbers[:5]
            if isinstance(num, (dict, str))
        ]

    def _compute_confidence(
        self,
        cluster: ArticleCluster,
        articles: List[NewsArticle],
        data: Dict[str, Any]
    ) -> ConfidenceScore:
        """Compute confidence score based on multiple factors."""
        factors = []
        score = 0.5  # Base score

        # Factor 1: Source diversity (more sources = higher confidence)
        if cluster.unique_sources >= 3:
            score += 0.15
            factors.append(f"{cluster.unique_sources} unique sources")
        elif cluster.unique_sources >= 2:
            score += 0.1
            factors.append(f"{cluster.unique_sources} sources")

        # Factor 2: Source credibility
        avg_cred = sum(a.source_credibility for a in articles) / len(articles) if articles else 0
        if avg_cred >= 0.9:
            score += 0.15
            factors.append("High credibility sources")
        elif avg_cred >= 0.8:
            score += 0.1
            factors.append("Good credibility sources")

        # Factor 3: Cluster coherence
        if cluster.coherence_score >= 0.8:
            score += 0.1
            factors.append("High cluster coherence")

        # Factor 4: Article count
        if cluster.article_count >= 5:
            score += 0.1
            factors.append(f"{cluster.article_count} articles covering story")

        # Cap at 1.0
        score = min(score, 1.0)

        return ConfidenceScore(score=score, factors=factors)

    def _slugify(self, text: str) -> str:
        """Convert text to URL-friendly slug."""
        if not text:
            return ""
        slug = text.lower()
        slug = re.sub(r'[^a-z0-9\s-]', '', slug)
        slug = re.sub(r'[\s_-]+', '-', slug)
        slug = slug.strip('-')
        return slug[:50]

    async def analyze_sector_impact(
        self,
        trend: MajorTrend,
        sector: Sector
    ) -> Optional[SectorImpact]:
        """
        Generate detailed sector impact analysis for a trend.

        Args:
            trend: MajorTrend object
            sector: Sector to analyze

        Returns:
            SectorImpact object
        """
        prompt = f"""Analyze how this market trend impacts the {sector.value} sector in India.

TREND: {trend.trend_title}
SUMMARY: {trend.trend_summary}
TYPE: {trend.trend_type.value}
SEVERITY: {trend.severity.value}

Respond with JSON:
{{
    "impact_type": "positive, negative, mixed, neutral, or disruptive",
    "impact_severity": "critical, high, medium, low",
    "relevance_score": 0.0 to 1.0,
    "impact_summary": "2-3 sentences on overall impact",
    "direct_effects": ["List of 3-5 direct effects on this sector"],
    "indirect_effects": ["List of 2-3 second-order effects"],
    "challenges": ["List of challenges companies in this sector will face"],
    "opportunities": ["List of opportunities for companies in this sector"],
    "pain_points": ["Specific business pain points this creates"],
    "urgent_needs": ["What companies in this sector need NOW"],
    "target_roles": ["Job titles of decision makers who care about this"],
    "recommended_services": ["CMI services that would help: Procurement Intelligence, Market Intelligence, Competitive Intelligence, Market Monitoring, Industry Analysis, Technology Research, Cross Border Expansion, Consumer Insights, Consulting and Advisory Services"],
    "pitch_angle": "One sentence value proposition for CMI"
}}"""

        try:
            response = await self.llm.generate_json(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT
            )

            if "error" in response:
                return None

            impact_type = _parse_enum_value(response.get("impact_type", "neutral"), ImpactType, ImpactType.NEUTRAL)
            impact_severity = _parse_enum_value(response.get("impact_severity", "medium"), Severity, Severity.MEDIUM)
            recommended_services = _match_keywords(response.get("recommended_services", []), SERVICE_KEYWORD_MAP)

            return SectorImpact(
                trend_id=trend.id,
                trend_title=trend.trend_title,
                sector=sector,
                impact_type=impact_type,
                impact_severity=impact_severity,
                relevance_score=float(response.get("relevance_score", 0.5)),
                impact_summary=response.get("impact_summary", ""),
                direct_effects=response.get("direct_effects", []),
                indirect_effects=response.get("indirect_effects", []),
                challenges=response.get("challenges", []),
                opportunities=response.get("opportunities", []),
                pain_points=response.get("pain_points", []),
                urgent_needs=response.get("urgent_needs", []),
                target_roles=response.get("target_roles", []),
                recommended_services=recommended_services,
                pitch_angle=response.get("pitch_angle", "")
            )

        except Exception as e:
            logger.error(f"Failed to analyze sector impact: {e}")
            return None


# Convenience function
async def synthesize_trends(
    clusters: List[ArticleCluster],
    articles: List[NewsArticle],
    mock_mode: bool = False
) -> List[MajorTrend]:
    """Quick trend synthesis function."""
    synthesizer = TrendSynthesizer(mock_mode=mock_mode)
    return await synthesizer.synthesize_all(clusters, articles)
