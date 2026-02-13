"""
Trend Detection Agent.
Detects Indian market trends using RSS feeds and Tavily search.
"""

import logging
import hashlib
from datetime import datetime
from typing import List

from ..schemas import TrendData, Severity, AgentState
from ..tools.rss_tool import RSSTool
from ..tools.tavily_tool import TavilyTool
from ..tools.llm_service import LLMService
from ..config import get_settings

logger = logging.getLogger(__name__)


class TrendAgent:
    """
    Agent responsible for detecting and enriching market trends.
    
    Pipeline:
    1. Fetch headlines from Google News RSS
    2. Enrich with Tavily search for context
    3. Use LLM to analyze and structure trend data
    """
    
    def __init__(self, mock_mode: bool = False):
        """Initialize trend agent with tools."""
        self.settings = get_settings()
        self.mock_mode = mock_mode or self.settings.mock_mode
        self.rss_tool = RSSTool(mock_mode=self.mock_mode)
        self.tavily_tool = TavilyTool(mock_mode=self.mock_mode)
        self.llm_service = LLMService(mock_mode=self.mock_mode)
    
    async def detect_trends(self, state: AgentState) -> AgentState:
        """
        Main entry point for trend detection.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with detected trends
        """
        logger.info("🔍 Starting trend detection...")
        
        try:
            # Step 1: Fetch news articles from all sources
            articles = await self.rss_tool.fetch_all_sources(
                max_per_source=5
            )
            # Convert NewsArticle objects to dicts for _process_headline
            rss_items = [
                {"title": a.title, "summary": a.summary, "link": a.url, "source": a.source_name}
                for a in articles
            ]
            logger.info(f"📰 Fetched {len(rss_items)} RSS items")
            
            if not rss_items:
                state.errors.append("No RSS items fetched")
                return state
            
            # Step 2: Process each headline into a structured trend
            trends = []
            max_trends = self.settings.max_trends
            
            for item in rss_items[:max_trends + 2]:  # Process a few extra
                try:
                    trend = await self._process_headline(item)
                    if trend:
                        trends.append(trend)
                        logger.info(f"✅ Processed trend: {trend.trend_title[:50]}...")
                except Exception as e:
                    logger.warning(f"Failed to process headline: {e}")
                
                # Stop if we have enough trends
                if len(trends) >= max_trends:
                    break
            
            state.trends = trends
            state.current_step = "trends_detected"
            logger.info(f"🎯 Detected {len(trends)} trends")
            
        except Exception as e:
            error_msg = f"Trend detection failed: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    async def _process_headline(self, rss_item: dict) -> TrendData:
        """
        Process a single RSS headline into a structured trend.
        
        Args:
            rss_item: RSS feed item with title, summary, link
            
        Returns:
            Structured TrendData object
        """
        title = rss_item.get("title", "")
        summary = rss_item.get("summary", "")
        link = rss_item.get("link", "")
        source = rss_item.get("source", "News")
        
        # Step 1: Enrich with Tavily search (skip in mock mode for speed)
        if not self.mock_mode:
            enriched = await self.tavily_tool.enrich_trend(title, summary)
        else:
            enriched = {"enriched_context": summary, "sources": []}
        
        # Step 2: Use LLM to analyze and structure
        analysis = await self._analyze_trend(title, summary, enriched)
        
        # Generate unique ID based on title
        trend_id = hashlib.md5(title.encode()).hexdigest()[:12]
        
        # Determine severity
        severity = self._determine_severity(analysis)
        
        # Use original title if LLM didn't provide a good one
        final_title = analysis.get("trend_title", title)
        if not final_title or final_title == "AI Adoption Surge in Indian Enterprises":
            final_title = title  # Use original RSS title
        
        final_summary = analysis.get("summary", summary)
        if not final_summary:
            final_summary = summary or enriched.get("enriched_context", "")
        
        return TrendData(
            id=trend_id,
            trend_title=final_title,
            summary=final_summary,
            severity=severity,
            industries_affected=analysis.get("industries_affected", ["Technology", "Business"]),
            source_links=[link] + [s.get("url", "") for s in enriched.get("sources", [])],
            keywords=analysis.get("keywords", title.lower().split()[:5]),
            detected_at=datetime.utcnow()
        )
    
    async def _analyze_trend(
        self,
        title: str,
        summary: str,
        enriched: dict
    ) -> dict:
        """
        Use LLM to analyze a NEWS EVENT and extract structured data.
        
        Args:
            title: News headline
            summary: Brief summary
            enriched: Enriched data from Tavily
            
        Returns:
            Structured analysis dict
        """
        context = enriched.get("enriched_context", "")
        sources = enriched.get("sources", [])
        source_snippets = "\n".join([s.get("snippet", "") for s in sources[:2]])
        
        prompt = f"""Analyze this BREAKING Indian business news and identify sales opportunities.

NEWS HEADLINE: {title}

SUMMARY: {summary}

ADDITIONAL CONTEXT:
{context}

SOURCE SNIPPETS:
{source_snippets}

This is a SPECIFIC news event that just happened. Analyze it for B2B sales opportunities.

Provide your analysis as JSON with these fields:
- trend_title: The specific event/announcement (max 100 chars)
- summary: What happened and WHY it matters for businesses (max 300 chars)
- industries_affected: List of 3-5 specific Indian industries/sectors DIRECTLY affected by this news
- keywords: List of 5-7 keywords including company names, regulations, or technologies mentioned
- trend_type: One of ["regulation", "funding", "acquisition", "layoffs", "expansion", "policy", "partnership", "ipo", "restructuring"]
- urgency: Why companies need to act NOW on this news

Focus on ACTIONABLE intelligence. Who needs to know about this news TODAY?"""

        system_prompt = """You are a B2B sales intelligence analyst for the Indian market.
Your job is to analyze BREAKING NEWS and identify which companies need to hear about this NOW.
Focus on specific, actionable insights - not generic trends.
Always respond with valid JSON only."""

        try:
            result = await self.llm_service.generate_json(
                prompt=prompt,
                system_prompt=system_prompt
            )
            return result
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            # Return basic structure
            return {
                "trend_title": title,
                "summary": summary,
                "industries_affected": ["Technology", "Manufacturing", "Services"],
                "keywords": title.lower().split()[:5]
            }
    
    def _determine_severity(self, analysis: dict) -> Severity:
        """Determine trend severity based on analysis."""
        trend_type = analysis.get("trend_type", "").lower()
        
        high_severity_types = ["disruption", "regulation", "funding"]
        medium_severity_types = ["growth", "technology"]
        
        if trend_type in high_severity_types:
            return Severity.HIGH
        elif trend_type in medium_severity_types:
            return Severity.MEDIUM
        else:
            return Severity.LOW


async def run_trend_agent(state: AgentState) -> AgentState:
    """Wrapper function for LangGraph."""
    agent = TrendAgent()
    return await agent.detect_trends(state)
