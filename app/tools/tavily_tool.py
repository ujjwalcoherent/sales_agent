"""
Tavily Search Tool for real-time web search.
Used for trend validation, company research, and contact discovery.
"""

import logging
from typing import List, Dict, Optional, Any
import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)


class TavilyTool:
    """
    Tavily API wrapper for web search.
    Used for trend enrichment, company finding, and contact research.
    """
    
    TAVILY_API_URL = "https://api.tavily.com/search"
    
    def __init__(self, mock_mode: bool = False):
        """Initialize Tavily tool."""
        self.settings = get_settings()
        self.mock_mode = mock_mode or self.settings.mock_mode
        self.api_key = self.settings.tavily_api_key
    
    async def search(
        self,
        query: str,
        search_depth: str = "basic",
        max_results: int = 5,
        include_answer: bool = True,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform a Tavily search.
        
        Args:
            query: Search query
            search_depth: "basic" or "advanced"
            max_results: Maximum number of results
            include_answer: Include AI-generated answer
            include_domains: Only search these domains
            exclude_domains: Exclude these domains
            
        Returns:
            Search results with answer, results list, and metadata
        """
        if self.mock_mode:
            return self._get_mock_search_result(query)
        
        if not self.api_key:
            logger.error("Tavily API key not configured")
            return {"error": "API key not configured", "results": []}
        
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_answer": include_answer,
        }
        
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.TAVILY_API_URL, json=payload)
                response.raise_for_status()
                
                data = response.json()
                logger.info(f"Tavily search for '{query[:50]}...' returned {len(data.get('results', []))} results")
                return data
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Tavily HTTP error: {e.response.status_code}")
            return {"error": f"HTTP {e.response.status_code}", "results": []}
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return {"error": str(e), "results": []}
    
    async def enrich_trend(self, trend_title: str, trend_summary: str) -> Dict[str, Any]:
        """
        Enrich a trend with additional context from web search.
        
        Args:
            trend_title: The trend headline
            trend_summary: Brief summary of the trend
            
        Returns:
            Enriched data with context, industries, and analysis
        """
        query = f"{trend_title} India market impact analysis 2025 2026"
        
        result = await self.search(
            query=query,
            search_depth="advanced" if not self.mock_mode else "basic",
            max_results=5,
            include_answer=True
        )
        
        return {
            "trend_title": trend_title,
            "original_summary": trend_summary,
            "enriched_context": result.get("answer", ""),
            "sources": [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", "")[:300]
                }
                for r in result.get("results", [])[:3]
            ]
        }
    
    async def find_companies(
        self,
        sector: str,
        company_size: str = "mid",
        limit: int = 5
    ) -> List[Dict]:
        """
        Find companies in a specific sector.
        
        Args:
            sector: Industry/sector to search
            company_size: "startup", "mid", or "enterprise"
            limit: Maximum number of companies
            
        Returns:
            List of company information
        """
        size_keywords = {
            "startup": "startups early-stage",
            "mid": "mid-sized growing companies",
            "enterprise": "large enterprise corporations"
        }
        
        size_term = size_keywords.get(company_size, "companies")
        query = f"top {size_term} {sector} India 2025 list companies"
        
        result = await self.search(
            query=query,
            max_results=limit + 2,
            include_answer=True
        )
        
        companies = []
        for item in result.get("results", [])[:limit]:
            companies.append({
                "source_title": item.get("title", ""),
                "source_url": item.get("url", ""),
                "snippet": item.get("content", ""),
                "sector": sector,
                "company_size": company_size
            })
        
        return companies
    
    async def find_contact(
        self,
        company_name: str,
        role: str,
        company_domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find contact information for a person at a company.
        
        Args:
            company_name: Name of the company
            role: Target role (e.g., "CTO", "CEO")
            company_domain: Optional company domain for better results
            
        Returns:
            Contact information found
        """
        query = f"{role} {company_name} India LinkedIn"
        
        result = await self.search(
            query=query,
            max_results=3,
            include_answer=True
        )
        
        return {
            "company_name": company_name,
            "target_role": role,
            "search_answer": result.get("answer", ""),
            "results": [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", "")[:200]
                }
                for r in result.get("results", [])
            ]
        }
    
    async def find_company_domain(self, company_name: str) -> Optional[str]:
        """
        Find the official domain for a company.
        
        Args:
            company_name: Name of the company
            
        Returns:
            Company domain if found
        """
        query = f"{company_name} India official website"
        
        result = await self.search(
            query=query,
            max_results=3,
            include_answer=False
        )
        
        # Look for company website in results
        from .domain_utils import extract_clean_domain, is_valid_company_domain
        
        for item in result.get("results", []):
            url = item.get("url", "")
            domain = extract_clean_domain(url)
            if domain and is_valid_company_domain(domain):
                # Check if domain seems related to company name
                company_lower = company_name.lower().replace(" ", "")
                if any(part in domain.lower() for part in company_lower.split()[:2]):
                    return domain
        
        return None
    
    def _get_mock_search_result(self, query: str) -> Dict[str, Any]:
        """Return mock search results with INTENT SIGNALS for testing."""
        query_lower = query.lower()
        
        # Companies with intent signals - struggling/expanding/affected
        if "struggling" in query_lower or "challenges" in query_lower or "facing issues" in query_lower:
            return {
                "answer": "Several mid-sized Indian companies are facing challenges due to market changes. Notable examples include regional suppliers restructuring operations.",
                "results": [
                    {
                        "title": "Petrosol Energy facing margin pressure amid oil price volatility",
                        "url": "https://economictimes.com/petrosol-margins",
                        "content": "Petrosol Energy (petrosol.in), a mid-sized oil equipment supplier with 180 employees, is struggling with rising input costs. The company is seeking strategic partners and considering restructuring its supply chain. CEO Rajesh Mehta stated they need market intelligence to navigate the volatility."
                    },
                    {
                        "title": "Gujarat Oilfield Services restructures amid market changes",
                        "url": "https://businessstandard.com/gos-restructure",
                        "content": "Gujarat Oilfield Services Ltd (gosindia.com), with 220 employees, announced restructuring plans. The company is facing challenges in procurement and seeking cost optimization strategies. They recently hired consultants for supply chain assessment."
                    },
                    {
                        "title": "Regional fuel distributors seek partnerships",
                        "url": "https://livemint.com/fuel-distributors",
                        "content": "Bharat Petroleum Distributors Association reports that mid-sized distributors like Shree Fuel Agencies (shreefuel.in, 85 employees) and Metro Petroleum (metropetro.co.in, 120 employees) are facing margin pressures and actively seeking consulting support for pricing strategy."
                    }
                ]
            }
        elif "expanding" in query_lower or "growing" in query_lower or "entering market" in query_lower:
            return {
                "answer": "Several mid-sized Indian companies are expanding operations and entering new markets.",
                "results": [
                    {
                        "title": "Oilmax Energy expands into renewable sector",
                        "url": "https://economictimes.com/oilmax-expansion",
                        "content": "Oilmax Energy Services (oilmaxenergy.com), a 150-employee oil equipment company, announced expansion into renewable energy equipment. The company is seeking market research on solar equipment demand and competitive landscape analysis."
                    },
                    {
                        "title": "Praj Industries eyes international expansion",
                        "url": "https://businesstoday.in/praj-global",
                        "content": "Praj Industries (praj.net), with 250 employees in their core division, is planning cross-border expansion. They recently commissioned a market entry feasibility study for Southeast Asian markets."
                    }
                ]
            }
        elif "oil" in query_lower or "energy" in query_lower or "fuel" in query_lower:
            return {
                "answer": "The Indian oil and energy sector has several mid-sized companies actively responding to market changes.",
                "results": [
                    {
                        "title": "Top mid-size oil equipment companies respond to price surge",
                        "url": "https://oilasia.com/india-suppliers",
                        "content": "Mid-sized oil equipment suppliers like Welspun Corp (welspuncorp.com, 280 employees in equipment division), Jindal SAW (jindalsaw.com, mid-size unit), and Deep Industries (deepind.com, 190 employees) are reassessing their strategies. Several are seeking procurement intelligence and supplier benchmarking services."
                    },
                    {
                        "title": "Regional refineries face strategic decisions",
                        "url": "https://energyworld.com/refineries-india",
                        "content": "Small-scale refineries like Numaligarh Refinery's mid-size suppliers, Mangalore Chemicals subsidiary (MCFL, mcfl.co.in, 140 employees), and Chennai Petroleum ancillary units are evaluating market positions. Industry experts note these companies need competitive intelligence to navigate the changing landscape."
                    }
                ]
            }
        elif "cto" in query_lower or "ceo" in query_lower or "linkedin" in query_lower:
            return {
                "answer": "Key decision makers at mid-sized Indian companies.",
                "results": [
                    {
                        "title": "Petrosol Energy Leadership - LinkedIn",
                        "url": "https://linkedin.com/in/rajesh-mehta-petrosol",
                        "content": "Rajesh Mehta, CEO at Petrosol Energy. 15+ years in oil and gas equipment industry. Previously VP at L&T Energy."
                    }
                ]
            }
        else:
            return {
                "answer": f"Analysis of '{query[:50]}' shows mid-sized Indian companies are actively responding to market changes.",
                "results": [
                    {
                        "title": "Mid-sized companies adapting to market trends",
                        "url": "https://economictimes.com/midsize-trends",
                        "content": "Indian mid-sized companies across sectors are seeking strategic guidance. Companies like TechnoServe India (technoserve.in, 180 employees), Prism Johnson's equipment division (prismjohnson.in, 200 employees), and regional manufacturing firms are investing in market research and consulting services."
                    }
                ]
            }
