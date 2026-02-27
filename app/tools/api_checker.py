"""
API Availability Checker — checks configuration, not connectivity.

Provides a quick status of which APIs are configured.
Doesn't make network calls (too slow for Streamlit UI).
"""

import logging
from typing import Dict, List

from ..config import get_settings

logger = logging.getLogger(__name__)


class APIChecker:
    """Quick API configuration checker (no network calls)."""

    def __init__(self):
        self.settings = get_settings()

    async def get_all_available(self) -> Dict[str, bool]:
        """Check which APIs are configured (not connectivity)."""
        return {
            "llm_NVIDIA": bool(self.settings.nvidia_api_key),
            "llm_Ollama": self.settings.use_ollama,
            "llm_OpenRouter": bool(self.settings.openrouter_api_key),
            "llm_GeminiDirect": bool(
                self.settings.gcp_project_id 
                or self.settings.vertex_express_api_key 
                or self.settings.gemini_api_key
            ),
            "llm_Groq": bool(self.settings.groq_api_key),
            "embedding_NVIDIA": bool(self.settings.nvidia_api_key and "nvidia" in self.settings.embedding_model.lower()),
            "embedding_HF": bool(self.settings.huggingface_api_key),
            "search_Tavily": bool(self.settings.tavily_api_keys),
            "news_NewsAPI": bool(self.settings.newsapi_org_key),
            "news_RapidAPI": bool(self.settings.rapidapi_key),
            "news_GNews": bool(self.settings.gnews_api_key),
            "news_MediaStack": bool(self.settings.mediastack_api_key),
            "news_TheNewsAPI": bool(self.settings.thenewsapi_key),
            "email_Apollo": bool(self.settings.apollo_api_key),
            "email_Hunter": bool(self.settings.hunter_api_key),
        }

    async def get_critical_issues(self) -> List[str]:
        """Get list of critical issues preventing app operation."""
        issues = []
        available = await self.get_all_available()
        
        # At least one LLM must be available
        llm_available = any(k.startswith("llm_") and v for k, v in available.items())
        if not llm_available:
            issues.append("❌ No LLM providers configured (need NVIDIA, Ollama, OpenRouter, Gemini, or Groq API keys)")
        
        # At least one embedding method must be available
        embed_available = any(k.startswith("embedding_") and v for k, v in available.items())
        if not embed_available:
            issues.append("❌ No embedding providers configured (need NVIDIA key or HuggingFace key)")
        
        # At least one search/news API should be available (not critical but helpful)
        search_available = any(k.startswith("search_") or k.startswith("news_") and v for k, v in available.items())
        if not search_available:
            issues.append("⚠️ No search/news APIs configured (will use RSS feeds only)")
        
        return issues

    async def get_status_summary(self) -> str:
        """Get a human-readable status summary."""
        available = await self.get_all_available()
        issues = await self.get_critical_issues()
        
        lines = ["📊 API Configuration Status:"]
        lines.append("")
        
        # Group by category
        categories = {
            "LLM Providers": [k for k in available if k.startswith("llm_")],
            "Embedding": [k for k in available if k.startswith("embedding_")],
            "Search/News": [k for k in available if k.startswith("search_") or k.startswith("news_")],
            "Email Finders": [k for k in available if k.startswith("email_")],
        }
        
        for category, keys in categories.items():
            if keys:
                available_count = sum(1 for k in keys if available.get(k))
                lines.append(f"  {category}: {available_count}/{len(keys)} configured")
                for key in keys:
                    status = "✅" if available.get(key) else "❌"
                    name = key.replace("llm_", "").replace("embedding_", "").replace("search_", "").replace("news_", "").replace("email_", "")
                    lines.append(f"    {status} {name}")
        
        if issues:
            lines.append("")
            lines.append("⚠️ Configuration Issues:")
            for issue in issues:
                lines.append(f"  {issue}")
        else:
            lines.append("")
            lines.append("✅ All critical APIs configured!")
        
        return "\n".join(lines)
