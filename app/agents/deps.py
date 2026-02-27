"""
Shared dependency container for pydantic-ai agents.

Extends the lazy-initialized tool pattern from PipelineDeps.
Every agent receives this via RunContext[AgentDeps] and accesses
tools through properties (lazy init on first use).

This replaces PipelineDeps for the multi-agent architecture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentDeps:
    """Shared dependencies for all pydantic-ai agents.

    Passed as deps_type to each Agent. Tools access via ctx.deps.
    Lazy initialization means tools are only created when first used.
    """

    def __hash__(self):
        # pydantic-ai and LangGraph sometimes hash the deps container
        # (e.g. as a cache key). Use object identity — each run gets a fresh deps.
        return id(self)

    mock_mode: bool = False
    log_callback: Optional[object] = field(default=None, repr=False)
    disabled_providers: List[str] = field(default_factory=list, repr=False)

    # Lazy-initialized tools
    _llm_service: Optional[object] = field(default=None, repr=False)
    _llm_lite_service: Optional[object] = field(default=None, repr=False)
    _tavily_tool: Optional[object] = field(default=None, repr=False)
    _rss_tool: Optional[object] = field(default=None, repr=False)
    _apollo_tool: Optional[object] = field(default=None, repr=False)
    _hunter_tool: Optional[object] = field(default=None, repr=False)
    _embedding_tool: Optional[object] = field(default=None, repr=False)
    _article_cache: Optional[object] = field(default=None, repr=False)
    _source_bandit: Optional[object] = field(default=None, repr=False)
    _company_bandit: Optional[object] = field(default=None, repr=False)

    # Mutable working data (set by agents during execution)
    _articles: List[Any] = field(default_factory=list, repr=False)
    _embeddings: List[Any] = field(default_factory=list, repr=False)
    _event_distribution: Dict[str, int] = field(default_factory=dict, repr=False)
    _trend_tree: Optional[object] = field(default=None, repr=False)
    _pipeline: Optional[object] = field(default=None, repr=False)
    _params_used: Dict[str, float] = field(default_factory=dict, repr=False)
    _trend_data: List[Any] = field(default_factory=list, repr=False)
    _impacts: List[Any] = field(default_factory=list, repr=False)
    _viable_impacts: List[Any] = field(default_factory=list, repr=False)
    _companies: List[Any] = field(default_factory=list, repr=False)
    _contacts: List[Any] = field(default_factory=list, repr=False)
    _outreach: List[Any] = field(default_factory=list, repr=False)
    _person_profiles: List[Any] = field(default_factory=list, repr=False)

    # Pipeline agents (causal chain + lead crystallization)
    _search_manager: Optional[object] = field(default=None, repr=False)
    _causal_results: List[Any] = field(default_factory=list, repr=False)
    _lead_sheets: List[Any] = field(default_factory=list, repr=False)
    # Per-run quality metrics for autonomous weight/bandit learning
    _signals: List[Any] = field(default_factory=list, repr=False)

    # Run recorder — captures step snapshots for mock replay (real runs only)
    _recorder: Optional[object] = field(default=None, repr=False)
    _pipeline_t0: float = 0.0  # Pipeline start time (set by orchestrator)

    @classmethod
    def create(
        cls,
        mock_mode: bool = False,
        log_callback=None,
        run_id: str = "",
        disabled_providers: list[str] | None = None,
    ) -> AgentDeps:
        """Create deps with settings-aware mock_mode.

        When mock_mode=False and run_id is provided, a RunRecorder is
        automatically created to capture step snapshots for replay.
        """
        from app.config import get_settings
        settings = get_settings()
        effective_mock = mock_mode or settings.mock_mode

        recorder = None
        if not effective_mock and run_id:
            from app.tools.run_recorder import RunRecorder
            recorder = RunRecorder(run_id=run_id)

        return cls(
            mock_mode=effective_mock,
            log_callback=log_callback,
            disabled_providers=disabled_providers or [],
            _recorder=recorder,
        )

    def _log(self, msg: str, level: str = "info"):
        """Log to both logger and optional UI callback."""
        getattr(logger, level, logger.info)(msg)
        if self.log_callback:
            try:
                self.log_callback(msg, level)
            except Exception:
                pass

    # ── Tool properties (lazy init) ──────────────────────────────────

    @property
    def llm_service(self):
        if self._llm_service is None:
            from app.tools.llm_service import LLMService
            self._llm_service = LLMService(
                mock_mode=self.mock_mode,
                disabled_providers=self.disabled_providers,
            )
        return self._llm_service

    @property
    def llm_lite_service(self):
        if self._llm_lite_service is None:
            from app.tools.llm_service import LLMService
            self._llm_lite_service = LLMService(
                mock_mode=self.mock_mode,
                lite=True,
                disabled_providers=self.disabled_providers,
            )
        return self._llm_lite_service

    @property
    def tavily_tool(self):
        if self._tavily_tool is None:
            from app.tools.tavily_tool import TavilyTool
            self._tavily_tool = TavilyTool(mock_mode=self.mock_mode)
        return self._tavily_tool

    @property
    def rss_tool(self):
        if self._rss_tool is None:
            from app.tools.rss_tool import RSSTool
            self._rss_tool = RSSTool(mock_mode=self.mock_mode)
        return self._rss_tool

    @property
    def apollo_tool(self):
        if self._apollo_tool is None:
            from app.tools.apollo_tool import ApolloTool
            self._apollo_tool = ApolloTool(mock_mode=self.mock_mode)
        return self._apollo_tool

    @property
    def hunter_tool(self):
        if self._hunter_tool is None:
            from app.tools.hunter_tool import HunterTool
            self._hunter_tool = HunterTool(mock_mode=self.mock_mode)
        return self._hunter_tool

    @property
    def embedding_tool(self):
        if self._embedding_tool is None:
            from app.tools.embeddings import EmbeddingTool
            self._embedding_tool = EmbeddingTool()
        return self._embedding_tool

    @property
    def article_cache(self):
        if self._article_cache is None:
            from app.tools.article_cache import ArticleCache
            self._article_cache = ArticleCache()
        return self._article_cache

    @property
    def source_bandit(self):
        if self._source_bandit is None:
            from app.learning.source_bandit import SourceBandit
            self._source_bandit = SourceBandit()
        return self._source_bandit

    @property
    def company_bandit(self):
        if self._company_bandit is None:
            from app.agents.company_relevance_bandit import CompanyRelevanceBandit
            self._company_bandit = CompanyRelevanceBandit()
        return self._company_bandit

    @property
    def search_manager(self):
        """Unified search manager — BM25 → SearXNG → DDG fallback chain."""
        if self._search_manager is None:
            try:
                from app.search.manager import SearchManager
                from app.config import get_settings
                settings = get_settings()
                self._search_manager = SearchManager(
                    searxng_url=getattr(settings, "searxng_url", "http://localhost:8888"),
                    searxng_enabled=getattr(settings, "searxng_enabled", False),
                )
            except Exception as e:
                logger.warning(f"SearchManager init failed: {e}")
        return self._search_manager

    @property
    def recorder(self):
        """RunRecorder instance (None in mock mode)."""
        return self._recorder

    def get_model(self):
        """Get the pydantic-ai model from ProviderManager.

        Returns the FallbackModel with cooldown-aware provider chain.
        Used by agents to override their placeholder model at runtime.
        """
        from app.tools.provider_manager import ProviderManager
        pm = ProviderManager(mock_mode=self.mock_mode)
        return pm.get_model()
