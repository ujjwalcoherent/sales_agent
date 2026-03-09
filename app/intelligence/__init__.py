"""
intelligence/ — Unified Sales Intelligence Engine

Replaces app/trends/, app/clustering/, and app/news/ with a single,
coherent pipeline built on the "math before LLM" principle.

Entry point: intelligence.pipeline.execute()

22 specialized agents, 3 discovery paths:
  - COMPANY_FIRST: deep intel on specific companies
  - INDUSTRY_FIRST: discover trends + companies in an industry
  - REPORT_DRIVEN: corroborate claims in analyst reports

Architecture references:
  - ReAct (Yao et al. 2022)     — TAO loop at every agent
  - Reflexion (Shinn et al. 2023) — verbal retry with critique
  - CRITIC (Gou et al. 2023)    — math tool verifies LLM output
  - MetaGPT (Hong et al. 2023)  — typed I/O at every boundary
  - Blackboard (Erman et al. 1980) — Signal Bus shared state
"""

from app.intelligence.models import (
    DiscoveryScope,
    DiscoveryMode,
    IntelligenceResult,
)

__all__ = ["DiscoveryScope", "DiscoveryMode", "IntelligenceResult"]
