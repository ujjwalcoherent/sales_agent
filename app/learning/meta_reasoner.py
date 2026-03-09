"""
MetaReasoner — Chain-of-thought reasoning layer for pipeline self-improvement.

An LLM-powered brain that THINKS at each pipeline step:
  - Questions its own decisions ("Is this trend actually novel or recycled?")
  - Evaluates quality with structured reasoning ("WHY is this synthesis vague?")
  - Projects outcomes before acting ("This trend pattern historically produces X leads")
  - Generates improvement hypotheses ("Next run: prioritize sources with entity richness >4")
  - Runs a full retrospective after each pipeline run

Uses lite LLM (GPT-4.1-nano, ~$0.10/1M tokens) to keep costs negligible
(~$0.02-0.05 per full pipeline run for all reasoning steps).

The reasoning traces are:
  1. Logged to data/reasoning_traces.jsonl for audit/debugging
  2. Fed into the signal bus for cross-loop learning
  3. Used to adjust the NEXT run's parameters dynamically

This is the system's "inner monologue" — it doesn't just learn from numbers,
it THINKS about what went well, what went wrong, and what to do differently.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TRACES_PATH = Path("./data/reasoning_traces.jsonl")
_HYPOTHESES_PATH = Path("./data/improvement_hypotheses.json")


@dataclass
class ReasoningTrace:
    """Structured chain-of-thought reasoning output from a single checkpoint."""
    step: str                                    # Pipeline step name
    run_id: str = ""
    timestamp: str = ""
    # Chain of thought
    observations: List[str] = field(default_factory=list)  # What the reasoner noticed
    concerns: List[str] = field(default_factory=list)       # What's worrying
    strengths: List[str] = field(default_factory=list)      # What's working well
    hypotheses: List[str] = field(default_factory=list)     # Improvement ideas
    # Structured scores
    quality_score: float = 0.0                              # 0-1 overall quality
    confidence: float = 0.5                                 # How confident in this assessment
    # Strategy adjustments for downstream steps
    strategy_adjustments: Dict[str, Any] = field(default_factory=dict)
    # Projections
    projected_outcome: str = ""                             # What the reasoner expects to happen next
    # Cost tracking
    reasoning_time_ms: float = 0.0
    tokens_used: int = 0


@dataclass
class Retrospective:
    """Full post-run chain-of-thought self-critique."""
    run_id: str = ""
    timestamp: str = ""
    # Overall assessment
    run_grade: str = ""                          # A/B/C/D/F
    summary: str = ""                            # One-paragraph assessment
    # What went well / poorly
    successes: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    surprises: List[str] = field(default_factory=list)       # Unexpected outcomes
    # Actionable improvements
    improvement_plan: List[Dict[str, str]] = field(default_factory=list)
    # e.g. [{"target": "source_bandit", "action": "increase decay for stale sources", "priority": "high"}]
    # Comparison to previous run
    vs_previous: str = ""                        # "Better/Worse/Similar"
    regression_risks: List[str] = field(default_factory=list)


class MetaReasoner:
    """Chain-of-thought reasoning engine for pipeline self-improvement.

    Sits between pipeline steps and produces structured reasoning traces.
    Uses lite LLM to keep costs at ~$0.02-0.05 per full pipeline run.
    """

    def __init__(self, llm_service=None, enabled: bool = True):
        """Initialize with an LLM service for reasoning.

        Args:
            llm_service: LLMService instance (lite preferred). If None, reasoning
                         is disabled gracefully — pipeline still works.
            enabled: Kill switch for reasoning (e.g. during mock runs).
        """
        self._llm = llm_service
        self._enabled = enabled and llm_service is not None
        self._traces: List[ReasoningTrace] = []
        self._run_id = ""
        self._previous_hypotheses = self._load_hypotheses()

    # ──────────────────────────────────────────────────────────────────
    # Reasoning checkpoints (called between pipeline steps)
    # ──────────────────────────────────────────────────────────────────

    async def reason_about_sources(
        self,
        article_count: int,
        source_stats: Dict[str, Dict[str, Any]],
        bandit_top_sources: List[str],
        hours_window: int,
    ) -> ReasoningTrace:
        """After source_intel: reason about article quality and source strategy.

        Questions the system asks itself:
        - Are we getting enough article diversity?
        - Are top sources still performing or coasting on prior?
        - Should we adjust the lookback window?
        - Are there blind spots in our source coverage?
        """
        trace = ReasoningTrace(step="source_intel", run_id=self._run_id)
        if not self._enabled:
            return trace

        t0 = time.time()

        # Build context for the reasoner
        source_summary = []
        for sid, stats in list(source_stats.items())[:15]:
            source_summary.append(
                f"  {sid}: {stats.get('articles', 0)} articles, "
                f"noise_rate={stats.get('noise_rate', 0):.0%}, "
                f"avg_quality={stats.get('avg_quality', 0):.2f}"
            )

        previous_advice = ""
        for h in self._previous_hypotheses:
            if h.get("target") in ("source_intel", "source_bandit", "sources"):
                previous_advice += f"\n- Previous suggestion: {h.get('action', '')}"

        prompt = f"""You are the meta-reasoning layer of an AI sales intelligence pipeline.
Your job is to THINK about the source collection step that just completed.

DATA COLLECTED:
- {article_count} articles from {len(source_stats)} sources (last {hours_window} hours)
- Top sources by bandit quality: {', '.join(bandit_top_sources[:5])}
- Source breakdown:
{chr(10).join(source_summary[:10])}
{previous_advice}

THINK step by step:
1. OBSERVE: What patterns do you see in the source data?
2. CONCERN: What worries you about this data collection?
3. STRENGTH: What's working well?
4. HYPOTHESIZE: What specific changes would improve next run's collection?
5. PROJECT: Based on this data quality, what lead quality do you expect downstream?

Respond as JSON:
{{
  "observations": ["..."],
  "concerns": ["..."],
  "strengths": ["..."],
  "hypotheses": [{{"target": "...", "action": "...", "priority": "high/medium/low"}}],
  "quality_score": 0.0-1.0,
  "confidence": 0.0-1.0,
  "projected_outcome": "one sentence prediction",
  "strategy_adjustments": {{}}
}}"""

        try:
            result = await self._llm.generate_json(prompt=prompt)
            if result and "error" not in result:
                trace.observations = result.get("observations", [])[:5]
                trace.concerns = result.get("concerns", [])[:5]
                trace.strengths = result.get("strengths", [])[:3]
                trace.hypotheses = [
                    h if isinstance(h, str) else h.get("action", str(h))
                    for h in result.get("hypotheses", [])[:5]
                ]
                trace.quality_score = min(1.0, max(0.0, float(result.get("quality_score", 0.5))))
                trace.confidence = min(1.0, max(0.0, float(result.get("confidence", 0.5))))
                trace.projected_outcome = result.get("projected_outcome", "")
                trace.strategy_adjustments = result.get("strategy_adjustments", {})
        except Exception as e:
            logger.debug(f"MetaReasoner source reasoning failed: {e}")
            trace.observations = [f"Reasoning failed: {e}"]

        trace.reasoning_time_ms = (time.time() - t0) * 1000
        trace.timestamp = datetime.now(timezone.utc).isoformat()
        self._traces.append(trace)
        self._persist_trace(trace)

        if trace.concerns:
            logger.info(f"MetaReasoner [sources]: concerns={trace.concerns[:2]}")
        if trace.hypotheses:
            logger.info(f"MetaReasoner [sources]: hypotheses={trace.hypotheses[:2]}")

        return trace

    async def reason_about_trends(
        self,
        trends: list,
        cluster_count: int,
        noise_rate: float,
        mean_oss: float,
        mean_coherence: float,
    ) -> ReasoningTrace:
        """After analysis: reason about trend quality and clustering decisions.

        Questions:
        - Are these trends genuinely novel or recycled from previous runs?
        - Is the clustering capturing real market signals or just noise?
        - Which trends are most actionable for sales?
        - Should we adjust clustering parameters?
        """
        trace = ReasoningTrace(step="analysis", run_id=self._run_id)
        if not self._enabled:
            return trace

        t0 = time.time()

        trend_summaries = []
        for t in trends[:12]:
            title = getattr(t, "trend_title", "") or getattr(t, "title", "")
            oss = getattr(t, "oss_score", 0.0)
            score = getattr(t, "trend_score", 0.0)
            industries = getattr(t, "industries_affected", [])
            trend_summaries.append(
                f"  - \"{title}\" (OSS={oss:.2f}, score={score:.2f}, industries={industries[:3]})"
            )

        prompt = f"""You are the meta-reasoning layer analyzing trend quality.

ANALYSIS RESULTS:
- {len(trends)} trends from {cluster_count} clusters
- Noise rate: {noise_rate:.0%} of articles were noise (not clustered)
- Mean OSS (specificity): {mean_oss:.3f} (>0.4 is good, <0.2 is garbage)
- Mean coherence: {mean_coherence:.3f} (>0.5 is coherent, <0.3 is noise)

TOP TRENDS:
{chr(10).join(trend_summaries[:10])}

THINK step by step:
1. OBSERVE: Which trends look genuinely actionable for B2B sales?
2. CONCERN: Which trends look generic/recycled/too vague for sales outreach?
3. QUALITY: Rate overall trend quality (are these trends a sales rep could USE?)
4. HYPOTHESIZE: What would produce better trends next run?
5. PROJECT: Which 2-3 trends will generate the best leads?

Respond as JSON:
{{
  "observations": ["..."],
  "concerns": ["..."],
  "strengths": ["..."],
  "hypotheses": ["..."],
  "quality_score": 0.0-1.0,
  "confidence": 0.0-1.0,
  "projected_outcome": "which trends will produce best leads and why",
  "strategy_adjustments": {{"drop_trends": [], "boost_trends": []}}
}}"""

        try:
            result = await self._llm.generate_json(prompt=prompt)
            if result and "error" not in result:
                trace.observations = result.get("observations", [])[:5]
                trace.concerns = result.get("concerns", [])[:5]
                trace.strengths = result.get("strengths", [])[:3]
                trace.hypotheses = result.get("hypotheses", [])[:5]
                trace.quality_score = min(1.0, max(0.0, float(result.get("quality_score", 0.5))))
                trace.confidence = min(1.0, max(0.0, float(result.get("confidence", 0.5))))
                trace.projected_outcome = result.get("projected_outcome", "")
                trace.strategy_adjustments = result.get("strategy_adjustments", {})
        except Exception as e:
            logger.debug(f"MetaReasoner trend reasoning failed: {e}")

        trace.reasoning_time_ms = (time.time() - t0) * 1000
        trace.timestamp = datetime.now(timezone.utc).isoformat()
        self._traces.append(trace)
        self._persist_trace(trace)

        if trace.quality_score < 0.4:
            logger.warning(
                f"MetaReasoner [trends]: LOW QUALITY score={trace.quality_score:.2f} — "
                f"concerns: {trace.concerns[:2]}"
            )

        return trace

    async def reason_about_leads(
        self,
        lead_count: int,
        company_count: int,
        contact_count: int,
        email_count: int,
        lead_summaries: List[Dict[str, Any]],
    ) -> ReasoningTrace:
        """After lead_gen: reason about lead quality and personalization strategy.

        Questions:
        - Are leads targeting the right decision-makers?
        - Are emails personalized to the person's role and pain points?
        - Which leads have the highest conversion potential?
        - What personalization improvements would increase response rates?
        """
        trace = ReasoningTrace(step="lead_gen", run_id=self._run_id)
        if not self._enabled:
            return trace

        t0 = time.time()

        lead_details = []
        for ls in lead_summaries[:8]:
            lead_details.append(
                f"  - Company: {ls.get('company', '?')}, "
                f"Contacts: {ls.get('contacts', 0)}, "
                f"Confidence: {ls.get('confidence', 0):.2f}, "
                f"Event: {ls.get('event_type', '?')}"
            )

        prompt = f"""You are the meta-reasoning layer evaluating lead generation quality.

LEAD GEN RESULTS:
- {lead_count} leads, {company_count} companies, {contact_count} contacts, {email_count} emails
- Lead details:
{chr(10).join(lead_details[:8])}

THINK step by step:
1. OBSERVE: What's the quality distribution? How many leads look actionable?
2. CONCERN: Which leads look weak? Why? (wrong company size, bad timing, generic pitch)
3. PERSONALIZATION: Are emails personalized to person's ROLE and specific PAIN POINTS?
4. HYPOTHESIZE: What would produce higher-converting leads next run?
5. PROJECT: If a sales rep called these leads today, what response rate would you expect?

Respond as JSON:
{{
  "observations": ["..."],
  "concerns": ["..."],
  "strengths": ["..."],
  "hypotheses": ["..."],
  "quality_score": 0.0-1.0,
  "confidence": 0.0-1.0,
  "projected_outcome": "expected conversion quality",
  "strategy_adjustments": {{"personalization_tips": [], "targeting_improvements": []}}
}}"""

        try:
            result = await self._llm.generate_json(prompt=prompt)
            if result and "error" not in result:
                trace.observations = result.get("observations", [])[:5]
                trace.concerns = result.get("concerns", [])[:5]
                trace.strengths = result.get("strengths", [])[:3]
                trace.hypotheses = result.get("hypotheses", [])[:5]
                trace.quality_score = min(1.0, max(0.0, float(result.get("quality_score", 0.5))))
                trace.confidence = min(1.0, max(0.0, float(result.get("confidence", 0.5))))
                trace.projected_outcome = result.get("projected_outcome", "")
                trace.strategy_adjustments = result.get("strategy_adjustments", {})
        except Exception as e:
            logger.debug(f"MetaReasoner lead reasoning failed: {e}")

        trace.reasoning_time_ms = (time.time() - t0) * 1000
        trace.timestamp = datetime.now(timezone.utc).isoformat()
        self._traces.append(trace)
        self._persist_trace(trace)
        return trace

    async def run_retrospective(
        self,
        run_id: str,
        total_duration_s: float,
        trend_count: int,
        lead_count: int,
        mean_oss: float,
        mean_kb_hit: float,
        mean_quality: float,
        signal_bus_summary: Dict[str, Any],
        previous_retrospective: Optional[Dict] = None,
    ) -> Retrospective:
        """Full post-run chain-of-thought self-critique.

        The most important reasoning step — this is where the system:
        1. Reviews ALL reasoning traces from this run
        2. Identifies patterns in what went well/poorly
        3. Generates concrete, testable improvement hypotheses
        4. Compares to previous run to detect regressions
        5. Produces an actionable improvement plan for next run
        """
        retro = Retrospective(run_id=run_id)
        if not self._enabled:
            return retro

        t0 = time.time()

        # Summarize this run's reasoning traces
        trace_summary = []
        for tr in self._traces:
            trace_summary.append(
                f"  [{tr.step}] quality={tr.quality_score:.2f}, "
                f"concerns: {tr.concerns[:2]}, hypotheses: {tr.hypotheses[:2]}"
            )

        prev_comparison = ""
        if previous_retrospective:
            prev_comparison = (
                f"\nPREVIOUS RUN: grade={previous_retrospective.get('run_grade', '?')}, "
                f"improvements planned: {previous_retrospective.get('improvement_plan', [])[:3]}"
            )

        prompt = f"""You are the meta-reasoning layer performing a FULL RETROSPECTIVE on this pipeline run.

RUN METRICS:
- Duration: {total_duration_s:.0f}s ({total_duration_s/60:.1f} min)
- Trends detected: {trend_count}
- Leads generated: {lead_count}
- Mean OSS (trend specificity): {mean_oss:.3f}
- Mean KB hit rate: {mean_kb_hit:.1%}
- Mean composite quality: {mean_quality:.3f}

SIGNAL BUS STATE:
- System confidence: {signal_bus_summary.get('system_confidence', '?')}
- Exploration budget: {signal_bus_summary.get('exploration_budget', '?')}
- Learning path: {signal_bus_summary.get('learning_path', '?')}
- Source degraded: {signal_bus_summary.get('source_degraded', [])}

REASONING TRACES (from earlier steps):
{chr(10).join(trace_summary) if trace_summary else '  (no traces available)'}
{prev_comparison}

PREVIOUS IMPROVEMENT HYPOTHESES:
{json.dumps(self._previous_hypotheses[:5], indent=2) if self._previous_hypotheses else '  (none - first run)'}

PERFORM A THOROUGH RETROSPECTIVE. Think like a senior engineer reviewing their own system:

1. GRADE this run (A/B/C/D/F) with justification
2. What SUCCEEDED? (specific wins, not generic praise)
3. What FAILED? (specific failures with root cause analysis)
4. Any SURPRISES? (unexpected outcomes — good or bad)
5. IMPROVEMENT PLAN: List 3-5 specific, testable changes for next run
   - Each must have: target component, specific action, expected impact, priority
6. REGRESSION RISKS: What could get worse if we make these changes?
7. Compare to PREVIOUS RUN: better, worse, or similar? Why?

Respond as JSON:
{{
  "run_grade": "A/B/C/D/F",
  "summary": "one paragraph assessment",
  "successes": ["specific win 1", "..."],
  "failures": ["specific failure with root cause", "..."],
  "surprises": ["unexpected outcome", "..."],
  "improvement_plan": [
    {{"target": "component_name", "action": "specific change", "expected_impact": "what improves", "priority": "high/medium/low"}}
  ],
  "regression_risks": ["risk 1", "..."],
  "vs_previous": "Better/Worse/Similar — because..."
}}"""

        try:
            result = await self._llm.generate_json(prompt=prompt)
            if result and "error" not in result:
                retro.run_grade = result.get("run_grade", "C")
                retro.summary = result.get("summary", "")
                retro.successes = result.get("successes", [])[:5]
                retro.failures = result.get("failures", [])[:5]
                retro.surprises = result.get("surprises", [])[:3]
                retro.improvement_plan = result.get("improvement_plan", [])[:5]
                retro.regression_risks = result.get("regression_risks", [])[:3]
                retro.vs_previous = result.get("vs_previous", "Unknown")
        except Exception as e:
            logger.debug(f"MetaReasoner retrospective failed: {e}")
            retro.summary = f"Retrospective reasoning failed: {e}"

        retro.timestamp = datetime.now(timezone.utc).isoformat()

        # Persist improvement hypotheses for next run
        if retro.improvement_plan:
            self._save_hypotheses(retro.improvement_plan)

        # Log the retrospective
        logger.info(f"MetaReasoner RETROSPECTIVE: Grade={retro.run_grade}")
        if retro.successes:
            logger.info(f"  Successes: {retro.successes[:3]}")
        if retro.failures:
            logger.warning(f"  Failures: {retro.failures[:3]}")
        if retro.improvement_plan:
            logger.info(f"  Improvement plan: {len(retro.improvement_plan)} items")
            for plan in retro.improvement_plan[:3]:
                if isinstance(plan, dict):
                    logger.info(f"    [{plan.get('priority', '?')}] {plan.get('target', '?')}: {plan.get('action', '?')}")

        # Persist retrospective trace
        self._persist_trace_raw({
            "step": "retrospective",
            "run_id": run_id,
            "timestamp": retro.timestamp,
            **asdict(retro),
            "reasoning_time_ms": (time.time() - t0) * 1000,
        })

        return retro

    # ──────────────────────────────────────────────────────────────────
    # Hypothesis management (cross-run learning)
    # ──────────────────────────────────────────────────────────────────

    def get_active_hypotheses(self, target: Optional[str] = None) -> List[Dict]:
        """Get improvement hypotheses from previous runs.

        These are the system's own ideas for how to improve.
        """
        if target:
            return [h for h in self._previous_hypotheses if h.get("target") == target]
        return self._previous_hypotheses

    def _load_hypotheses(self) -> List[Dict]:
        """Load improvement hypotheses generated by previous retrospectives."""
        if _HYPOTHESES_PATH.exists():
            try:
                with open(_HYPOTHESES_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            except Exception:
                pass
        return []

    def _save_hypotheses(self, hypotheses: List[Dict]) -> None:
        """Save improvement hypotheses for next run to read."""
        try:
            _HYPOTHESES_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_HYPOTHESES_PATH, "w", encoding="utf-8") as f:
                json.dump(hypotheses, f, indent=2)
        except Exception as e:
            logger.warning(f"MetaReasoner: failed to save hypotheses: {e}")

    # ──────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────

    def _persist_trace(self, trace: ReasoningTrace) -> None:
        """Append a reasoning trace to the JSONL log."""
        self._persist_trace_raw(asdict(trace))

    def _persist_trace_raw(self, data: dict) -> None:
        """Append raw dict to the reasoning trace log."""
        try:
            _TRACES_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_TRACES_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, default=str) + "\n")
        except Exception as e:
            logger.debug(f"MetaReasoner trace persistence failed: {e}")

    @property
    def traces(self) -> List[ReasoningTrace]:
        return self._traces

    def get_run_summary(self) -> Dict[str, Any]:
        """Get summary of all reasoning from this run for the signal bus."""
        if not self._traces:
            return {}

        avg_quality = sum(t.quality_score for t in self._traces) / len(self._traces)
        all_concerns = []
        all_hypotheses = []
        all_adjustments: Dict[str, Any] = {}
        for t in self._traces:
            all_concerns.extend(t.concerns[:2])
            all_hypotheses.extend(t.hypotheses[:2])
            if t.strategy_adjustments:
                all_adjustments[t.step] = t.strategy_adjustments

        return {
            "avg_reasoning_quality": round(avg_quality, 3),
            "total_concerns": len(all_concerns),
            "top_concerns": all_concerns[:5],
            "top_hypotheses": all_hypotheses[:5],
            "reasoning_steps": len(self._traces),
            "total_reasoning_time_ms": sum(t.reasoning_time_ms for t in self._traces),
            "strategy_adjustments": all_adjustments,
        }

    async def record_run_metrics(self, metrics: Dict[str, Any]) -> None:
        """Record intelligence pipeline run metrics to reasoning trace log.

        Called by event_bus on intelligence.run.complete. Persists structured
        metrics for cross-run introspection and improvement hypothesis generation.
        """
        self._persist_trace_raw({
            "step": "intelligence_run_complete",
            "run_id": metrics.get("run_id", ""),
            "articles_fetched": metrics.get("articles_fetched", 0),
            "articles_filtered": metrics.get("articles_filtered", 0),
            "clusters_passed": metrics.get("clusters_passed", 0),
            "clusters_rejected": metrics.get("clusters_rejected", 0),
            "noise_rate": metrics.get("noise_rate", 0.0),
            "mean_coherence": metrics.get("mean_coherence", 0.0),
            "gap4_dropped": metrics.get("gap4_dropped", []),
            "match_results": metrics.get("match_results", 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


# ── Singleton ──────────────────────────────────────────────────────────────────
_INSTANCE: Optional["MetaReasoner"] = None


def get_meta_reasoner() -> "MetaReasoner":
    """Get or create MetaReasoner singleton (no-LLM mode for event bus use).

    The full LLM-enabled MetaReasoner is instantiated by the orchestrator
    with a live LLMService. This singleton is for event bus logging only.
    """
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = MetaReasoner(llm_service=None, enabled=False)
    return _INSTANCE
