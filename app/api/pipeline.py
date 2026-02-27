"""Pipeline API router -- trigger runs, stream progress via SSE, get results.

SSE Design: The orchestrator's graph.astream(stream_mode="updates") yields
per-node output deltas (avoids msgpack serialization of complex AgentDeps).
Each delta has current_step which maps to a progress percentage. The
_execute_pipeline background task pushes events to an asyncio.Queue that the
SSE endpoint reads from.

Mock replay: When mock_mode=True, if a previous recording exists in
data/recordings/, the pipeline replays that recording through the same SSE
channel with compressed timing (~45s) instead of running the real pipeline.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.run_manager import PipelineRun, run_manager, STEP_PROGRESS
from app.api.schemas import (
    PipelineRunRequest,
    PipelineRunResponse,
    PipelineStatusResponse,
    PipelineResultResponse,
    TrendResponse,
    LeadResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Target duration for mock replay (seconds)
REPLAY_TARGET_SECONDS = 45


@router.post("/run", response_model=PipelineRunResponse)
async def start_pipeline(
    body: PipelineRunRequest,
    background_tasks: BackgroundTasks,
):
    """Start a pipeline run in the background. Returns run_id for SSE streaming.

    When mock_mode=True and a recording exists, replays that recording with
    compressed timing (~45s) instead of running the real pipeline.
    """
    if run_manager.is_running:
        raise HTTPException(409, "Pipeline already running")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run = run_manager.create_run(run_id)

    # Decide: replay a recording or run the real pipeline
    if body.mock_mode:
        from app.tools.run_recorder import get_recording, get_latest_recording
        recording_dir = None
        if body.replay_run_id:
            recording_dir = get_recording(body.replay_run_id)
        if not recording_dir:
            recording_dir = get_latest_recording()

        if recording_dir:
            background_tasks.add_task(_replay_pipeline, run, recording_dir)
            return PipelineRunResponse(
                run_id=run_id,
                status="started",
                message=f"Mock replay started (from {recording_dir.name}). "
                        f"Stream at /api/v1/pipeline/stream/{run_id}",
            )

    background_tasks.add_task(_execute_pipeline, run, body.mock_mode, body.disabled_providers)

    return PipelineRunResponse(
        run_id=run_id,
        status="started",
        message=f"Pipeline started. Stream progress at /api/v1/pipeline/stream/{run_id}",
    )


class _SSELogHandler(logging.Handler):
    """Captures pipeline log messages and pushes them into the SSE queue.

    Attached to orchestrator / agent loggers for the duration of a pipeline run
    so that every logger.info() call appears in the frontend terminal.
    """

    def __init__(self, run: PipelineRun):
        super().__init__(logging.DEBUG)
        self._run = run

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            # Skip noisy / internal messages
            if any(skip in msg for skip in ("HTTP Request:", "httpx", "httpcore")):
                return
            level = record.levelname.lower()
            self._run.event_queue.put_nowait({
                "event": "log",
                "message": msg,
                "level": level,
            })
        except (asyncio.QueueFull, Exception):
            pass


# Logger names to capture for the terminal panel
_PIPELINE_LOGGERS = [
    "app.agents.orchestrator",
    "app.agents.source_intel",
    "app.agents.analysis",
    "app.agents.market_impact",
    "app.agents.quality",
    "app.agents.lead_gen",
    "app.agents.lead_crystallizer",
    "app.agents.causal_council",
    "app.agents.workers.impact_agent",
    "app.agents.workers.company_agent",
    "app.agents.workers.contact_agent",
    "app.agents.workers.email_agent",
    "app.trends.engine",
    "app.trends.synthesis",
    "app.news.scraper",
    "app.tools.llm_service",
    "app.tools.provider_manager",
]


async def _execute_pipeline(run: PipelineRun, mock_mode: bool, disabled_providers: list[str] | None = None):
    """Background task: run the LangGraph pipeline and emit SSE events."""
    from app.agents.orchestrator import create_pipeline_graph
    from app.agents.deps import AgentDeps

    run.status = "running"

    # Reset provider health, cooldowns, and agent cache so stale failures from
    # previous runs don't block this run's LLM calls.
    try:
        from app.tools.provider_health import provider_health
        from app.tools.provider_manager import ProviderManager
        from app.tools.llm_service import LLMService
        provider_health.reset_for_new_run()
        ProviderManager.reset_cooldowns()
        LLMService.clear_cache()
        logger.info("Provider health + cooldowns + agent cache reset for new run")
    except Exception as e:
        logger.warning(f"Provider reset failed: {e}")

    # Attach SSE log handler to pipeline loggers so every logger.info() call
    # appears in the frontend terminal panel in real-time.
    sse_handler = _SSELogHandler(run)
    sse_handler.setFormatter(logging.Formatter("%(message)s"))
    attached_loggers: list[logging.Logger] = []
    for name in _PIPELINE_LOGGERS:
        lg = logging.getLogger(name)
        lg.addHandler(sse_handler)
        attached_loggers.append(lg)

    def progress_callback(msg, level="info"):
        """Forward pipeline log messages to the SSE queue."""
        try:
            run.event_queue.put_nowait({
                "event": "log",
                "message": msg,
                "level": level,
            })
        except asyncio.QueueFull:
            pass

    import time as _time
    deps = AgentDeps.create(
        mock_mode=mock_mode,
        log_callback=progress_callback,
        run_id=run.run_id,
        disabled_providers=disabled_providers or [],
    )
    deps._pipeline_t0 = _time.time()

    initial_state = {
        "deps": deps,
        "run_id": run.run_id,
        "trends": [],
        "impacts": [],
        "companies": [],
        "contacts": [],
        "outreach_emails": [],
        "errors": [],
        "current_step": "init",
        "retry_counts": {},
        "agent_reasoning": {},
    }

    config = {"configurable": {"thread_id": run.run_id}}

    try:
        graph = create_pipeline_graph()

        # Use stream_mode="updates" to avoid msgpack serialization of complex
        # deps objects (ChromaDB, LLM models, etc.) that happens with "values".
        # "updates" yields per-node output deltas — lighter and doesn't serialize
        # the full state graph.
        async for node_output in graph.astream(initial_state, config, stream_mode="updates"):
            # node_output is {node_name: {key: value}} — merge into running state
            for _node_name, delta in node_output.items():
                if not isinstance(delta, dict):
                    continue
                step = delta.get("current_step", run.current_step)
                run.current_step = step
                run.progress_pct = STEP_PROGRESS.get(step, run.progress_pct)

                # Merge delta into final_state
                if run.final_state is None:
                    run.final_state = dict(initial_state)
                run.final_state.update(delta)

                run.trends_count = len(run.final_state.get("trends", []))
                run.companies_count = len(run.final_state.get("companies", []))
                run.errors = run.final_state.get("errors", [])

                # Count leads from deps
                d = run.final_state.get("deps")
                if d:
                    run.leads_count = len(getattr(d, "_lead_sheets", []))

            await run.event_queue.put({
                "event": "progress",
                "step": run.current_step,
                "progress_pct": run.progress_pct,
                "trends": run.trends_count,
                "companies": run.companies_count,
                "leads": run.leads_count,
            })

        # Persist outputs to disk + DB (same as CLI/Streamlit path)
        try:
            from app.agents.orchestrator import save_outputs
            await save_outputs(run.final_state, run.run_id)
        except Exception as e:
            logger.warning(f"Output persistence failed (non-fatal): {e}")

        run.status = "completed"
        run.completed_at = datetime.now(timezone.utc)
        elapsed = (run.completed_at - run.started_at).total_seconds()

        # Update DB run record with actual elapsed time
        try:
            from app.database import get_database
            db = get_database()
            db.update_pipeline_run(run.run_id, {
                "status": "completed",
                "run_time_seconds": round(elapsed, 1),
            })
        except Exception:
            pass

        await run.event_queue.put({
            "event": "complete",
            "run_id": run.run_id,
            "summary": {
                "trends": run.trends_count,
                "companies": run.companies_count,
                "leads": run.leads_count,
                "runtime": round(elapsed, 1),
            },
        })

        logger.info(
            f"Pipeline {run.run_id} completed: "
            f"{run.trends_count} trends, {run.leads_count} leads, "
            f"{elapsed:.0f}s"
        )

    except Exception as e:
        # If we got past all nodes (recordings exist) but hit a serialization
        # error, treat as completed with warning
        if run.final_state and "msgpack" in str(e).lower():
            logger.warning(f"Pipeline {run.run_id} completed with serialization warning: {e}")
            try:
                from app.agents.orchestrator import save_outputs
                await save_outputs(run.final_state, run.run_id)
            except Exception as save_err:
                logger.warning(f"Output persistence failed: {save_err}")
            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc)
            elapsed = (run.completed_at - run.started_at).total_seconds()
            await run.event_queue.put({
                "event": "complete",
                "run_id": run.run_id,
                "summary": {
                    "trends": run.trends_count,
                    "companies": run.companies_count,
                    "leads": run.leads_count,
                    "runtime": round(elapsed, 1),
                },
            })
        else:
            run.status = "failed"
            run.errors.append(str(e))
            run.completed_at = datetime.now(timezone.utc)
            logger.error(f"Pipeline {run.run_id} failed: {e}")
            await run.event_queue.put({
                "event": "error",
                "message": str(e),
            })

    finally:
        # Detach the SSE handler to avoid leaking into future runs
        for lg in attached_loggers:
            lg.removeHandler(sse_handler)


def _generate_replay_logs(step_name: str, step_data: dict, step_info: dict) -> list[str]:
    """Generate synthetic log messages for replay from recorded step data."""
    msgs = []
    duration = step_info.get("duration_s", 0)

    if step_name == "source_intel_complete":
        articles = step_data.get("articles", [])
        msgs.append(f"=== STEP 1: SOURCE INTEL AGENT ===")
        msgs.append(f"Collected {len(articles)} articles from RSS feeds")
        sources = set()
        for a in articles[:100]:
            src = a.get("source_id", "") or a.get("source_name", "") or a.get("source", "")
            if src:
                sources.add(src)
        if sources:
            msgs.append(f"Sources: {', '.join(list(sources)[:8])}...")
        msgs.append(f"Source intel completed in {duration:.0f}s")

    elif step_name == "analysis_complete":
        trend_count = step_data.get("trend_count", 0)
        trends = step_data.get("trends", [])
        msgs.append(f"=== STEP 2: ANALYSIS AGENT ===")
        msgs.append(f"Detected {trend_count} market trends")
        for t in trends[:5]:
            title = t.get("trend_title", "") or t.get("title", "")
            oss = t.get("oss_score", 0)
            msgs.append(f"  Trend: {title[:70]} (OSS={oss})")
        if len(trends) > 5:
            msgs.append(f"  ... and {len(trends) - 5} more trends")
        msgs.append(f"Analysis completed in {duration:.0f}s")

    elif step_name == "impact_complete":
        impacts = step_data.get("impacts", [])
        msgs.append(f"=== STEP 3: MARKET IMPACT AGENT ===")
        msgs.append(f"Analyzed impact for {len(impacts)} trends")
        msgs.append(f"Impact analysis completed in {duration:.0f}s")

    elif step_name == "quality_complete":
        msgs.append(f"=== QUALITY VALIDATION ===")
        viable = step_data.get("viable_count", "?")
        msgs.append(f"Quality gate passed: {viable} viable impacts")
        msgs.append(f"Quality validation completed in {duration:.0f}s")

    elif step_name == "causal_council_complete":
        results = step_data.get("causal_results", [])
        total_hops = step_data.get("total_hops", 0)
        msgs.append(f"=== STEP 3.7: CAUSAL COUNCIL ===")
        msgs.append(f"Traced causal chains for {len(results)} trends ({total_hops} total hops)")
        for r in results[:3]:
            title = (r.get("event_summary", "") or r.get("trend_title", ""))[:50]
            hops = len(r.get("hops", []))
            msgs.append(f"  {title}: {hops} causal hops")
        msgs.append(f"Causal council completed in {duration:.0f}s")

    elif step_name == "lead_crystallize_complete":
        leads = step_data.get("lead_sheets", [])
        msgs.append(f"=== STEP 3.8: LEAD CRYSTALLIZER ===")
        msgs.append(f"Crystallized {len(leads)} call sheets")
        for ls in leads[:3]:
            company = ls.get("company_name", "?")
            conf = ls.get("confidence", 0)
            msgs.append(f"  {company} — confidence {conf:.0%}")
        if len(leads) > 3:
            msgs.append(f"  ... and {len(leads) - 3} more leads")
        msgs.append(f"Lead crystallization completed in {duration:.0f}s")

    elif step_name == "lead_gen_complete":
        companies = step_data.get("company_count", 0)
        contacts = step_data.get("contact_count", 0)
        msgs.append(f"=== STEP 4: LEAD GEN AGENT ===")
        msgs.append(f"Found {companies} companies, {contacts} contacts")
        msgs.append(f"Lead generation completed in {duration:.0f}s")

    elif step_name == "learning_update_complete":
        signals = step_data.get("signal_count", 0)
        msgs.append(f"=== LEARNING UPDATE ===")
        msgs.append(f"Processed {signals} learning signals")
        avg_oss = step_data.get("avg_oss", 0)
        msgs.append(f"Average OSS: {avg_oss:.3f}")
        msgs.append(f"Learning update completed in {duration:.0f}s")

    else:
        msgs.append(f"Step {step_name} completed in {duration:.0f}s")

    return msgs


def _load_run_from_recording(run_id: str, recording_dir: Path) -> PipelineRun:
    """Load a completed run from recording files (no SSE streaming).

    Used when a run is not in memory (e.g. after server restart) but
    recording files exist on disk.
    """
    manifest = json.loads((recording_dir / "manifest.json").read_text(encoding="utf-8"))
    run = run_manager.create_run(run_id)

    replay_trends = []
    replay_leads = []
    replay_impacts = []
    replay_companies = []
    replay_contacts = []
    replay_outreach = []

    for step_info in manifest["steps"]:
        step_name = step_info["step"]
        sf = recording_dir / f"{step_info['order']:02d}_{step_name}.json"
        if not sf.exists():
            continue
        sd = json.loads(sf.read_text(encoding="utf-8"))
        if step_name == "analysis_complete":
            replay_trends = sd.get("trends", [])
            run.trends_count = len(replay_trends)
        if step_name == "impact_complete":
            replay_impacts = sd.get("impacts", [])
        if step_name == "lead_crystallize_complete":
            replay_leads = sd.get("lead_sheets", [])
            run.leads_count = sd.get("lead_count", len(replay_leads))
        if step_name == "lead_gen_complete":
            run.companies_count = sd.get("company_count", 0)
            replay_companies = sd.get("companies", [])
            replay_contacts = sd.get("contacts", [])
            replay_outreach = sd.get("outreach", [])

    run.result = {
        "recording": str(recording_dir),
        "replay": True,
        "trends": replay_trends,
        "leads": replay_leads,
        "impacts": replay_impacts,
        "companies": replay_companies,
        "contacts": replay_contacts,
        "outreach": replay_outreach,
    }
    run.status = "completed"
    run.completed_at = datetime.now(timezone.utc)
    return run


async def _replay_pipeline(run: PipelineRun, recording_dir: Path):
    """Replay a recorded pipeline run with compressed timing for demos.

    Reads the recording manifest and step files, then streams the same SSE
    events that a real pipeline would produce, but with timing compressed
    to ~45 seconds total.
    """
    run.status = "running"

    try:
        manifest = json.loads((recording_dir / "manifest.json").read_text(encoding="utf-8"))
        real_duration = manifest.get("total_duration_s", 300)

        # Compress timing: e.g. 3600s real → 45s demo = 80x speedup
        speed_factor = max(real_duration / REPLAY_TARGET_SECONDS, 1.0)

        logger.info(
            f"Replaying recording {recording_dir.name}: "
            f"{manifest['step_count']} steps, {real_duration:.0f}s → "
            f"~{REPLAY_TARGET_SECONDS}s (speed {speed_factor:.0f}x)"
        )

        for step_info in manifest["steps"]:
            step_name = step_info["step"]
            order = step_info["order"]
            step_file = recording_dir / f"{order:02d}_{step_name}.json"

            if not step_file.exists():
                logger.warning(f"Replay: missing step file {step_file}")
                continue

            step_data = json.loads(step_file.read_text(encoding="utf-8"))

            # Simulate processing time (compressed)
            demo_delay = step_info.get("duration_s", 5) / speed_factor
            await asyncio.sleep(demo_delay)

            # Update run state from recorded data
            run.current_step = step_name
            run.progress_pct = STEP_PROGRESS.get(step_name, run.progress_pct)

            # Extract counts from recorded data
            if "trend_count" in step_data:
                run.trends_count = step_data["trend_count"]
            if "company_count" in step_data:
                run.companies_count = step_data["company_count"]
            if "lead_count" in step_data:
                run.leads_count = step_data["lead_count"]

            # Emit SSE progress event (same format as real pipeline)
            await run.event_queue.put({
                "event": "progress",
                "step": step_name,
                "progress_pct": run.progress_pct,
                "trends": run.trends_count,
                "companies": run.companies_count,
                "leads": run.leads_count,
            })

            # Emit log messages from recorded data or generate from step data
            log_msgs = step_data.get("log_messages", [])
            if not log_msgs:
                log_msgs = _generate_replay_logs(step_name, step_data, step_info)
            for log_msg in log_msgs:
                msg_text = log_msg if isinstance(log_msg, str) else log_msg.get("text", str(log_msg))
                msg_level = "info" if isinstance(log_msg, str) else log_msg.get("level", "info")
                try:
                    run.event_queue.put_nowait({
                        "event": "log",
                        "message": msg_text,
                        "level": msg_level,
                    })
                except asyncio.QueueFull:
                    pass
                await asyncio.sleep(0.05)  # drip-feed logs for visual effect

        # Build result data from recorded steps for the result endpoint
        replay_trends = []
        replay_leads = []
        replay_impacts = []
        replay_companies = []
        replay_contacts = []
        replay_outreach = []
        for step_info in manifest["steps"]:
            step_name = step_info["step"]
            sf = recording_dir / f"{step_info['order']:02d}_{step_name}.json"
            if not sf.exists():
                continue
            sd = json.loads(sf.read_text(encoding="utf-8"))
            if step_name == "analysis_complete":
                replay_trends = sd.get("trends", [])
                run.trends_count = len(replay_trends)
            if step_name == "impact_complete":
                replay_impacts = sd.get("impacts", [])
            if step_name == "lead_crystallize_complete":
                replay_leads = sd.get("lead_sheets", [])
                run.leads_count = sd.get("lead_count", len(replay_leads))
            if step_name == "lead_gen_complete":
                run.companies_count = sd.get("company_count", 0)
                replay_companies = sd.get("companies", [])
                replay_contacts = sd.get("contacts", [])
                replay_outreach = sd.get("outreach", [])

        run.result = {
            "recording": str(recording_dir),
            "replay": True,
            "trends": replay_trends,
            "leads": replay_leads,
            "impacts": replay_impacts,
            "companies": replay_companies,
            "contacts": replay_contacts,
            "outreach": replay_outreach,
        }

        run.status = "completed"
        run.completed_at = datetime.now(timezone.utc)
        elapsed = (run.completed_at - run.started_at).total_seconds()

        await run.event_queue.put({
            "event": "complete",
            "run_id": run.run_id,
            "summary": {
                "trends": run.trends_count,
                "companies": run.companies_count,
                "leads": run.leads_count,
                "runtime": round(elapsed, 1),
                "replay": True,
                "original_run": recording_dir.name,
            },
        })

        logger.info(
            f"Replay {run.run_id} completed in {elapsed:.1f}s "
            f"(original: {real_duration:.0f}s)"
        )

    except Exception as e:
        run.status = "failed"
        run.errors.append(str(e))
        run.completed_at = datetime.now(timezone.utc)
        logger.error(f"Replay {run.run_id} failed: {e}")

        await run.event_queue.put({
            "event": "error",
            "message": f"Replay failed: {e}",
        })


@router.get("/stream/{run_id}")
async def stream_progress(run_id: str):
    """SSE endpoint -- stream real-time pipeline progress to the frontend.

    Connect with EventSource in Next.js:
        const es = new EventSource(`/api/v1/pipeline/stream/${runId}`);
        es.onmessage = (e) => { const data = JSON.parse(e.data); ... };
    """
    run = run_manager.get_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")

    async def event_generator():
        heartbeat_interval = 15  # seconds
        while True:
            try:
                event = await asyncio.wait_for(
                    run.event_queue.get(),
                    timeout=heartbeat_interval,
                )
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("event") in ("complete", "error"):
                    break
            except asyncio.TimeoutError:
                # Heartbeat to keep connection alive through proxies
                yield f"data: {json.dumps({'event': 'heartbeat'})}\n\n"
            except Exception:
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/status/{run_id}", response_model=PipelineStatusResponse)
async def get_status(run_id: str):
    """Poll pipeline status (fallback for clients that can't use SSE)."""
    run = run_manager.get_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")

    elapsed = (
        datetime.now(timezone.utc) - run.started_at
    ).total_seconds()

    return PipelineStatusResponse(
        run_id=run_id,
        status=run.status,
        current_step=run.current_step,
        progress_pct=run.progress_pct,
        trends_detected=run.trends_count,
        companies_found=run.companies_count,
        leads_generated=run.leads_count,
        errors=run.errors,
        started_at=run.started_at.isoformat(),
        elapsed_seconds=round(elapsed, 1),
    )


@router.get("/result/{run_id}", response_model=PipelineResultResponse)
async def get_result(run_id: str):
    """Get full pipeline results after completion."""
    run = run_manager.get_run(run_id)
    # If run is in memory but has no enrichment data, reload from recordings
    if run and run.result and isinstance(run.result, dict) and not run.result.get("companies"):
        logger.info(f"Result {run_id}: reloading from recordings (missing enrichment)")
        run = None  # Force recording reload
    if not run:
        # Fallback: load from recordings (handles server restart / in-memory loss)
        from app.tools.run_recorder import get_recording
        recording_dir = get_recording(run_id)
        if recording_dir and (recording_dir / "manifest.json").exists():
            run = _load_run_from_recording(run_id, recording_dir)
        else:
            raise HTTPException(404, "Run not found")
    if run.status not in ("completed", "failed"):
        raise HTTPException(202, "Pipeline still running")

    # Build trend + lead responses from final state OR replay data
    trends = []
    leads = []

    logger.info(
        f"Result {run_id}: has_result={run.result is not None}, "
        f"is_replay={run.result.get('replay') if isinstance(run.result, dict) else 'N/A'}, "
        f"has_final_state={run.final_state is not None}, "
        f"companies={len(run.result.get('companies', [])) if isinstance(run.result, dict) else 'N/A'}"
    )

    # Check if this is a replay result (dicts from JSON recordings)
    if run.result and isinstance(run.result, dict) and run.result.get("replay"):
        # Index impact data by trend_title for joining to trends
        impact_by_title: dict[str, dict] = {}
        for imp in run.result.get("impacts", []):
            title = imp.get("trend_title", "")
            if title:
                impact_by_title[title] = imp

        for t in run.result.get("trends", []):
            # Recording uses trend_title/industries_affected; schema uses title/industries
            _parse = lambda v, fallback=[]: v if isinstance(v, (list, dict)) else (json.loads(v) if isinstance(v, str) and v.startswith(("[", "{")) else fallback)
            trend_title = t.get("trend_title", "") or t.get("title", "")
            imp = impact_by_title.get(trend_title, {})
            trends.append(TrendResponse(
                id=str(t.get("id", "")),
                title=trend_title,
                summary=t.get("summary", ""),
                severity=t.get("severity", ""),
                trend_type=t.get("trend_type", ""),
                industries=_parse(t.get("industries_affected", []) or t.get("industries", [])),
                keywords=_parse(t.get("keywords", [])),
                trend_score=float(t.get("trend_score", 0)),
                actionability_score=float(t.get("actionability_score", 0)),
                oss_score=float(t.get("oss_score", 0)),
                article_count=int(t.get("article_count", 0)),
                event_5w1h=_parse(t.get("event_5w1h", {}), {}),
                causal_chain=_parse(t.get("causal_chain", [])),
                buying_intent=_parse(t.get("buying_intent", {}), {}),
                affected_companies=_parse(t.get("affected_companies", [])),
                actionable_insight=t.get("actionable_insight", ""),
                article_snippets=_parse(t.get("article_snippets", [])),
                source_links=_parse(t.get("source_links", [])),
                # Impact analysis fields (joined by trend_title)
                direct_impact=_parse(imp.get("direct_impact", [])),
                indirect_impact=_parse(imp.get("indirect_impact", [])),
                midsize_pain_points=_parse(imp.get("midsize_pain_points", [])),
                target_roles=_parse(imp.get("target_roles", [])),
                pitch_angle=imp.get("pitch_angle", ""),
                evidence_citations=_parse(imp.get("evidence_citations", [])),
                who_needs_help=imp.get("who_needs_help", ""),
                council_confidence=float(imp.get("council_confidence", 0)),
            ))

        # Build enrichment indexes from recorded companies/contacts/outreach
        _company_by_name: dict[str, dict] = {}
        for c in run.result.get("companies", []):
            _company_by_name[c.get("company_name", "")] = c
        _contacts_by_company: dict[str, list] = {}
        for ct in run.result.get("contacts", []):
            cname = ct.get("company_name", "")
            _contacts_by_company.setdefault(cname, []).append(ct)
        _outreach_by_company: dict[str, dict] = {}
        for em in run.result.get("outreach", []):
            cname = em.get("company_name", "")
            if cname not in _outreach_by_company:
                _outreach_by_company[cname] = em

        for i, sheet in enumerate(run.result.get("leads", [])):
            cname = sheet.get("company_name", "")
            company = _company_by_name.get(cname, {})
            contacts = _contacts_by_company.get(cname, [])
            outreach = _outreach_by_company.get(cname, {})
            # Pick best contact (first with email, or first available)
            contact = next((c for c in contacts if c.get("email")), contacts[0] if contacts else {})
            leads.append(LeadResponse(
                id=i,
                company_name=cname,
                company_cin=sheet.get("company_cin", ""),
                company_state=sheet.get("company_state", ""),
                company_city=sheet.get("company_city", ""),
                company_size_band=sheet.get("company_size_band", ""),
                company_website=company.get("website", ""),
                company_domain=company.get("domain", ""),
                reason_relevant=company.get("reason_relevant", ""),
                hop=sheet.get("hop", 1),
                lead_type=sheet.get("lead_type", ""),
                trend_title=sheet.get("trend_title", ""),
                event_type=sheet.get("event_type", ""),
                contact_name=contact.get("person_name", ""),
                contact_role=contact.get("role", "") or sheet.get("contact_role", ""),
                contact_email=contact.get("email", ""),
                contact_linkedin=contact.get("linkedin_url", ""),
                email_confidence=int(contact.get("email_confidence", 0)),
                email_subject=outreach.get("subject", ""),
                email_body=outreach.get("body", ""),
                trigger_event=sheet.get("trigger_event", ""),
                pain_point=sheet.get("pain_point", ""),
                service_pitch=sheet.get("service_pitch", ""),
                opening_line=sheet.get("opening_line", ""),
                urgency_weeks=sheet.get("urgency_weeks", 4),
                confidence=sheet.get("confidence", 0.0),
                oss_score=sheet.get("oss_score", 0.0),
                data_sources=sheet.get("data_sources", []),
                company_news=sheet.get("company_news", []),
            ))
    elif run.final_state:
        # Index impacts by trend_title for joining
        _impact_by_title: dict[str, Any] = {}
        for imp in run.final_state.get("impacts", []):
            title = getattr(imp, "trend_title", "")
            if title:
                _impact_by_title[title] = imp

        for t in run.final_state.get("trends", []):
            # TrendData uses trend_title/industries_affected; TrendResponse uses title/industries
            t_title = getattr(t, "trend_title", "") or getattr(t, "title", "")
            imp = _impact_by_title.get(t_title)
            trends.append(TrendResponse(
                id=str(getattr(t, "id", "")),
                title=t_title,
                summary=getattr(t, "summary", ""),
                severity=str(getattr(t, "severity", "")),
                trend_type=getattr(t, "trend_type", ""),
                industries=getattr(t, "industries_affected", []) or getattr(t, "industries", []),
                keywords=getattr(t, "keywords", []),
                trend_score=getattr(t, "trend_score", 0.0),
                actionability_score=getattr(t, "actionability_score", 0.0),
                oss_score=getattr(t, "oss_score", 0.0),
                article_count=getattr(t, "article_count", 0),
                event_5w1h=getattr(t, "event_5w1h", {}),
                causal_chain=getattr(t, "causal_chain", []),
                buying_intent=getattr(t, "buying_intent", {}),
                affected_companies=getattr(t, "affected_companies", []),
                actionable_insight=getattr(t, "actionable_insight", ""),
                article_snippets=getattr(t, "article_snippets", []),
                source_links=getattr(t, "source_links", []),
                # Impact analysis fields (joined by trend_title)
                direct_impact=getattr(imp, "direct_impact", []) if imp else [],
                indirect_impact=getattr(imp, "indirect_impact", []) if imp else [],
                midsize_pain_points=getattr(imp, "midsize_pain_points", []) if imp else [],
                target_roles=getattr(imp, "target_roles", []) if imp else [],
                pitch_angle=getattr(imp, "pitch_angle", "") if imp else "",
                evidence_citations=getattr(imp, "evidence_citations", []) if imp else [],
                who_needs_help=getattr(imp, "who_needs_help", "") if imp else "",
                council_confidence=getattr(imp, "council_confidence", 0.0) if imp else 0.0,
            ))
        deps = run.final_state.get("deps")
        if deps:
            # Build lookup indexes for enrichment data (contacts, companies, emails)
            _company_by_name: dict[str, Any] = {}
            for c in getattr(deps, "_companies", []):
                _company_by_name[getattr(c, "company_name", "")] = c
            _contacts_by_company: dict[str, list] = {}
            for ct in getattr(deps, "_contacts", []):
                cname = getattr(ct, "company_name", "")
                _contacts_by_company.setdefault(cname, []).append(ct)
            _outreach_by_company: dict[str, Any] = {}
            for em in getattr(deps, "_outreach", []):
                cname = getattr(em, "company_name", "")
                if cname not in _outreach_by_company:
                    _outreach_by_company[cname] = em

            for i, sheet in enumerate(getattr(deps, "_lead_sheets", [])):
                cname = getattr(sheet, "company_name", "")
                company = _company_by_name.get(cname)
                contacts = _contacts_by_company.get(cname, [])
                outreach = _outreach_by_company.get(cname)
                # Pick best contact (first with email, or first available)
                contact = next((c for c in contacts if getattr(c, "email", "")), contacts[0] if contacts else None)
                leads.append(LeadResponse(
                    id=i,
                    company_name=cname,
                    company_cin=getattr(sheet, "company_cin", ""),
                    company_state=getattr(sheet, "company_state", ""),
                    company_city=getattr(sheet, "company_city", ""),
                    company_size_band=getattr(sheet, "company_size_band", ""),
                    company_website=getattr(company, "website", "") if company else "",
                    company_domain=getattr(company, "domain", "") if company else "",
                    reason_relevant=getattr(company, "reason_relevant", "") if company else "",
                    hop=getattr(sheet, "hop", 1),
                    lead_type=getattr(sheet, "lead_type", ""),
                    trend_title=getattr(sheet, "trend_title", ""),
                    event_type=getattr(sheet, "event_type", ""),
                    contact_name=getattr(contact, "person_name", "") if contact else "",
                    contact_role=getattr(contact, "role", "") if contact else getattr(sheet, "contact_role", ""),
                    contact_email=getattr(contact, "email", "") if contact else "",
                    contact_linkedin=getattr(contact, "linkedin_url", "") if contact else "",
                    email_confidence=getattr(contact, "email_confidence", 0) if contact else 0,
                    email_subject=getattr(outreach, "subject", "") if outreach else "",
                    email_body=getattr(outreach, "body", "") if outreach else "",
                    trigger_event=getattr(sheet, "trigger_event", ""),
                    pain_point=getattr(sheet, "pain_point", ""),
                    service_pitch=getattr(sheet, "service_pitch", ""),
                    opening_line=getattr(sheet, "opening_line", ""),
                    urgency_weeks=getattr(sheet, "urgency_weeks", 4),
                    confidence=getattr(sheet, "confidence", 0.0),
                    oss_score=getattr(sheet, "oss_score", 0.0),
                    data_sources=getattr(sheet, "data_sources", []),
                    company_news=getattr(sheet, "company_news", []),
                ))

    elapsed = 0.0
    if run.completed_at:
        elapsed = (run.completed_at - run.started_at).total_seconds()

    return PipelineResultResponse(
        run_id=run_id,
        status=run.status,
        trends_detected=run.trends_count,
        companies_found=run.companies_count,
        leads_generated=run.leads_count,
        run_time_seconds=round(elapsed, 1),
        errors=run.errors,
        trends=trends,
        leads=leads,
    )


@router.get("/runs")
async def list_runs(limit: int = 20):
    """List recent pipeline runs — merges in-memory (active) + DB (historical)."""
    # In-memory runs (active + recently completed)
    memory_runs = run_manager.list_runs(limit=limit)
    seen_ids = set()
    results = []

    for r in memory_runs:
        seen_ids.add(r.run_id)
        results.append(PipelineStatusResponse(
            run_id=r.run_id,
            status=r.status,
            current_step=r.current_step,
            progress_pct=r.progress_pct,
            trends_detected=r.trends_count,
            companies_found=r.companies_count,
            leads_generated=r.leads_count,
            errors=r.errors,
            started_at=r.started_at.isoformat(),
            elapsed_seconds=(
                (r.completed_at or datetime.now(timezone.utc)) - r.started_at
            ).total_seconds(),
        ))

    # DB runs (historical, survives restarts)
    if len(results) < limit:
        try:
            from app.database import get_database
            db = get_database()
            db_runs = db.get_pipeline_runs(limit=limit)
            for dr in db_runs:
                if dr["run_id"] not in seen_ids:
                    results.append(PipelineStatusResponse(
                        run_id=dr["run_id"],
                        status=dr["status"],
                        current_step="learning_update_complete",
                        progress_pct=100 if dr["status"] == "completed" else 0,
                        trends_detected=dr["trends_detected"],
                        companies_found=dr["companies_found"],
                        leads_generated=dr["leads_generated"],
                        errors=dr.get("errors", []),
                        started_at=dr.get("started_at", ""),
                        elapsed_seconds=dr.get("run_time_seconds", 0),
                    ))
        except Exception as e:
            logger.warning(f"DB run query failed: {e}")

    # Sort by started_at descending, trim to limit
    results.sort(key=lambda r: r.started_at, reverse=True)
    return results[:limit]
