"""
India Trend Lead Generation Agent - Streamlit Interactive Dashboard
Human-in-the-loop workflow for testing and controlling the agent pipeline.
"""

import asyncio
import base64
import json
from datetime import datetime
from pathlib import Path
import sys

import nest_asyncio
import pandas as pd
import streamlit as st

# Streamlit runs inside an existing event loop — nest_asyncio makes
# asyncio.run() safe to call from within it (avoids RuntimeError).
# Streamlit Cloud uses uvloop which nest_asyncio can't patch — fall back
# to the default asyncio event loop in that case.
try:
    nest_asyncio.apply()
except ValueError:
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    nest_asyncio.apply(loop)

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

# Import UI components (Streamlit-specific)
from app.ui.sidebar import render_sidebar
from app.ui.styles import apply_custom_styles
from app.ui.visualizations import FLOW_AVAILABLE, render_tree_node, render_flow_tree, render_sector_heatmap

# Import framework-agnostic helpers
from app.shared.helpers import _rebuild_list, escape_for_html, format_llm_text

# Import agents and schemas
from app.agents.workers.company_agent import CompanyDiscovery as CompanyAgent
from app.agents.workers.contact_agent import ContactFinder as ContactAgent
from app.agents.workers.email_agent import EmailGenerator as EmailAgent
from app.agents.workers.impact_agent import ImpactAnalyzer as ImpactAgent
from app.config import DEFAULT_ACTIVE_SOURCES, get_domestic_source_ids, get_settings
from app.schemas import AgentState, CompanyData, ContactData, ImpactAnalysis, LeadRecord, OutreachEmail, TrendData
from app.tools.rss_tool import RSSTool
from app.tools.api_checker import APIChecker
from app.trends.engine import RecursiveTrendEngine

# Keys to clear when resetting
_PIPELINE_STATE_KEYS = [
    'trends', 'selected_trends', 'selected_source_articles', 'impacts',
    'companies', 'selected_companies', 'contacts', 'selected_contacts',
    'outreach_emails', 'logs', 'agent_state', 'articles', 'clusters',
    'major_trends', 'trend_tree', 'engine_metrics',
]

# Page config
st.set_page_config(page_title="India Lead Gen Agent", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")
apply_custom_styles()


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'current_step': 0, 'mock_mode': False, 'articles': [], 'clusters': [],
        'major_trends': [], 'trend_tree': None, 'engine_metrics': {}, 'trends': [],
        'selected_trends': [], 'selected_source_articles': [], 'impacts': [],
        'companies': [], 'selected_companies': [], 'contacts': [],
        'selected_contacts': [], 'outreach_emails': [], 'logs': [],
        'pipeline_running': False, 'agent_state': None, 'max_per_source': 10,
        'active_sources': DEFAULT_ACTIVE_SOURCES,
        'audit_sources': False, 'source_audit_results': {},
        'step_1_running': False, 'step_2_running': False,
        'step_3_running': False, 'step_4_running': False,
        # Granular mock controls (which parts to mock when mock_mode is ON)
        'mock_rss': True, 'mock_llm': True, 'mock_search': True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_log_area = None


def _refresh_log_area():
    if _log_area is None or not st.session_state.logs:
        return
    with _log_area.container():
        with st.expander("Pipeline Log", expanded=False):
            # Download link (HTML — no Streamlit key, safe to re-render)
            _debug_log = Path("trend_engine_debug.log")
            if _debug_log.exists():
                b64 = base64.b64encode(_debug_log.read_bytes()).decode()
                st.markdown(
                    f'<div style="text-align:right;margin:-4px 0 6px;">'
                    f'<a href="data:text/plain;base64,{b64}" download="trend_engine_debug.log" '
                    f'title="Download raw debug log" style="text-decoration:none;font-size:16px;opacity:0.7;">📥</a></div>',
                    unsafe_allow_html=True,
                )
            for entry in reversed(st.session_state.logs[-20:]):
                color = {"info": "#888", "success": "#2ed573", "warning": "#ffa502", "error": "#ff4757"}.get(entry['level'], "#888")
                st.markdown(f'<span style="color: #555; font-size: 11px;">{entry["time"]}</span> <span style="color: {color};">{entry["icon"]}</span> <span style="color: #aaa; font-size: 12px;">{entry["message"]}</span>', unsafe_allow_html=True)


def add_log(message: str, level: str = "info"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    icons = {"info": "●", "success": "✓", "warning": "⚠", "error": "✗"}
    st.session_state.logs.append({"time": timestamp, "level": level, "icon": icons.get(level, "●"), "message": message})
    _refresh_log_area()


def _to_trend_data(major_trends: list) -> list:
    return [
        TrendData(
            id=str(mt.id),
            trend_title=mt.trend_title,
            summary=mt.trend_summary,
            severity=mt.severity,
            industries_affected=[
                s.value if hasattr(s, 'value') else str(s) for s in mt.primary_sectors
            ],
            source_links=[],
            keywords=mt.key_entities[:10] if mt.key_entities else mt.key_keywords[:10],
            trend_type=getattr(mt.trend_type, 'value', str(mt.trend_type)) if mt.trend_type else "general",
            actionable_insight=mt.actionable_insight or "",
            event_5w1h=mt.event_5w1h or {},
            causal_chain=mt.causal_chain or [],
            buying_intent=mt.buying_intent or {},
            affected_companies=mt.affected_companies or [],
            affected_regions=[
                r.value if hasattr(r, 'value') else str(r) for r in (mt.affected_regions or [])
            ],
            trend_score=mt.trend_score,
            actionability_score=mt.actionability_score,
            oss_score=getattr(mt, 'oss_score', 0.0),
            article_count=mt.article_count,
            article_snippets=getattr(mt, 'article_snippets', []) or [],
        ) for mt in major_trends
    ]


async def _run_trend_pipeline(progress) -> list:
    """Run the trend detection pipeline."""
    mock_mode = st.session_state.mock_mode
    # Granular: RSS can be real even in mock mode
    mock_rss = mock_mode and st.session_state.get("mock_rss", True)
    mock_llm = mock_mode and st.session_state.get("mock_llm", True)
    settings = get_settings()

    def log(msg, level="info"):
        add_log(msg, level)
        progress.info(f"{msg}")

    max_per_source = st.session_state.get('max_per_source', 10)
    rss_tool = RSSTool(mock_mode=mock_rss)

    # S2: Optional source audit before fetching
    if st.session_state.get('audit_sources', False):
        log("Auditing news sources...")
        audit_results = await rss_tool.audit_sources(timeout=8.0)
        st.session_state.source_audit_results = audit_results
        healthy = [sid for sid, r in audit_results.items() if r.get("status") == "ok"]
        broken = [sid for sid, r in audit_results.items() if r.get("status") in ("broken", "empty")]
        skipped = [sid for sid, r in audit_results.items() if r.get("status") == "no_key"]
        log(f"Source audit: {len(healthy)} healthy, {len(broken)} broken, {len(skipped)} no API key", "info")

    log(f"Fetching up to {max_per_source} articles/source...")
    articles = await rss_tool.fetch_all_sources(max_per_source=max_per_source, hours_ago=72)
    st.session_state.articles = articles

    if len(articles) < 3:
        log("Not enough articles", "error")
        return []

    engine = RecursiveTrendEngine(
        dedup_threshold=settings.dedup_threshold, dedup_num_perm=settings.dedup_num_perm,
        dedup_shingle_size=settings.dedup_shingle_size,
        semantic_dedup_threshold=settings.semantic_dedup_threshold,
        spacy_model=settings.spacy_model,
        max_depth=settings.engine_max_depth, max_concurrent_llm=settings.engine_max_concurrent_llm, mock_mode=mock_llm,
        country=settings.country,
        domestic_source_ids=get_domestic_source_ids(settings.country_code),
    )

    # Attach a log handler so engine log messages appear in Streamlit progress
    import logging as _logging

    class _StreamlitLogHandler(_logging.Handler):
        """Forward engine log messages to Streamlit progress area."""
        def emit(self, record):
            try:
                msg = self.format(record)
                level = {"DEBUG": "info", "INFO": "info", "WARNING": "warning", "ERROR": "error"}.get(record.levelname, "info")
                add_log(msg, level)
                progress.info(msg)
            except Exception:
                pass

    _st_handler = _StreamlitLogHandler()
    _st_handler.setLevel(_logging.INFO)
    _st_handler.setFormatter(_logging.Formatter('%(message)s'))
    engine_logger = _logging.getLogger("app.trends.engine")
    engine_logger.addHandler(_st_handler)

    try:
        log(f"Running full pipeline on {len(articles)} articles...")
        # Add 5-minute timeout to entire pipeline (prevents hanging)
        try:
            tree = await asyncio.wait_for(engine.run(articles), timeout=300.0)
        except asyncio.TimeoutError:
            log("Pipeline timeout (5 minutes) — partial results might be available", "warning")
            # Try to return whatever we have so far
            if hasattr(engine, 'metrics') and engine.metrics:
                log(f"Partial metrics: {engine.metrics}", "info")
            raise TimeoutError("Pipeline exceeded 5-minute timeout")

        st.session_state.trend_tree = tree
        st.session_state.engine_metrics = engine.metrics
        major_trends = tree.to_major_trends()
        st.session_state.major_trends = major_trends

        total_time = engine.metrics.get("total_seconds", 0)
        log(f"Pipeline complete: {len(tree.nodes)} trends in {total_time:.1f}s", "success")
        return _to_trend_data(major_trends) if major_trends else []

    except asyncio.TimeoutError:
        log("Pipeline timeout: exceeded maximum execution time", "error")
        return []
    except Exception as e:
        log(f"Pipeline failed: {e}", "error")
        return []
    finally:
        engine_logger.removeHandler(_st_handler)


def render_step_0_trends():
    """Step 0: Trend Detection."""
    def _on_detect():
        st.session_state.pipeline_running = True
        for key in _PIPELINE_STATE_KEYS:
            st.session_state[key] = None if key in ('trend_tree', 'agent_state') else ({} if key == 'engine_metrics' else [])

    # Compact control row
    is_running = st.session_state.get('pipeline_running', False)
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.session_state.audit_sources = st.checkbox(
            "Audit sources first", value=st.session_state.get('audit_sources', False),
            help="Check health of all news sources before fetching. Slower but identifies broken feeds.",
            disabled=is_running,
        )
    with col2:
        max_per_source = st.number_input("Per source", min_value=5, max_value=100, value=st.session_state.get('max_per_source', 10), step=5, label_visibility="collapsed", disabled=is_running)
        st.session_state.max_per_source = max_per_source
    with col3:
        if st.button("🔍 Detect", type="primary", use_container_width=True, disabled=st.session_state.get('pipeline_running', False), on_click=_on_detect):
            progress = st.empty()
            try:
                trends = asyncio.run(_run_trend_pipeline(progress))
                st.session_state.trends = trends
                st.session_state.pipeline_running = False
                st.rerun()
            except Exception as e:
                st.session_state.pipeline_running = False
                st.error(f"Error: {e}")

    # S3: Source Health Dashboard (after audit)
    audit_results = st.session_state.get('source_audit_results', {})
    if audit_results:
        with st.expander("Source Health", expanded=False):
            from app.config import NEWS_SOURCES
            status_icons = {"ok": "🟢", "slow": "🟡", "empty": "🟡", "broken": "🔴", "no_key": "⚪"}
            rows = []
            for sid, result in sorted(audit_results.items(), key=lambda x: x[1].get("status", "")):
                name = (NEWS_SOURCES.get(sid) or {}).get("name", sid)
                status = result.get("status", "unknown")
                icon = status_icons.get(status, "⚪")
                resp_ms = result.get("response_time_ms") or 0
                count = result.get("article_count") or 0
                err = result.get("error") or ""
                rows.append({"": icon, "Source": name, "Status": status, "Articles": count, "ms": int(resp_ms), "Error": err[:60]})
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)

    if st.session_state.trends:
        st.markdown("### Detected Trends")
        tree = st.session_state.get('trend_tree')
        metrics = st.session_state.get('engine_metrics', {})

        # Display pipeline stats bar
        if tree or st.session_state.articles:
            article_counts = metrics.get("article_counts", {})
            articles_in = article_counts.get("input", len(st.session_state.articles))
            articles_unique = article_counts.get("after_semantic_dedup", article_counts.get("after_dedup", articles_in))
            unique_sources = len({a.source_name for a in st.session_state.articles}) if st.session_state.articles else 0
            total_nodes = len(tree.nodes) if tree else 0
            major_trends = len(tree.root_ids) if tree else len(st.session_state.trends)
            total_secs = metrics.get("total_seconds", 0)
            dedup_removed = articles_in - articles_unique
            dedup_pct = (dedup_removed / articles_in * 100) if articles_in > 0 else 0

            # V4: Run quality metrics from OSS pipeline
            run_quality = metrics.get("run_quality", {})
            mean_oss = run_quality.get("mean_oss")
            actionable_rate = run_quality.get("actionable_rate")
            oss_improvement = run_quality.get("oss_improvement")

            oss_stat = ""
            if mean_oss is not None:
                oss_color = "#00ff88" if mean_oss >= 0.5 else "#ffa502" if mean_oss >= 0.3 else "#ff4757"
                oss_stat = f'<div class="pipeline-stat"><div class="pipeline-stat-value" style="color:{oss_color};">{mean_oss:.2f}</div><div class="pipeline-stat-label">Avg OSS</div></div>'
            act_stat = ""
            if actionable_rate is not None:
                act_color = "#00ff88" if actionable_rate >= 0.4 else "#ffa502" if actionable_rate >= 0.2 else "#ff4757"
                act_stat = f'<div class="pipeline-stat"><div class="pipeline-stat-value" style="color:{act_color};">{actionable_rate:.0%}</div><div class="pipeline-stat-label">Actionable</div></div>'

            stats_html = (
                '<div class="pipeline-stats">'
                f'<div class="pipeline-stat"><div class="pipeline-stat-value">{articles_in}</div><div class="pipeline-stat-label">Articles</div></div>'
                f'<div class="pipeline-stat"><div class="pipeline-stat-value">{articles_unique}</div><div class="pipeline-stat-label">Unique ({dedup_pct:.0f}% dedup)</div></div>'
                f'<div class="pipeline-stat"><div class="pipeline-stat-value">{unique_sources}</div><div class="pipeline-stat-label">Sources</div></div>'
                f'<div class="pipeline-stat"><div class="pipeline-stat-value">{major_trends}</div><div class="pipeline-stat-label">Major Trends</div></div>'
                f'<div class="pipeline-stat"><div class="pipeline-stat-value">{total_nodes}</div><div class="pipeline-stat-label">Total Nodes</div></div>'
                + oss_stat + act_stat +
                f'<div class="pipeline-stat"><div class="pipeline-stat-value">{total_secs:.1f}s</div><div class="pipeline-stat-label">Time</div></div>'
                '</div>'
            )
            st.markdown(stats_html, unsafe_allow_html=True)

        # V6: Sector heatmap (expandable)
        if tree and tree.nodes:
            with st.expander("Sector Heatmap", expanded=False):
                render_sector_heatmap(tree)

        # Filters — collapsed by default for clean look
        with st.expander("Filters", expanded=False):
            filter_cols = st.columns(4)
            with filter_cols[0]:
                signal_filter = st.multiselect("Signal", ["STRONG", "WEAK", "NOISE"], default=["STRONG", "WEAK"])
            with filter_cols[1]:
                severity_filter = st.multiselect("Severity", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"])
            with filter_cols[2]:
                min_articles = st.slider("Min Articles", 1, 50, 3)
            with filter_cols[3]:
                depth_filter = st.multiselect("Depth", ["MAJOR", "SUB", "MICRO"], default=["MAJOR", "SUB"])

        def _passes_filter(node) -> bool:
            sig = str(node.signal_strength).upper()
            if signal_filter and sig not in signal_filter:
                return False
            sev = node.severity.value.upper() if hasattr(node.severity, 'value') else str(node.severity).upper()
            if severity_filter and sev not in severity_filter:
                return False
            if node.article_count < min_articles:
                return False
            depth_label = {1: "MAJOR", 2: "SUB", 3: "MICRO"}.get(node.depth, "MAJOR")
            return not depth_filter or depth_label in depth_filter

        # View toggle — clean radio instead of buttons
        if "trend_view_mode" not in st.session_state:
            st.session_state.trend_view_mode = "flow" if FLOW_AVAILABLE else "tree"

        view_options = ["Flow", "List"] if FLOW_AVAILABLE else ["List"]
        current_idx = 0 if st.session_state.trend_view_mode == "flow" and FLOW_AVAILABLE else (1 if FLOW_AVAILABLE else 0)
        view_mode = st.radio("View", view_options, horizontal=True, label_visibility="collapsed", index=current_idx)
        st.session_state.trend_view_mode = "flow" if view_mode == "Flow" else "tree"

        articles_map = {str(a.id): a for a in st.session_state.articles} if st.session_state.articles else {}
        selected_nodes, all_selected_articles = [], []

        if tree and tree.root_ids:
            # Sort: STRONG first, WEAK middle, NOISE at bottom
            _sig_order = {"strong": 0, "weak": 1, "noise": 2}
            valid_root_ids = [rid for rid in tree.root_ids if tree.nodes.get(str(rid)) is not None]
            sorted_root_ids = sorted(
                valid_root_ids,
                key=lambda rid: _sig_order.get(str(tree.nodes[str(rid)].signal_strength).lower(), 3),
            )

            mode = st.session_state.trend_view_mode
            if mode == "flow" and FLOW_AVAILABLE:
                render_flow_tree(tree, articles_map, node_filter=_passes_filter)
                for root_id in sorted_root_ids:
                    root_node = tree.nodes.get(str(root_id))
                    if root_node and _passes_filter(root_node):
                        selected_nodes.append(TrendData(
                            id=str(root_node.id), trend_title=root_node.trend_title, summary=root_node.trend_summary,
                            severity=root_node.severity, industries_affected=[s.value if hasattr(s, 'value') else str(s) for s in root_node.primary_sectors],
                            source_links=[], keywords=root_node.key_entities[:10] if root_node.key_entities else root_node.key_keywords[:10],
                        ))
            else:
                node_index = 0
                for root_id in sorted_root_ids:
                    root_node = tree.nodes.get(str(root_id))
                    if root_node:
                        node_index = render_tree_node(root_node, tree, articles_map, selected_nodes, all_selected_articles, node_index, node_filter=_passes_filter)

        st.session_state.selected_source_articles = all_selected_articles
        st.session_state.selected_trends = selected_nodes

        if selected_nodes:
            _, _, continue_col = st.columns([4, 1, 1])
            with continue_col:
                if st.button(f"Continue → ({len(selected_nodes)} selected)", type="primary", use_container_width=True):
                    st.session_state.current_step = 1
                    st.rerun()


def render_step_1_impacts():
    """Step 1: Impact Analysis."""

    def _on_analyze():
        st.session_state.step_1_running = True

    if not st.session_state.impacts:
        st.info(f"Ready to analyze impacts for {len(st.session_state.selected_trends)} selected trends")
        back_col, _, col_action = st.columns([1, 4, 1])
        with back_col:
            if st.button("← Back", use_container_width=True, key="impact_back_pre"):
                st.session_state.current_step = 0
                st.rerun()
        with col_action:
            if st.button("🔍 Analyze", type="primary", use_container_width=True,
                         disabled=st.session_state.get('step_1_running', False),
                         on_click=_on_analyze):
                with st.spinner("Analyzing impacts..."):
                    add_log("Starting impact analysis...")
                    try:
                        async def run():
                            add_log(f"Analyzing {len(st.session_state.selected_trends)} trends...")
                            _mock_llm = st.session_state.mock_mode and st.session_state.get("mock_llm", True)
                            agent = ImpactAgent(mock_mode=_mock_llm, log_callback=add_log)
                            state = AgentState(trends=_rebuild_list(st.session_state.selected_trends, TrendData))
                            result = await agent.analyze_impacts(state)
                            return result
                        result = asyncio.run(run())
                        st.session_state.impacts = result.impacts
                        if hasattr(result, 'cross_trend_insight') and result.cross_trend_insight:
                            st.session_state.cross_trend_insight = result.cross_trend_insight
                        st.session_state.step_1_running = False
                        add_log(f"Impact analysis complete: {len(st.session_state.impacts)} impacts", "success")
                        st.rerun()
                    except Exception as e:
                        st.session_state.step_1_running = False
                        add_log(f"Impact analysis failed: {e}", "error")
                        st.error(f"Analysis failed: {e}")

    if st.session_state.impacts:
        _conf_tooltip = (
            "Quality-based: 25% evidence grounding (concrete data), "
            "20% specificity (employee ranges/locations), 20% depth (structure), "
            "15% problem concreteness, 10% service fit, 10% cross-validation. "
            "Penalizes vague jargon. Capped at 95%."
        )

        # ── Header row ────────────────────────────────────────────────
        _imp_msg_col, _imp_rerun_col = st.columns([5, 1])
        with _imp_msg_col:
            st.success(f"Analyzed {len(st.session_state.impacts)} trend impacts.")
        with _imp_rerun_col:
            if st.button("🔄 Re-analyze", key="rerun_impacts", use_container_width=True):
                st.session_state.impacts = []
                st.session_state.step_1_running = False
                st.rerun()

        # ── FILTERS — collapsed like News section ─────────────────────
        # Collect filter dimensions from all impacts
        _all_services = sorted({s for imp in st.session_state.impacts for s in (imp.relevant_services or [])})
        _all_sectors = sorted({s for imp in st.session_state.impacts for s in (imp.positive_sectors or []) + (imp.negative_sectors or [])})

        with st.expander("Filters", expanded=False):
            _f_cols = st.columns(3)
            with _f_cols[0]:
                _conf_filter = st.multiselect(
                    "Confidence", ["HIGH (70%+)", "MEDIUM (40-70%)", "LOW (<40%)"],
                    default=["HIGH (70%+)", "MEDIUM (40-70%)", "LOW (<40%)"],
                    key="impact_conf_filter",
                )
            with _f_cols[1]:
                _svc_filter = st.multiselect(
                    "Service", _all_services, default=_all_services,
                    key="impact_svc_filter",
                )
            with _f_cols[2]:
                _sector_filter = st.multiselect(
                    "Sector", _all_sectors, default=[],
                    key="impact_sector_filter",
                    help="Leave empty = show all",
                )

        def _impact_passes_filter(imp) -> bool:
            conf = getattr(imp, 'council_confidence', 0) or 0
            # Confidence filter
            conf_label = "HIGH (70%+)" if conf >= 0.7 else "MEDIUM (40-70%)" if conf >= 0.4 else "LOW (<40%)"
            if _conf_filter and conf_label not in _conf_filter:
                return False
            # Service filter
            if _svc_filter and not any(s in _svc_filter for s in (imp.relevant_services or [])):
                return False
            # Sector filter (empty = show all)
            if _sector_filter:
                imp_sectors = set(imp.positive_sectors or []) | set(imp.negative_sectors or [])
                if not any(s in _sector_filter for s in imp_sectors):
                    return False
            return True

        _filtered_impacts = [imp for imp in st.session_state.impacts if _impact_passes_filter(imp)]
        if len(_filtered_impacts) < len(st.session_state.impacts):
            st.caption(f"Showing {len(_filtered_impacts)} of {len(st.session_state.impacts)} impacts")

        # ── IMPACT CARDS — enriched with pitch + tags ─────────────────
        _cols_per_row = min(3, len(_filtered_impacts)) if _filtered_impacts else 1
        for _row_start in range(0, len(_filtered_impacts), _cols_per_row):
            _row_imps = _filtered_impacts[_row_start:_row_start + _cols_per_row]
            _cols = st.columns(len(_row_imps))
            for _ci, _imp in enumerate(_row_imps):
                with _cols[_ci]:
                    _conf = getattr(_imp, 'council_confidence', 0) or 0
                    _cc = "#2ed573" if _conf >= 0.7 else "#ffa502" if _conf >= 0.4 else "#ff4757"
                    _cp = int(_conf * 100)
                    _svc = escape_for_html(_imp.relevant_services[0]) if _imp.relevant_services else ""
                    _title_short = escape_for_html(_imp.trend_title[:50] + ("..." if len(_imp.trend_title) > 50 else ""))
                    _pitch_short = escape_for_html((_imp.pitch_angle or "")[:120] + ("..." if len(_imp.pitch_angle or "") > 120 else ""))
                    _n_affected = len(_imp.direct_impact) if _imp.direct_impact else 0
                    _n_pain = len(_imp.midsize_pain_points) if _imp.midsize_pain_points else 0
                    _n_projects = len(_imp.consulting_projects) if _imp.consulting_projects else 0

                    # Sector tags HTML
                    _sectors = (_imp.positive_sectors or [])[:3]
                    _sector_tags = ''.join(
                        f'<span style="background:rgba(46,213,115,0.12);color:#2ed573;font-size:9px;'
                        f'padding:1px 5px;border-radius:3px;margin-right:3px;">{escape_for_html(s)}</span>'
                        for s in _sectors
                    )

                    # Service tag
                    _svc_tag_html = ''
                    if _svc:
                        _svc_tag_html = (
                            f'<span style="background:rgba(0,212,255,0.15);color:#00d4ff;font-size:10px;'
                            f'font-weight:600;padding:2px 6px;border-radius:3px;">{_svc}</span>'
                        )

                    _card_html = (
                        f'<div style="background:#111122;border:1px solid #2a2a3e;border-radius:8px;'
                        f'padding:14px;min-height:180px;" title="{_conf_tooltip}">'
                        # Header: confidence + title
                        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
                        f'<span style="background:{_cc};color:#0a0a1a;font-size:11px;font-weight:700;'
                        f'padding:2px 6px;border-radius:3px;">{_cp}%</span>'
                        f'<span style="color:#e0e0e0;font-size:13px;font-weight:600;line-height:1.3;">{_title_short}</span></div>'
                        # Service + sector tags
                        f'<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px;">'
                        f'{_svc_tag_html}{_sector_tags}</div>'
                        # Pitch snippet
                        f'<div style="color:#a0a0b0;font-size:12px;line-height:1.4;margin-bottom:10px;'
                        f'min-height:34px;">{_pitch_short}</div>'
                        # Metrics row
                        f'<div style="display:flex;gap:12px;font-size:11px;color:#666;">'
                        f'<span>{_n_affected} affected</span>'
                        f'<span>{_n_pain} problems</span>'
                        f'<span>{_n_projects} projects</span></div>'
                        # Confidence bar
                        f'<div style="background:#1a1a2e;border-radius:3px;height:4px;margin-top:8px;">'
                        f'<div style="background:{_cc};width:{_cp}%;height:4px;border-radius:3px;"></div></div>'
                        f'</div>'
                    )
                    st.markdown(_card_html, unsafe_allow_html=True)

                    # "View Details" button opens dialog
                    _imp_idx = st.session_state.impacts.index(_imp)
                    if st.button("View Details", key=f"view_imp_{_imp_idx}", use_container_width=True):
                        st.session_state._impact_dialog_idx = _imp_idx
                        st.rerun()

        # ── IMPACT DETAIL DIALOG ──────────────────────────────────────
        if '_impact_dialog_idx' in st.session_state and st.session_state._impact_dialog_idx is not None:
            _dialog_idx = st.session_state._impact_dialog_idx

            @st.dialog("Impact Details", width="large")
            def _show_impact_dialog():
                impact = st.session_state.impacts[_dialog_idx]
                _conf = getattr(impact, 'council_confidence', 0) or 0
                _cc = "#2ed573" if _conf >= 0.7 else "#ffa502" if _conf >= 0.4 else "#ff4757"
                _svc_label = impact.relevant_services[0] if impact.relevant_services else ""

                # ── Hero: Confidence + Service + Pitch ──
                _svc_tag = ""
                if _svc_label:
                    _svc_tag = (
                        f'<span style="background:#00d4ff;color:#0a0a1a;font-size:11px;'
                        f'font-weight:700;padding:2px 8px;border-radius:3px;margin-right:8px;">'
                        f'{escape_for_html(_svc_label)}</span>'
                    )
                _conf_badge = (
                    f'<span style="background:{_cc};color:#0a0a1a;font-size:11px;'
                    f'font-weight:700;padding:2px 6px;border-radius:3px;margin-right:8px;"'
                    f' title="{_conf_tooltip}">{int(_conf*100)}%</span>'
                )

                st.markdown(
                    f'<h3 style="margin-top:0;">{escape_for_html(impact.trend_title)}</h3>',
                    unsafe_allow_html=True)

                if impact.pitch_angle:
                    st.markdown(
                        f'<div style="background:rgba(0,212,255,0.06);border-left:3px solid #00d4ff;'
                        f'padding:10px 14px;border-radius:4px;margin-bottom:14px;">'
                        f'{_conf_badge}{_svc_tag}'
                        f'<span style="color:#00d4ff;font-weight:600;font-size:12px;">PITCH →</span> '
                        f'<span style="color:#e0e0e0;font-size:13px;">{escape_for_html(impact.pitch_angle)}</span>'
                        f'</div>', unsafe_allow_html=True)

                # ── Two-column: Who + Problems | Projects + Services ──
                col_left, col_right = st.columns(2)
                with col_left:
                    if impact.direct_impact:
                        items = ''.join([f'<div class="impact-item">{escape_for_html(item)}</div>' for item in impact.direct_impact[:6]])
                        st.markdown(f'<div class="impact-section"><div class="impact-label impact-label-direct">Who Gets Affected</div>{items}</div>', unsafe_allow_html=True)
                    if impact.midsize_pain_points:
                        items = ''.join([f'<div class="impact-item">{escape_for_html(item)}</div>' for item in impact.midsize_pain_points[:5]])
                        st.markdown(f'<div class="impact-section"><div class="impact-label impact-label-pain">Their Problems</div>{items}</div>', unsafe_allow_html=True)
                with col_right:
                    if impact.consulting_projects:
                        items = ''.join([f'<div class="impact-item">{escape_for_html(item)}</div>' for item in impact.consulting_projects[:5]])
                        st.markdown(f'<div class="impact-section"><div class="impact-label impact-label-consulting">What We Can Do</div>{items}</div>', unsafe_allow_html=True)
                    if impact.relevant_services:
                        items = ''.join([f'<div class="impact-item">{escape_for_html(item)}</div>' for item in impact.relevant_services[:4]])
                        st.markdown(f'<div class="impact-section"><div class="impact-label impact-label-services">Our Services</div>{items}</div>', unsafe_allow_html=True)

                # ── Full Analysis ──
                if impact.detailed_reasoning:
                    with st.expander("Full Analysis", expanded=False):
                        st.markdown(
                            f'<div style="color:#e0e0e0;font-size:14px;line-height:1.7;">'
                            f'{format_llm_text(impact.detailed_reasoning)}</div>',
                            unsafe_allow_html=True)

                # ── AI Activity (if enabled) ──
                if st.session_state.get("show_ai_activity", False):
                    with st.expander("AI Activity", expanded=False):
                        if _conf:
                            st.markdown(
                                f'<div style="margin-bottom:12px;" title="{_conf_tooltip}">'
                                f'<span style="color:{_cc};font-weight:600;cursor:help;">'
                                f'Council Confidence: {_conf:.0%} &#9432;</span>'
                                f'<div style="background:#1a1a2e;border-radius:4px;height:6px;margin-top:4px;">'
                                f'<div style="background:{_cc};width:{int(_conf*100)}%;height:6px;border-radius:4px;"></div>'
                                f'</div></div>', unsafe_allow_html=True)

                        # Tabbed AI activity view
                        perspectives = getattr(impact, 'council_perspectives', [])
                        citations = getattr(impact, 'evidence_citations', [])
                        svc_recs = getattr(impact, 'service_recommendations', [])
                        indirect_reasoning = getattr(impact, 'indirect_impact_reasoning', '') or ''

                        _ai_tab_names = []
                        if perspectives:
                            _ai_tab_names.append("Analysis")
                        if citations:
                            _ai_tab_names.append(f"Evidence ({len(citations)})")
                        if svc_recs:
                            _ai_tab_names.append(f"Services ({len(svc_recs)})")
                        if indirect_reasoning:
                            _ai_tab_names.append("Reasoning")

                        if _ai_tab_names:
                            _ai_tabs = st.tabs(_ai_tab_names)
                            _tab_idx = 0

                            if perspectives:
                                with _ai_tabs[_tab_idx]:
                                    for p in perspectives:
                                        role = p.get('role', 'analyst').replace('_', ' ').title()
                                        p_conf = p.get('confidence', 0)
                                        analysis = p.get('analysis', '')
                                        st.markdown(f"**{role}** — confidence: {p_conf:.0%}")
                                        if analysis:
                                            st.markdown(
                                                f'<div style="background:#0d0d1a;border-left:2px solid #2a2a3e;'
                                                f'padding:8px 12px;margin-bottom:6px;border-radius:4px;'
                                                f'font-size:13px;line-height:1.6;color:#c0c0d0;">'
                                                f'{format_llm_text(analysis)}</div>',
                                                unsafe_allow_html=True)
                                        findings = p.get('key_findings', [])
                                        if findings:
                                            st.markdown("**Key Findings:**")
                                            for finding in findings:
                                                st.markdown(f"- {finding}")
                                _tab_idx += 1

                            if citations:
                                with _ai_tabs[_tab_idx]:
                                    for _ci, c in enumerate(citations, 1):
                                        if not c or not c.strip():
                                            continue
                                        st.markdown(
                                            f'<div style="background:#0d0d1a;padding:6px 10px;'
                                            f'border-radius:4px;margin-bottom:4px;font-size:12px;color:#a0a0b0;">'
                                            f'<span style="color:#00d4ff;font-weight:600;">#{_ci}</span> '
                                            f'{format_llm_text(c)}</div>',
                                            unsafe_allow_html=True)
                                    if not any(c and c.strip() for c in citations):
                                        st.caption("No evidence citations were produced for this analysis.")
                                _tab_idx += 1

                            if svc_recs:
                                with _ai_tabs[_tab_idx]:
                                    for r in svc_recs:
                                        if isinstance(r, dict):
                                            urgency = r.get('urgency', 'medium')
                                            _u_color = {"high": "#ff4757", "medium": "#ffa502", "low": "#2ed573"}.get(urgency, "#888")
                                            st.markdown(
                                                f'<div style="background:#0d0d1a;padding:8px 12px;'
                                                f'border-left:3px solid {_u_color};border-radius:4px;margin-bottom:6px;">'
                                                f'<div style="font-size:13px;font-weight:600;color:#e0e0e0;">'
                                                f'{escape_for_html(r.get("service", ""))}: '
                                                f'{escape_for_html(r.get("offering", ""))}</div>'
                                                f'<div style="font-size:12px;color:#a0a0b0;margin-top:2px;">'
                                                f'{escape_for_html(r.get("justification", ""))}</div>'
                                                f'<div style="font-size:10px;color:{_u_color};margin-top:4px;">'
                                                f'Urgency: {urgency.upper()}</div>'
                                                f'</div>', unsafe_allow_html=True)
                                _tab_idx += 1

                            if indirect_reasoning:
                                with _ai_tabs[_tab_idx]:
                                    st.markdown(
                                        f'<div style="background:#0d0d1a;border:1px solid #2a2a3e;'
                                        f'border-radius:6px;padding:12px;'
                                        f'font-size:13px;line-height:1.6;color:#c0c0d0;">'
                                        f'{format_llm_text(indirect_reasoning)}</div>',
                                        unsafe_allow_html=True)

                # ── Feedback buttons ──
                fb_cols = st.columns([1, 1, 1, 3])
                trend_id = getattr(impact, 'trend_id', '') or impact.trend_title
                _fb_signals = {}
                _fb_tree = st.session_state.get('trend_tree')
                if _fb_tree and hasattr(_fb_tree, 'nodes'):
                    _fb_node = _fb_tree.nodes.get(str(trend_id))
                    if _fb_node and _fb_node.signals:
                        _fb_signals = {
                            k: v for k, v in _fb_node.signals.items()
                            if isinstance(v, (int, float))
                        }
                with fb_cols[0]:
                    if st.button("Good trend", key=f"dlg_fb_good_{trend_id}"):
                        from app.tools.feedback import save_feedback
                        save_feedback("trend", str(trend_id), "good_trend",
                                      signals=_fb_signals,
                                      metadata={"title": impact.trend_title})
                        st.toast("Feedback saved: Good trend")
                with fb_cols[1]:
                    if st.button("Bad trend", key=f"dlg_fb_bad_{trend_id}"):
                        from app.tools.feedback import save_feedback
                        save_feedback("trend", str(trend_id), "bad_trend",
                                      signals=_fb_signals,
                                      metadata={"title": impact.trend_title})
                        st.toast("Feedback saved: Bad trend")
                with fb_cols[2]:
                    if st.button("Already knew", key=f"dlg_fb_known_{trend_id}"):
                        from app.tools.feedback import save_feedback
                        save_feedback("trend", str(trend_id), "already_knew",
                                      signals=_fb_signals,
                                      metadata={"title": impact.trend_title})
                        st.toast("Feedback saved: Already knew")

                # Close button inside dialog
                st.markdown("---")
                if st.button("Close", key="dlg_close", use_container_width=True):
                    st.session_state._impact_dialog_idx = None
                    st.rerun()

            _show_impact_dialog()

        # ── Cross-trend compound impact synthesis ─────────────────────
        cross_insight = st.session_state.get('cross_trend_insight')
        if cross_insight and isinstance(cross_insight, dict):
            st.markdown("---")
            st.markdown("### Compound Impact Analysis")
            if cross_insight.get('cross_trend_insight'):
                st.markdown(cross_insight['cross_trend_insight'])
            compound = cross_insight.get('compound_impacts', [])
            if compound:
                for ci in compound[:4]:
                    with st.expander(f"{ci.get('company_type', 'Unknown')}", expanded=False):
                        st.markdown(f"**Affected by trends:** {ci.get('affected_by_trends', [])}")
                        st.markdown(f"**Compound challenge:** {ci.get('compound_challenge', '')}")
                        st.success(f"**Opportunity:** {ci.get('consulting_opportunity', '')}")
            mega = cross_insight.get('mega_opportunity', '')
            if mega:
                st.success(f"**Best Combined Pitch:** {mega}")

        back_col, _, continue_col = st.columns([1, 4, 1])
        with back_col:
            if st.button("← Back", use_container_width=True):
                st.session_state.step_1_running = False
                st.session_state.current_step = 0
                st.rerun()
        with continue_col:
            if st.button("Continue →", type="primary", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()


def render_step_2_companies():
    """Step 2: Company Discovery."""

    def _on_find_companies():
        st.session_state.step_2_running = True

    # Quality gate: filter out low-confidence impacts before expensive company search
    from app.config import get_settings as _gs_gate
    _gate_threshold = _gs_gate().min_trend_confidence_for_agents
    all_impacts = _rebuild_list(st.session_state.impacts or [], ImpactAnalysis)
    viable_impacts = [imp for imp in all_impacts if getattr(imp, 'council_confidence', 1.0) >= _gate_threshold]
    skipped = len(all_impacts) - len(viable_impacts)
    if skipped > 0:
        st.caption(f"Quality gate: {skipped} low-confidence impact(s) filtered (threshold={_gate_threshold})")

    if not st.session_state.companies:
        st.info(f"Ready to find target companies for {len(viable_impacts)} impact(s)")
        back_col, _, col_action = st.columns([1, 4, 1])
        with back_col:
            if st.button("← Back", use_container_width=True, key="company_back_pre"):
                st.session_state.current_step = 1
                st.rerun()
        with col_action:
            if st.button("🔍 Find", type="primary", use_container_width=True,
                         disabled=st.session_state.get('step_2_running', False),
                         on_click=_on_find_companies):
                with st.spinner("Finding target companies..."):
                    add_log("Finding target companies...")
                    try:
                        async def run():
                            _mock_search = st.session_state.mock_mode and st.session_state.get("mock_search", True)
                            agent = CompanyAgent(mock_mode=_mock_search, log_callback=add_log)
                            state = AgentState(trends=_rebuild_list(st.session_state.selected_trends, TrendData), impacts=viable_impacts)
                            result = await agent.find_companies(state)
                            return result.companies
                        st.session_state.companies = asyncio.run(run())
                        st.session_state.step_2_running = False
                        add_log(f"Found {len(st.session_state.companies)} companies", "success")
                        st.rerun()
                    except Exception as e:
                        st.session_state.step_2_running = False
                        add_log(f"Company search failed: {e}", "error")
                        st.error(f"Search failed: {e}")

    if st.session_state.companies:
        _co_msg_col, _co_rerun_col = st.columns([5, 1])
        with _co_msg_col:
            st.success(f"Found {len(st.session_state.companies)} target companies. Review and select below.")
        with _co_rerun_col:
            if st.button("🔄 Re-search", key="rerun_companies", use_container_width=True):
                st.session_state.companies = []
                st.session_state.step_2_running = False
                st.rerun()

        # AI Activity summary for company discovery
        if st.session_state.get("show_ai_activity", False):
            with st.expander("AI Activity — Company Discovery", expanded=False):
                companies_list = st.session_state.companies
                ner_count = sum(1 for c in companies_list if c.verification_source == "ner_match")
                wiki_count = sum(1 for c in companies_list if c.verification_source == "wikipedia")
                unverified_count = sum(1 for c in companies_list if c.verification_source == "unverified")
                st.markdown(f"**V7 Verification:** {ner_count} NER-matched, {wiki_count} Wikipedia-verified, {unverified_count} unverified")
                # Show confidence distribution
                confs = [c.verification_confidence for c in companies_list if c.verification_confidence]
                if confs:
                    avg_conf = sum(confs) / len(confs)
                    st.markdown(f"**Avg verification confidence:** {avg_conf:.0%}")
                # Show size distribution
                size_dist = {}
                for c in companies_list:
                    sz = str(c.company_size.value if hasattr(c.company_size, 'value') else c.company_size)
                    size_dist[sz] = size_dist.get(sz, 0) + 1
                if size_dist:
                    st.markdown(f"**Company sizes:** {', '.join(f'{k}: {v}' for k, v in sorted(size_dist.items()))}")
        # Build trend_id → trend_title lookup for badges
        _trend_title_map = {}
        for t in st.session_state.get('selected_trends', []):
            tid = getattr(t, 'trend_id', '') or getattr(t, 'id', '')
            ttitle = getattr(t, 'trend_title', '') or getattr(t, 'title', '')
            if tid:
                _trend_title_map[tid] = ttitle

        selected = []
        for company in st.session_state.companies:
            col_check, col_card = st.columns([0.05, 0.95])
            with col_check:
                if st.checkbox("", key=f"co_{company.id}", value=True, label_visibility="collapsed"):
                    selected.append(company)
            with col_card:
                # NER verification badge
                verified_badge = ''
                if company.ner_verified:
                    src = escape_for_html(company.verification_source or 'verified')
                    verified_badge = f'<span style="background: #2ed57333; color: #2ed573; padding: 2px 8px; border-radius: 10px; font-size: 10px; margin-left: 8px;">✓ {src}</span>'
                elif company.verification_source == "unverified":
                    verified_badge = '<span style="background: #ffa50233; color: #ffa502; padding: 2px 8px; border-radius: 10px; font-size: 10px; margin-left: 8px;">Unverified</span>'

                # Company size
                size_str = company.company_size
                if hasattr(size_str, 'value'):
                    size_str = size_str.value
                size_badge = f'<span style="background: rgba(255,255,255,0.08); color: #aaa; padding: 2px 8px; border-radius: 10px; font-size: 10px;">{escape_for_html(str(size_str).upper())}</span>'

                # Industry badge
                industry_badge = f'<span style="background: rgba(0,255,136,0.1); color: #00ff88; padding: 2px 8px; border-radius: 10px; font-size: 10px;">{escape_for_html(company.industry or "Other")}</span>'

                # Trend source badge
                trend_badge = ''
                if company.trend_id and company.trend_id in _trend_title_map:
                    trend_name = escape_for_html(_trend_title_map[company.trend_id][:40])
                    trend_badge = f'<div style="margin-top: 4px;"><span style="background: rgba(0,212,255,0.1); color: #00d4ff; padding: 2px 8px; border-radius: 10px; font-size: 10px;">📌 {trend_name}</span></div>'

                # Website link
                website_html = ''
                if company.website:
                    safe_url = escape_for_html(company.website)
                    website_html = f'<a href="{safe_url}" target="_blank" style="color: #00d4ff; font-size: 11px; text-decoration: none;">🔗 Website</a>'

                desc = escape_for_html(company.description or company.reason_relevant or '')
                name = escape_for_html(company.company_name)

                website_section = f'<div style="margin-top: 4px;">{website_html}</div>' if website_html else ''
                card_html = (
                    '<div class="company-card">'
                    '<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px;">'
                    '<div>'
                    f'<span style="font-size: 15px; font-weight: 600; color: #e0e0e0;">{name}</span>'
                    f'{verified_badge}'
                    '</div>'
                    '<div style="display: flex; gap: 6px; align-items: center;">'
                    f'{size_badge} {industry_badge}'
                    '</div>'
                    '</div>'
                    f'<p style="color: #999; margin: 8px 0 6px 0; font-size: 13px; line-height: 1.4;">{desc}</p>'
                    f'{trend_badge}'
                    f'{website_section}'
                    '</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

        st.session_state.selected_companies = selected

        back_col, _, continue_col = st.columns([1, 4, 1])
        with back_col:
            if st.button("← Back", use_container_width=True):
                st.session_state.step_2_running = False
                st.session_state.current_step = 1
                st.rerun()
        with continue_col:
            if st.button(f"Continue → ({len(selected)})", type="primary", use_container_width=True):
                st.session_state.current_step = 3
                st.rerun()


def render_step_3_contacts():
    """Step 3: Contact Finding."""

    def _on_find_contacts():
        st.session_state.step_3_running = True

    if not st.session_state.contacts:
        st.info(f"Ready to find decision makers for {len(st.session_state.selected_companies)} companies")
        back_col, _, col_action = st.columns([1, 4, 1])
        with back_col:
            if st.button("← Back", use_container_width=True, key="contact_back_pre"):
                st.session_state.current_step = 2
                st.rerun()
        with col_action:
            if st.button("🔍 Find", type="primary", use_container_width=True,
                         disabled=st.session_state.get('step_3_running', False),
                         on_click=_on_find_contacts):
                with st.spinner("Finding decision makers..."):
                    add_log(f"Finding contacts for {len(st.session_state.selected_companies)} companies...")
                    try:
                        async def run():
                            _mock_llm = st.session_state.mock_mode and st.session_state.get("mock_llm", True)
                            agent = ContactAgent(mock_mode=_mock_llm)
                            state = AgentState(trends=_rebuild_list(st.session_state.selected_trends, TrendData),
                                               impacts=_rebuild_list(st.session_state.impacts, ImpactAnalysis),
                                               companies=_rebuild_list(st.session_state.selected_companies, CompanyData))
                            result = await agent.find_contacts(state)
                            return result.contacts
                        st.session_state.contacts = asyncio.run(run())
                        st.session_state.step_3_running = False
                        add_log(f"Found {len(st.session_state.contacts)} contacts", "success")
                        st.rerun()
                    except Exception as e:
                        st.session_state.step_3_running = False
                        add_log(f"Contact search failed: {e}", "error")
                        st.error(f"Search failed: {e}")

    if st.session_state.contacts:
        st.success(f"Found {len(st.session_state.contacts)} decision makers. Review and select below.")
        selected = []
        for contact in st.session_state.contacts:
            col_check, col_card = st.columns([0.05, 0.95])
            with col_check:
                if st.checkbox("", key=f"ct_{contact.id}", value=True, label_visibility="collapsed"):
                    selected.append(contact)
            with col_card:
                name = escape_for_html(contact.person_name)
                role = escape_for_html(contact.role)
                company = escape_for_html(contact.company_name)
                email = escape_for_html(contact.email or "No email found")

                # Email confidence
                conf = getattr(contact, 'email_confidence', 0) or 0
                conf_color = "#2ed573" if conf >= 80 else "#ffa502" if conf >= 50 else "#ff4757"
                conf_html = f'<span style="color: {conf_color}; font-size: 11px;">{conf}%</span>' if conf > 0 else ''

                # LinkedIn
                linkedin_html = ''
                if contact.linkedin_url:
                    safe_url = escape_for_html(contact.linkedin_url)
                    linkedin_html = f'<a href="{safe_url}" target="_blank" style="color: #0077b5; font-size: 11px; text-decoration: none;">LinkedIn ↗</a>'

                # Email source badge
                source_html = ''
                if getattr(contact, 'email_source', ''):
                    source_html = f'<span style="background: rgba(255,255,255,0.08); color: #aaa; padding: 2px 6px; border-radius: 10px; font-size: 10px;">via {escape_for_html(contact.email_source)}</span>'

                card_html = (
                    '<div class="contact-card">'
                    '<div style="display: flex; justify-content: space-between; align-items: flex-start;">'
                    '<div>'
                    f'<span style="font-size: 15px; font-weight: 600; color: #e0e0e0;">{name}</span>'
                    f'<span style="color: #888; font-size: 13px; margin-left: 8px;">{role}</span>'
                    '</div>'
                    f'<span style="color: #aaa; font-size: 12px;">{company}</span>'
                    '</div>'
                    '<div style="display: flex; gap: 12px; align-items: center; margin-top: 8px; flex-wrap: wrap;">'
                    f'<span style="color: #ccc; font-size: 12px;">{email}</span>'
                    f'{conf_html} {source_html} {linkedin_html}'
                    '</div>'
                    '</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

        st.session_state.selected_contacts = selected

        back_col, _, continue_col = st.columns([1, 4, 1])
        with back_col:
            if st.button("← Back", use_container_width=True):
                st.session_state.step_3_running = False
                st.session_state.current_step = 2
                st.rerun()
        with continue_col:
            if st.button(f"Continue → ({len(selected)})", type="primary", use_container_width=True):
                st.session_state.current_step = 4
                st.rerun()


def render_step_4_emails():
    """Step 4: Email Generation."""

    def _on_generate():
        st.session_state.step_4_running = True

    if not st.session_state.outreach_emails:
        st.info(f"Ready to generate personalized pitches for {len(st.session_state.selected_contacts)} contacts")
        back_col, _, action_col = st.columns([1, 4, 1])
        with back_col:
            if st.button("← Back", use_container_width=True, key="email_back_pre"):
                st.session_state.current_step = 3
                st.rerun()
        with action_col:
            st.button("✉️ Generate", type="primary", use_container_width=True,
                      disabled=st.session_state.get('step_4_running', False),
                      on_click=_on_generate, key="email_generate_btn")

        if st.session_state.get('step_4_running', False):
            with st.spinner("Generating personalized pitches..."):
                add_log(f"Generating emails for {len(st.session_state.selected_contacts)} contacts...")
                try:
                    async def run():
                        _mock_llm = st.session_state.mock_mode and st.session_state.get("mock_llm", True)
                        agent = EmailAgent(mock_mode=_mock_llm)
                        state = AgentState(trends=_rebuild_list(st.session_state.selected_trends, TrendData),
                                           impacts=_rebuild_list(st.session_state.impacts, ImpactAnalysis),
                                           companies=_rebuild_list(st.session_state.selected_companies, CompanyData),
                                           contacts=_rebuild_list(st.session_state.selected_contacts, ContactData))
                        result = await agent.process_emails(state)
                        return result.outreach_emails, result.contacts
                    emails, updated_contacts = asyncio.run(run())
                    st.session_state.outreach_emails = emails
                    st.session_state.contacts = updated_contacts
                    st.session_state.step_4_running = False
                    add_log(f"Generated {len(emails)} emails", "success")
                    st.rerun()
                except Exception as e:
                    st.session_state.step_4_running = False
                    add_log(f"Email generation failed: {e}", "error")
                    st.error(f"Generation failed: {e}")

    if st.session_state.outreach_emails:
        # Build scored leads
        trends = _rebuild_list(st.session_state.selected_trends, TrendData)
        impacts = _rebuild_list(st.session_state.impacts, ImpactAnalysis)
        companies = _rebuild_list(st.session_state.selected_companies, CompanyData)
        contacts = _rebuild_list(st.session_state.contacts, ContactData)

        trend_map = {t.trend_title: t for t in trends}
        impact_map = {i.trend_title: i for i in impacts}
        company_map = {c.company_name: c for c in companies}
        contact_map = {(c.company_name, c.person_name): c for c in contacts}

        scored_leads = []
        emails_rebuilt = _rebuild_list(st.session_state.outreach_emails, OutreachEmail)
        for email in emails_rebuilt:
            trend = trend_map.get(email.trend_title, trends[0] if trends else None)
            impact = impact_map.get(email.trend_title, impacts[0] if impacts else None)
            company = company_map.get(email.company_name)
            contact = contact_map.get((email.company_name, email.person_name))
            if trend and impact and company and contact:
                lead = LeadRecord(trend=trend, impact=impact, company=company, contact=contact, outreach=email)
                lead.compute_score()
                scored_leads.append((lead.lead_score, email, lead))
            else:
                scored_leads.append((0.0, email, None))

        scored_leads.sort(key=lambda x: x[0], reverse=True)

        st.success(f"Generated {len(scored_leads)} personalized pitches, sorted by lead score.")

        for score, email, lead in scored_leads:
            score_color = "#2ed573" if score >= 60 else "#ffa502" if score >= 40 else "#ff4757"
            with st.expander(f"[{score:.0f}/100] {email.person_name} @ {email.company_name}", expanded=True):
                # Styled email card
                safe_email = escape_for_html(email.email or '')
                safe_subject = escape_for_html(email.subject)
                safe_body = escape_for_html(email.body)
                safe_trend = escape_for_html(email.trend_title[:50])
                conf = email.email_confidence

                email_html = (
                    '<div class="email-card">'
                    '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">'
                    '<div style="display: flex; gap: 10px; align-items: center;">'
                    f'<span style="font-size: 22px; font-weight: 700; color: {score_color};">{score:.0f}</span>'
                    '<span style="color: #666; font-size: 12px;">/ 100</span>'
                    f'<span style="color: #ccc; font-size: 12px;">To: {safe_email}</span>'
                    f'<span style="background: rgba(255,255,255,0.08); padding: 2px 8px; border-radius: 10px; font-size: 10px; color: #aaa;">Confidence: {conf}%</span>'
                    '</div>'
                    f'<span style="color: #555; font-size: 11px;">{safe_trend}</span>'
                    '</div>'
                    '<div style="background: rgba(255,204,0,0.08); border-radius: 8px; padding: 10px 14px; margin-bottom: 10px;">'
                    '<div style="color: #888; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Subject</div>'
                    f'<div style="color: #e0e0e0; font-size: 14px; font-weight: 500;">{safe_subject}</div>'
                    '</div>'
                    f'<div style="background: rgba(255,255,255,0.03); border-radius: 8px; padding: 14px; white-space: pre-wrap; font-size: 13px; color: #ccc; line-height: 1.6;">{safe_body}</div>'
                    '</div>'
                )
                st.markdown(email_html, unsafe_allow_html=True)

                # Lead feedback buttons — enriched metadata for company relevance bandit
                lead_fb_cols = st.columns([1, 1, 1, 3])
                lead_id = f"{email.company_name}_{email.person_name}"
                _fb_company_size = company.company_size if company else "mid"
                if hasattr(_fb_company_size, "value"):
                    _fb_company_size = _fb_company_size.value
                _fb_event_type = getattr(trend, "validated_event_type", "") or "general" if trend else "general"
                _fb_severity = trend.severity if trend else "medium"
                if hasattr(_fb_severity, "value"):
                    _fb_severity = _fb_severity.value
                _fb_industry = company.industry if company else ""
                _fb_meta = {
                    "company": email.company_name,
                    "contact": email.person_name,
                    "trend": email.trend_title,
                    "company_size": str(_fb_company_size),
                    "company_industry": _fb_industry,
                    "event_type": _fb_event_type,
                    "trend_severity": str(_fb_severity),
                }
                with lead_fb_cols[0]:
                    if st.button("Would email", key=f"lfb_yes_{lead_id}"):
                        from app.tools.feedback import save_feedback
                        save_feedback("lead", lead_id, "would_email",
                                      signals={"lead_score": score},
                                      metadata=_fb_meta)
                        st.toast("Feedback saved: Would email")
                with lead_fb_cols[1]:
                    if st.button("Maybe", key=f"lfb_maybe_{lead_id}"):
                        from app.tools.feedback import save_feedback
                        save_feedback("lead", lead_id, "maybe",
                                      signals={"lead_score": score},
                                      metadata=_fb_meta)
                        st.toast("Feedback saved: Maybe")
                with lead_fb_cols[2]:
                    if st.button("Bad lead", key=f"lfb_bad_{lead_id}"):
                        from app.tools.feedback import save_feedback
                        save_feedback("lead", lead_id, "bad_lead",
                                      signals={"lead_score": score},
                                      metadata=_fb_meta)
                        st.toast("Feedback saved: Bad lead")

                # Copy-friendly version
                with st.expander("Copy-friendly version", expanded=False):
                    st.code(f"To: {email.email}\nSubject: {email.subject}\n\n{email.body}", language=None)

        # Export + navigation row
        back_col, col_json, col_csv, col_new = st.columns([1, 1, 1, 1])
        with back_col:
            if st.button("← Back", use_container_width=True):
                st.session_state.step_4_running = False
                st.session_state.current_step = 3
                st.rerun()
        with col_json:
            export_data = {
                "generated_at": datetime.now().isoformat(),
                "leads": [
                    {
                        "lead_score": score,
                        "email": e.model_dump(),
                        "company": l.company.model_dump() if l else {},
                        "contact": l.contact.model_dump() if l else {},
                    }
                    for score, e, l in scored_leads
                ],
                "trends": [t.model_dump() for t in trends],
            }
            st.download_button("JSON", json.dumps(export_data, indent=2, default=str),
                               f"leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "application/json",
                               use_container_width=True, key="export_json")
        with col_csv:
            csv_data = [
                {"Score": score, "Company": e.company_name, "Contact": e.person_name, "Email": e.email, "Subject": e.subject}
                for score, e, _ in scored_leads
            ]
            st.download_button("CSV", pd.DataFrame(csv_data).to_csv(index=False),
                               f"leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv",
                               use_container_width=True, key="export_csv")
        with col_new:
            if st.button("New Run", type="primary", use_container_width=True):
                st.session_state.current_step = 0
                for key in _PIPELINE_STATE_KEYS:
                    if key in ('trend_tree', 'agent_state'):
                        st.session_state[key] = None
                    elif key == 'engine_metrics':
                        st.session_state[key] = {}
                    else:
                        st.session_state[key] = []
                st.rerun()


def main():
    """Main application."""
    # Check and display API availability on first load
    if 'api_status_checked' not in st.session_state:
        try:
            checker = APIChecker()
            import asyncio
            # Use asyncio.run() via existing event loop (nest_asyncio allows this)
            def get_status():
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(checker.get_status_summary())
            
            def get_issues():
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(checker.get_critical_issues())
            
            status_summary = get_status()
            issues = get_issues()
            
            st.session_state.api_status_summary = status_summary
            st.session_state.api_issues = issues
            st.session_state.api_status_checked = True
        except Exception as e:
            import logging as _logging_err
            _logger = _logging_err.getLogger(__name__)
            _logger.warning(f"Could not check API status: {e}")
            st.session_state.api_issues = [f"⚠️ Could not check API configuration: {e}"]
            st.session_state.api_status_checked = True
    
    # Show critical issues if any
    if st.session_state.get('api_issues'):
        for issue in st.session_state.api_issues:
            try:
                if issue.startswith("❌"):
                    st.error(issue)
                elif issue.startswith("⚠️"):
                    st.warning(issue)
            except Exception:
                st.caption(issue)
    
    init_session_state()

    render_sidebar()

    # Step indicator bar (original HTML styling — visual only)
    step_names = ["News", "Impacts", "Companies", "Contacts", "Emails"]
    current = st.session_state.current_step
    items = []
    for i, name in enumerate(step_names):
        if i < current:
            cls = "completed"
            icon = "&#10003;"
        elif i == current:
            cls = "active"
            icon = str(i + 1)
        else:
            cls = "pending"
            icon = str(i + 1)
        items.append(f'<div class="step-bar-item {cls}"><span>{icon}</span> {name}</div>')
        if i < len(step_names) - 1:
            items.append('<div class="step-bar-separator">&#x2192;</div>')
    st.markdown(f'<div class="step-bar">{"".join(items)}</div>', unsafe_allow_html=True)

    global _log_area
    step_container = st.container()
    _log_area = st.empty()
    _refresh_log_area()

    step_renderers = {0: render_step_0_trends, 1: render_step_1_impacts, 2: render_step_2_companies, 3: render_step_3_contacts, 4: render_step_4_emails}
    renderer = step_renderers.get(st.session_state.current_step)
    if renderer:
        with step_container:
            renderer()



if __name__ == "__main__":
    main()
