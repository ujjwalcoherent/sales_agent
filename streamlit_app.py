"""
India Trend Lead Generation Agent - Streamlit Interactive Dashboard
Human-in-the-loop workflow for testing and controlling the agent pipeline.
"""

import asyncio
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

# Import shared components
from app.shared.sidebar import render_sidebar
from app.shared.styles import apply_custom_styles
from app.shared.helpers import _rebuild_list, escape_for_html
from app.shared.visualizations import FLOW_AVAILABLE, render_tree_node, render_flow_tree, render_sector_heatmap

# Import agents and schemas
from app.agents.company_agent import CompanyAgent
from app.agents.contact_agent import ContactAgent
from app.agents.email_agent import EmailAgent
from app.agents.impact_agent import ImpactAgent
from app.config import DEFAULT_ACTIVE_SOURCES, get_domestic_source_ids, get_settings
from app.schemas import AgentState, CompanyData, ContactData, ImpactAnalysis, LeadRecord, TrendData
from app.tools.rss_tool import RSSTool
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
            for entry in reversed(st.session_state.logs[-20:]):
                color = {"info": "#888", "success": "#2ed573", "warning": "#ffa502", "error": "#ff4757"}.get(entry['level'], "#888")
                st.markdown(f'<span style="color: #555; font-size: 11px;">{entry["time"]}</span> <span style="color: {color};">{entry["icon"]}</span> <span style="color: #aaa; font-size: 12px;">{entry["message"]}</span>', unsafe_allow_html=True)
            # Debug log download
            log_path = Path("trend_engine_debug.log")
            if log_path.exists():
                log_content = log_path.read_text(encoding="utf-8", errors="replace")
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.download_button("Download Debug Log", log_content, file_name="trend_engine_debug.log", mime="text/plain")
                with col2:
                    if st.checkbox("View Debug Log", value=False):
                        st.code(log_content[-5000:] if len(log_content) > 5000 else log_content, language="log")


def add_log(message: str, level: str = "info"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    icons = {"info": "●", "success": "✓", "warning": "⚠", "error": "✗"}
    st.session_state.logs.append({"time": timestamp, "level": level, "icon": icons.get(level, "●"), "message": message})
    _refresh_log_area()


def _to_trend_data(major_trends: list) -> list:
    return [
        TrendData(
            id=str(mt.id), trend_title=mt.trend_title, summary=mt.trend_summary, severity=mt.severity,
            industries_affected=[s.value if hasattr(s, 'value') else str(s) for s in mt.primary_sectors],
            source_links=[], keywords=mt.key_entities[:10] if mt.key_entities else mt.key_keywords[:10],
        ) for mt in major_trends
    ]


async def _run_trend_pipeline(progress) -> list:
    """Run the trend detection pipeline."""
    import time as _time
    mock_mode = st.session_state.mock_mode
    settings = get_settings()

    def log(msg, level="info"):
        add_log(msg, level)
        progress.info(f"{msg}")

    max_per_source = st.session_state.get('max_per_source', 10)
    rss_tool = RSSTool(mock_mode=mock_mode)

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
        dedup_threshold=settings.dedup_threshold, dedup_shingle_size=settings.dedup_shingle_size,
        semantic_dedup_threshold=settings.semantic_dedup_threshold,
        spacy_model=settings.spacy_model, umap_n_components=settings.umap_n_components,
        umap_n_neighbors=settings.umap_n_neighbors, umap_min_dist=settings.umap_min_dist,
        umap_metric=settings.umap_metric, min_cluster_size=settings.hdbscan_min_cluster_size,
        min_samples=settings.hdbscan_min_samples, cluster_selection_method=settings.hdbscan_cluster_selection,
        max_depth=settings.engine_max_depth, max_concurrent_llm=settings.engine_max_concurrent_llm, mock_mode=mock_mode,
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
        tree = await engine.run(articles)

        st.session_state.trend_tree = tree
        st.session_state.engine_metrics = engine.metrics
        major_trends = tree.to_major_trends()
        st.session_state.major_trends = major_trends

        total_time = engine.metrics.get("total_seconds", 0)
        log(f"Pipeline complete: {len(tree.nodes)} trends in {total_time:.1f}s", "success")
        return _to_trend_data(major_trends) if major_trends else []

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

            stats_html = (
                '<div class="pipeline-stats">'
                f'<div class="pipeline-stat"><div class="pipeline-stat-value">{articles_in}</div><div class="pipeline-stat-label">Articles</div></div>'
                f'<div class="pipeline-stat"><div class="pipeline-stat-value">{articles_unique}</div><div class="pipeline-stat-label">Unique ({dedup_pct:.0f}% dedup)</div></div>'
                f'<div class="pipeline-stat"><div class="pipeline-stat-value">{unique_sources}</div><div class="pipeline-stat-label">Sources</div></div>'
                f'<div class="pipeline-stat"><div class="pipeline-stat-value">{major_trends}</div><div class="pipeline-stat-label">Major Trends</div></div>'
                f'<div class="pipeline-stat"><div class="pipeline-stat-value">{total_nodes}</div><div class="pipeline-stat-label">Total Nodes</div></div>'
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
        _, col_action = st.columns([5, 1])
        with col_action:
            if st.button("🔍 Analyze", type="primary", use_container_width=True,
                         disabled=st.session_state.get('step_1_running', False),
                         on_click=_on_analyze):
                with st.spinner("Analyzing impacts..."):
                    add_log("Starting impact analysis...")
                    try:
                        async def run():
                            add_log(f"Analyzing {len(st.session_state.selected_trends)} trends...")
                            agent = ImpactAgent(mock_mode=st.session_state.mock_mode)
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
        st.success(f"Analyzed {len(st.session_state.impacts)} trend impacts. Review below.")
        for impact in st.session_state.impacts:
            with st.expander(f"{impact.trend_title}", expanded=True):
                col_left, col_right = st.columns(2)
                with col_left:
                    if impact.direct_impact:
                        items = ''.join([f'<div class="impact-item">{escape_for_html(item)}</div>' for item in impact.direct_impact[:6]])
                        st.markdown(f'<div class="impact-section"><div class="impact-label impact-label-direct">Direct Impact</div>{items}</div>', unsafe_allow_html=True)
                    if impact.indirect_impact:
                        items = ''.join([f'<div class="impact-item">{escape_for_html(item)}</div>' for item in impact.indirect_impact[:4]])
                        st.markdown(f'<div class="impact-section"><div class="impact-label impact-label-indirect">Indirect Impact</div>{items}</div>', unsafe_allow_html=True)
                    if impact.midsize_pain_points:
                        items = ''.join([f'<div class="impact-item">{escape_for_html(item)}</div>' for item in impact.midsize_pain_points[:4]])
                        st.markdown(f'<div class="impact-section"><div class="impact-label impact-label-pain">Pain Points</div>{items}</div>', unsafe_allow_html=True)
                with col_right:
                    if impact.consulting_projects:
                        items = ''.join([f'<div class="impact-item">{escape_for_html(item)}</div>' for item in impact.consulting_projects[:5]])
                        st.markdown(f'<div class="impact-section"><div class="impact-label impact-label-consulting">Consulting Projects</div>{items}</div>', unsafe_allow_html=True)
                    if impact.additional_verticals:
                        items = ''.join([f'<div class="impact-item">{escape_for_html(item)}</div>' for item in impact.additional_verticals[:4]])
                        st.markdown(f'<div class="impact-section"><div class="impact-label impact-label-verticals">Additional Verticals</div>{items}</div>', unsafe_allow_html=True)
                    if impact.relevant_services:
                        items = ''.join([f'<div class="impact-item">{escape_for_html(item)}</div>' for item in impact.relevant_services[:4]])
                        st.markdown(f'<div class="impact-section"><div class="impact-label impact-label-services">Relevant Services</div>{items}</div>', unsafe_allow_html=True)
                if impact.pitch_angle:
                    st.info(f"**Pitch Angle:** {impact.pitch_angle}")
                if impact.reasoning:
                    with st.expander("Detailed Reasoning", expanded=False):
                        st.markdown(impact.reasoning)

        # I1: Cross-trend compound impact synthesis
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

    if not st.session_state.companies:
        st.info("Ready to find target companies based on impact analysis")
        _, col_action = st.columns([5, 1])
        with col_action:
            if st.button("🔍 Find", type="primary", use_container_width=True,
                         disabled=st.session_state.get('step_2_running', False),
                         on_click=_on_find_companies):
                with st.spinner("Finding target companies..."):
                    add_log("Finding target companies...")
                    try:
                        async def run():
                            agent = CompanyAgent(mock_mode=st.session_state.mock_mode)
                            state = AgentState(trends=_rebuild_list(st.session_state.selected_trends, TrendData), impacts=_rebuild_list(st.session_state.impacts, ImpactAnalysis))
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
        st.success(f"Found {len(st.session_state.companies)} target companies. Review and select below.")
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
        _, col_action = st.columns([5, 1])
        with col_action:
            if st.button("🔍 Find", type="primary", use_container_width=True,
                         disabled=st.session_state.get('step_3_running', False),
                         on_click=_on_find_contacts):
                with st.spinner("Finding decision makers..."):
                    add_log(f"Finding contacts for {len(st.session_state.selected_companies)} companies...")
                    try:
                        async def run():
                            agent = ContactAgent(mock_mode=st.session_state.mock_mode)
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
        _, col_action = st.columns([5, 1])
        with col_action:
            if st.button("🔍 Generate", type="primary", use_container_width=True,
                         disabled=st.session_state.get('step_4_running', False),
                         on_click=_on_generate):
                with st.spinner("Generating personalized pitches..."):
                    add_log(f"Generating emails for {len(st.session_state.selected_contacts)} contacts...")
                    try:
                        async def run():
                            agent = EmailAgent(mock_mode=st.session_state.mock_mode)
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
        for email in st.session_state.outreach_emails:
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
                               f"leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "application/json", use_container_width=True)
        with col_csv:
            csv_data = [
                {"Score": score, "Company": e.company_name, "Contact": e.person_name, "Email": e.email, "Subject": e.subject}
                for score, e, _ in scored_leads
            ]
            st.download_button("CSV", pd.DataFrame(csv_data).to_csv(index=False),
                               f"leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv", use_container_width=True)
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
    init_session_state()
    render_sidebar()

    # Step indicator bar
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
