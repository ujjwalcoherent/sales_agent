"""
Sidebar component for the CMI Sales Agent Streamlit app.
Contains the main sidebar rendering and provider status functions.
"""

import streamlit as st

from app.config import DEFAULT_ACTIVE_SOURCES, NEWS_SOURCES, get_settings


# Keys to clear when resetting the pipeline.
_PIPELINE_STATE_KEYS = [
    'trends', 'selected_trends', 'selected_source_articles',
    'impacts', 'companies', 'selected_companies',
    'contacts', 'selected_contacts', 'outreach_emails',
    'logs', 'agent_state', 'articles', 'clusters', 'major_trends',
    'trend_tree', 'engine_metrics',
]


def _render_provider_status(settings) -> None:
    """Display connection status for each LLM and embedding provider."""
    import os
    from pathlib import Path

    # LLM Providers (in priority order)
    # 1. Full Vertex AI (service account — uses GCP credits)
    if settings.gcp_project_id and settings.gcp_service_account_file:
        sa_path = settings.gcp_service_account_file
        if not os.path.isabs(sa_path):
            sa_path = str(Path(__file__).resolve().parents[2] / sa_path)
        if os.path.exists(sa_path):
            st.caption(f"✅ **Vertex AI**: {settings.gemini_model} (GCP credits)")
        else:
            st.caption("⚠️ Vertex AI: SA file missing")
    elif settings.vertex_express_api_key:
        st.caption(f"✅ Vertex Express: {settings.gemini_model} (free 10 RPM)")
    elif settings.gemini_api_key:
        st.caption(f"✅ Gemini API: {settings.gemini_model}")

    if settings.nvidia_api_key:
        st.caption(f"✅ NVIDIA: {settings.nvidia_model}")
    if settings.openrouter_api_key:
        st.caption(f"✅ OpenRouter: {settings.openrouter_model}")

    # Ollama
    if settings.use_ollama:
        import httpx
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{settings.ollama_base_url}/api/tags")
                if resp.status_code == 200:
                    st.caption(f"✅ Ollama: {settings.ollama_model}")
                else:
                    st.caption("❌ Ollama: not responding")
        except Exception:
            st.caption("❌ Ollama: offline")

    # Embeddings
    st.caption(f"✅ Embeddings: {settings.embedding_provider}")

    if getattr(settings, 'offline_mode', False):
        st.caption("🔌 **Offline mode** — Ollama only")

    has_any = (settings.gcp_project_id or settings.vertex_express_api_key
               or settings.gemini_api_key or settings.nvidia_api_key
               or settings.openrouter_api_key or settings.use_ollama)
    if not has_any:
        st.error("No LLM providers configured!")


@st.cache_data(ttl=300)
def _get_learning_status() -> dict:
    """Load self-learning status from logs (cached 5 min)."""
    result = {"runs": 0, "mean_oss": None, "oss_improvement": None,
              "weight_drift": None, "top_sources": [], "bottom_sources": [],
              "weight_adapted": False}
    try:
        from pathlib import Path
        import json

        # Pipeline run log — last run quality
        plog = Path("data/pipeline_run_log.jsonl")
        if plog.exists():
            lines = [l for l in plog.read_text(encoding="utf-8").splitlines() if l.strip()]
            result["runs"] = len(lines)
            if lines:
                last = json.loads(lines[-1])
                rq = last.get("run_quality", {})
                result["mean_oss"] = rq.get("mean_oss")
                result["oss_improvement"] = rq.get("oss_improvement")
                sb = last.get("source_bandit", {})
                top5 = sb.get("top_5", {})
                bot5 = sb.get("bottom_5", {})
                result["top_sources"] = [(k, round(v, 3)) for k, v in list(top5.items())[:3]]
                result["bottom_sources"] = [(k, round(v, 3)) for k, v in list(bot5.items())[:2]]

        # Weight learner — check if it's adapting
        cslog = Path("data/cluster_signal_log.jsonl")
        if cslog.exists():
            records = [json.loads(l) for l in cslog.read_text(encoding="utf-8").splitlines() if l.strip()]
            unique_runs = len(set(r.get("run_id", "") for r in records))
            result["signal_runs"] = unique_runs
            result["signal_records"] = len(records)

        try:
            from app.learning.weight_learner import compute_learned_weights, _weight_cache
            from app.trends.signals.composite import DEFAULT_WEIGHTS
            _weight_cache.clear()
            weights = compute_learned_weights("actionability", DEFAULT_WEIGHTS)
            drift = sum(abs(weights.get(k, 0) - DEFAULT_WEIGHTS.get(k, 0)) for k in DEFAULT_WEIGHTS)
            adapted = sum(1 for k in DEFAULT_WEIGHTS if abs(weights.get(k, 0) - DEFAULT_WEIGHTS[k]) > 0.005)
            result["weight_drift"] = round(drift, 4)
            result["weight_adapted"] = adapted > 0
            result["weights_changed"] = adapted
        except Exception:
            pass
    except Exception:
        pass
    return result


def _render_self_learning_status() -> None:
    """Show self-learning health panel in sidebar."""
    try:
        status = _get_learning_status()
    except Exception:
        return

    st.markdown("---")
    st.markdown("### 🧠 Self-Learning Status")

    runs = status.get("runs", 0)
    mean_oss = status.get("mean_oss")
    improvement = status.get("oss_improvement")

    # OSS quality gauge
    if mean_oss is not None:
        oss_color = "🟢" if mean_oss >= 0.6 else "🟡" if mean_oss >= 0.3 else "🔴"
        trend_arrow = ""
        if improvement is not None:
            trend_arrow = f" {'↑' if improvement > 0 else '↓'}{abs(improvement):.2f}"
        st.caption(f"{oss_color} Avg OSS: **{mean_oss:.2f}**{trend_arrow} ({runs} runs)")
    else:
        st.caption(f"📊 {runs} runs logged")

    # Weight learner status
    signal_runs = status.get("signal_runs", 0)
    signal_records = status.get("signal_records", 0)
    if status.get("weight_adapted"):
        n_changed = status.get("weights_changed", 0)
        drift = status.get("weight_drift", 0)
        st.caption(f"✅ Weights adapting: {n_changed} factors (drift={drift:.3f})")
    elif signal_runs > 0:
        st.caption(f"⏳ Learning: {signal_runs}/3 runs, {signal_records} clusters logged")
    else:
        st.caption("⏳ Collecting data for weight adaptation...")

    # Top sources
    top = status.get("top_sources", [])
    if top:
        tops = ", ".join(f"{s}({v})" for s, v in top)
        st.caption(f"⭐ Top: {tops}")

    bot = status.get("bottom_sources", [])
    if bot:
        bots = ", ".join(f"{s}({v})" for s, v in bot)
        st.caption(f"⚠️ Low: {bots}")


def render_sidebar():
    """Render the sidebar with controls and status."""
    with st.sidebar:
        st.markdown("# 🎯 CMI Sales Agent")
        st.markdown("*Coherent Market Insights*")
        st.markdown("---")

        # Mode toggle
        st.markdown("### ⚙️ Settings")
        st.session_state.mock_mode = st.toggle(
            "Mock Mode",
            value=st.session_state.mock_mode,
            help="Use mock data instead of real API calls. Enable sub-options below to selectively use real APIs."
        )

        if st.session_state.mock_mode:
            st.info("🔧 Mock mode — choose what to mock:")
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.session_state.mock_rss = st.checkbox(
                    "News", value=st.session_state.get("mock_rss", True),
                    help="Mock RSS/news feeds",
                )
            with m_col2:
                st.session_state.mock_llm = st.checkbox(
                    "AI", value=st.session_state.get("mock_llm", True),
                    help="Mock LLM/AI analysis calls",
                )
            with m_col3:
                st.session_state.mock_search = st.checkbox(
                    "Search", value=st.session_state.get("mock_search", True),
                    help="Mock Tavily web search",
                )
            # Show what's real vs mocked
            real_parts = []
            if not st.session_state.mock_rss:
                real_parts.append("News")
            if not st.session_state.mock_llm:
                real_parts.append("AI")
            if not st.session_state.mock_search:
                real_parts.append("Search")
            if real_parts:
                st.caption(f"Real: {', '.join(real_parts)} • Mocked: rest")
        else:
            st.warning("⚡ Live mode: Real API calls")
            # Reset granular mocks when going live
            st.session_state.mock_rss = True
            st.session_state.mock_llm = True
            st.session_state.mock_search = True

        st.session_state.show_tooltips = st.toggle(
            "Show Tooltips",
            value=st.session_state.get("show_tooltips", get_settings().show_tooltips),
            help="Show formula tooltips on hover for all metrics"
        )

        st.session_state.show_ai_activity = st.toggle(
            "Show AI Activity",
            value=st.session_state.get("show_ai_activity", False),
            help="Show detailed AI reasoning, council perspectives, and confidence scores at each step"
        )

        # Provider status
        st.markdown("---")
        st.markdown("### 🔌 Provider Status")
        settings = get_settings()
        _render_provider_status(settings)

        # News Sources info
        st.markdown("---")
        st.markdown("### 📰 News Sources")
        rss_count = len([s for s in DEFAULT_ACTIVE_SOURCES if NEWS_SOURCES.get(s, {}).get("source_type") == "rss"])
        api_count = len([s for s in DEFAULT_ACTIVE_SOURCES if NEWS_SOURCES.get(s, {}).get("source_type") == "api"])
        st.caption(f"🔗 {rss_count} RSS feeds • 🔌 {api_count} APIs")

        # Pipeline steps - Consultant Flow
        st.markdown("---")
        st.markdown("### 📋 Consultant Pipeline")

        steps = [
            ("1️⃣", "News Detection", 0),
            ("2️⃣", "Opportunity Analysis", 1),
            ("3️⃣", "Target Companies", 2),
            ("4️⃣", "Decision Makers", 3),
            ("5️⃣", "Pitch Generation", 4)
        ]

        for icon, name, idx in steps:
            if idx < st.session_state.current_step:
                if st.button(f"✓ {icon} {name}", key=f"sidebar_nav_{idx}", use_container_width=True):
                    st.session_state.current_step = idx
                    st.rerun()
            elif idx == st.session_state.current_step:
                st.markdown(f"<span class='step-active'>→ {icon} {name}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='step-pending'>○ {icon} {name}</span>", unsafe_allow_html=True)

        # Stats
        st.markdown("---")
        st.markdown("### 📊 Current Stats")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Articles", len(st.session_state.get('articles', [])))
            st.metric("Trends", len(st.session_state.trends))
            st.metric("Companies", len(st.session_state.companies))
        with col2:
            articles = st.session_state.get('articles', [])
            st.metric("Sources", len({a.source_name for a in articles}) if articles else 0)
            st.metric("Contacts", len(st.session_state.contacts))
            st.metric("Emails", len(st.session_state.outreach_emails))

        # Show signal distribution if tree exists
        tree = st.session_state.get('trend_tree')
        if tree and hasattr(tree, 'nodes') and tree.nodes:
            strong = sum(1 for n in tree.nodes.values() if n.signal_strength == "strong")
            weak = sum(1 for n in tree.nodes.values() if n.signal_strength == "weak")
            noise = sum(1 for n in tree.nodes.values() if n.signal_strength == "noise")
            st.caption(f"Signals: {strong} strong, {weak} weak, {noise} noise")

        # Feedback summary
        try:
            from app.tools.feedback import get_feedback_summary
            fb = get_feedback_summary()
            if fb["total"] > 0:
                st.markdown("---")
                st.markdown("### 📝 Feedback")
                t = fb["trends"]
                l = fb["leads"]
                trend_total = t["good_trend"] + t["bad_trend"] + t["already_knew"]
                lead_total = l["would_email"] + l["maybe"] + l["bad_lead"]
                if trend_total:
                    st.caption(
                        f"Trends ({trend_total}): "
                        f"{t['good_trend']} good, {t['bad_trend']} bad, {t['already_knew']} known"
                    )
                if lead_total:
                    st.caption(
                        f"Leads ({lead_total}): "
                        f"{l['would_email']} email, {l['maybe']} maybe, {l['bad_lead']} bad"
                    )
        except Exception:
            pass

        # Self-learning status panel
        _render_self_learning_status()

        # Reset button
        st.markdown("---")
        if st.button("🔄 Reset Pipeline", type="secondary", use_container_width=True):
            st.session_state.current_step = 0
            for key in _PIPELINE_STATE_KEYS:
                st.session_state[key] = None if key in ('trend_tree', 'agent_state') else ({} if key == 'engine_metrics' else [])
            st.rerun()
