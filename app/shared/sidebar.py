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
    import httpx

    # Check Ollama
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{settings.ollama_base_url}/api/tags")
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                st.caption(f"✅ Ollama: {', '.join(models)}")
            else:
                st.caption("❌ Ollama: not responding")
    except Exception:
        st.caption("❌ Ollama: offline")

    # Check cloud providers
    cloud_providers = {
        "Gemini": settings.gemini_api_key,
        "OpenRouter": settings.openrouter_api_key,
        "Groq": settings.groq_api_key,
        "HuggingFace": settings.huggingface_api_key,
    }
    for name, api_key in cloud_providers.items():
        if api_key:
            st.caption(f"✅ {name}: configured")

    has_any_provider = settings.use_ollama or any(cloud_providers.values())
    if not has_any_provider:
        st.error("No LLM providers configured!")


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
            help="Use mock data instead of real API calls"
        )

        if st.session_state.mock_mode:
            st.info("🔧 Mock mode: No API credits used")
        else:
            st.warning("⚡ Live mode: Real API calls")

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
                st.markdown(f"<span class='step-completed'>✓ {icon} {name}</span>", unsafe_allow_html=True)
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

        # Reset button
        st.markdown("---")
        if st.button("🔄 Reset Pipeline", type="secondary", use_container_width=True):
            st.session_state.current_step = 0
            for key in _PIPELINE_STATE_KEYS:
                st.session_state[key] = None if key in ('trend_tree', 'agent_state') else ({} if key == 'engine_metrics' else [])
            st.rerun()
