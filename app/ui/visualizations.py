"""
Visualization components for trend display in the CMI Sales Agent.
Contains tree rendering, flow diagrams, bubble maps, and detail panels.

V2: Sparkline SVG rendering (inline, no Plotly overhead)
V3: Score breakdown mini-bars
V4: Related trends display
"""

import html
import streamlit as st
import streamlit.components.v1 as st_components

from app.config import get_settings
from app.schemas import TrendData
from app.shared.helpers import escape_for_html, strip_html_tags, truncate_text


# ══════════════════════════════════════════════════════════════════════════════
# TOOLTIP CONTENT — formula or source for every displayed metric
# ══════════════════════════════════════════════════════════════════════════════

TOOLTIP_CONTENT = {
    # Metrics grid
    "trend_score": (
        "Overall significance: 45% momentum (accelerating?)"
        " + 30% volume (how many articles?) + 25% source diversity"
    ),
    "actionability": (
        "Sales outreach value (0\u20131). Weighted mix of trigger event,"
        " source authority, recency, specificity, and 6 more signals."
        " See breakdown below"
    ),
    "source_diversity": (
        "What fraction of articles come from different outlets."
        " 1.0 = every article from a unique source. Below 0.2 = echo chamber"
    ),
    # Cluster quality
    "grade": (
        "Cluster quality A\u2013F: 35% coherence (do articles match?),"
        " 25% entity overlap, 20% keyword overlap,"
        " 10% source diversity, 10% time spread"
    ),
    "coherence": (
        "How similar the articles are to each other."
        " Higher = articles clearly discuss the same topic"
    ),
    "entity_match": (
        "% of articles that mention at least one of the cluster\u2019s"
        " top entities. Higher = consistent reporting"
    ),
    # Source credibility
    "dynamic_score": (
        "Computed credibility for this cluster:"
        " 40% source tier + 25% cross-citation"
        " + 20% originality + 15% factual agreement"
    ),
    "base_tier": (
        "Average preset credibility of the sources."
        " Tier 1 (ET, RBI, SEBI) = 0.90\u20130.99."
        " Tier 2 (YourStory, Inc42) = 0.85\u20130.90"
    ),
    "cross_citation": (
        "How much do different publishers cover the same facts?"
        " More publishers = higher score"
    ),
    # Badges — depth
    "depth_badge": (
        "AI Council classifies trend scope:"
        " MAJOR = broad theme worth standalone attention,"
        " SUB = subtopic under a bigger theme,"
        " MICRO = narrow niche point"
    ),
    # Badges — signal
    "signal_badge": (
        "Ranked against all trends: STRONG = top 50%,"
        " WEAK = 10\u201350% (still growing), NOISE = bottom 10%"
    ),
    # Badges — severity
    "severity_badge": (
        "AI reads the articles and judges business urgency:"
        " HIGH = act now, MEDIUM = monitor, LOW = informational"
    ),
    # Badges — article count
    "article_count": (
        "Number of deduplicated articles in this trend cluster."
        " More articles = stronger evidence the trend is real"
    ),
    # AI Council
    "ai_council": (
        "Three-stage AI validation pipeline. "
        "Stage A: Trend Validator ranks importance, filters noise. "
        "Stage B: Impact Council rates business impact "
        "(industry + strategy + risk). "
        "Stage C: Lead Validator scores company-trend fit. "
        "A Validator Agent also checks all output for hallucinations"
    ),
    "business_importance": (
        "AI Council\u2019s rating of how significant this trend is"
        " for consulting. 70%+ = high priority,"
        " 40\u201370% = moderate, below 40% = low priority"
    ),
    "event_type": (
        "AI-classified event category (e.g., regulation, acquisition,"
        " expansion). Detected via embedding similarity to real headlines,"
        " then validated by LLM for ambiguous cases"
    ),
    # Trend velocity
    "trend_velocity": (
        "Article publication rate over time."
        " The sparkline shows how fast articles appeared in each time window"
    ),
    "momentum_badge": (
        "SPIKING = sudden 3x burst (breaking news),"
        " ACCELERATING = consistently growing,"
        " DECELERATING = fading, STEADY = flat"
    ),
    "forecast": (
        "Predicted direction based on recent velocity trend."
        " Uses slope of the last 3 time windows"
    ),
    # LLM-generated sections
    "buying_intent": (
        "AI-assessed purchase urgency: IMMEDIATE = active need,"
        " SHORT TERM = within weeks, MEDIUM TERM = future opportunity"
    ),
    "why_this_matters": (
        "AI-generated market observation grounded in the source articles."
        " Factual, not a sales pitch"
    ),
    "entities": (
        "Companies, people, and organizations extracted from the articles"
        " using NER. Ordered by prominence \u2014 mentioned earlier = more important"
    ),
    # 5W1H and causal chain
    "event_5w1h": (
        "AI-extracted event details: Who is involved, What happened,"
        " When, Where, Why, and Who is affected"
    ),
    "causal_chain": (
        "AI-traced sequence of cause and effect."
        " Shows how one event leads to the next"
    ),
    # Score breakdown factors
    "recency": (
        "~12% weight (adapts with feedback) \u2014 How fresh is the news?"
        " Decays fast: 12hrs = 65%, 24hrs = 18%"
    ),
    "velocity": (
        "~7% weight (adapts) \u2014 Articles per hour."
        " Above 2/hr = hot, above 5/hr = breaking news"
    ),
    "specificity": (
        "~12% weight (adapts) \u2014 Does it name specific companies,"
        " amounts, or people? Vague trends score low"
    ),
    "regulatory": (
        "~12% weight (adapts) \u2014 Is there a government or policy angle?"
        " Yes = full bonus, no = zero"
    ),
    "trigger": (
        "~14% weight (largest, adapts) \u2014 Event urgency:"
        " regulation = 0.95, crisis = 0.90, acquisition = 0.80,"
        " general = 0.30"
    ),
    "diversity": (
        "~7% weight (adapts) \u2014 How many independent outlets are reporting?"
        " More sources = more credible"
    ),
    "authority": (
        "~13% weight (adapts) \u2014 Average source credibility."
        " Tier 1 sources (RBI, ET) push this higher. Blends static"
        " credibility with source bandit Thompson Sampling posterior"
    ),
    "financial": (
        "~5% weight (adapts) \u2014 Are dollar amounts, revenue,"
        " or funding figures mentioned? Yes/no bonus"
    ),
    "person": (
        "~3% weight (adapts) \u2014 Is a key decision-maker (CEO, minister)"
        " named in the articles? Yes/no bonus"
    ),
    "event_focus": (
        "~5% weight (adapts) \u2014 Do articles agree on the event type?"
        " Higher = consistent narrative, not mixed signals"
    ),
    "cmi_relevance": (
        "~10% weight (adapts) \u2014 Relevance to CMI service portfolio."
        " Higher = directly matches our consulting/intelligence offerings"
    ),
    "confidence": (
        "Computed from output QUALITY, not just item counts."
        " 25% evidence grounding (citations with concrete data like numbers/names),"
        " 20% specificity (company types with employee ranges/locations),"
        " 20% analytical depth (reasoning length + paragraph structure),"
        " 15% problem concreteness (pain points with specific consequences),"
        " 10% service fit (recs with substantive justifications),"
        " 10% cross-validation (internal consistency checks)."
        " Penalizes vague consulting jargon. Capped at 95%."
    ),
    "cross_trend": (
        "Entity bridges linking this sub-trend to sub-trends in OTHER"
        " parent clusters. Shows chain reactions across topics"
    ),
}


def _tt(key: str) -> str:
    """Return data-tooltip attr if enabled, else empty string."""
    enabled = st.session_state.get("show_tooltips", get_settings().show_tooltips)
    if not enabled:
        return ""
    text = TOOLTIP_CONTENT.get(key, "")
    if not text:
        return ""
    safe = text.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
    return f'data-tooltip="{safe}"'


# ══════════════════════════════════════════════════════════════════════════════
# V2: SPARKLINE SVG GENERATOR — Google Trends / Meltwater style
# ══════════════════════════════════════════════════════════════════════════════

def render_sparkline_svg(
    histogram: list,
    width: int = 160,
    height: int = 32,
    color: str = "#00d4ff",
    fill_color: str = "rgba(0,212,255,0.15)",
) -> str:
    """
    Generate inline SVG sparkline from temporal histogram data.

    No external dependency — pure SVG string. Renders a mini area chart
    showing article velocity over time.

    Args:
        histogram: List of dicts with 'velocity' (or 'count') key.
        width: SVG width in pixels.
        height: SVG height in pixels.
        color: Line stroke color.
        fill_color: Area fill color.

    Returns:
        HTML string with embedded SVG. Empty string if no data.
    """
    if not histogram or len(histogram) < 2:
        return ""

    values = [h.get("velocity", h.get("count", 0)) for h in histogram]
    max_val = max(values) if values else 1
    if max_val == 0:
        max_val = 1

    # Generate SVG points
    n = len(values)
    padding = 2
    usable_w = width - 2 * padding
    usable_h = height - 2 * padding

    points = []
    for i, v in enumerate(values):
        x = padding + (i / max(n - 1, 1)) * usable_w
        y = padding + usable_h - (v / max_val) * usable_h
        points.append(f"{x:.1f},{y:.1f}")

    polyline_points = " ".join(points)

    # Area fill: close path at bottom
    fill_points = f"{padding},{height - padding} " + polyline_points + f" {width - padding},{height - padding}"

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'style="display:inline-block; vertical-align:middle;">'
        f'<polygon points="{fill_points}" fill="{fill_color}" />'
        f'<polyline points="{polyline_points}" fill="none" stroke="{color}" '
        f'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />'
        f'</svg>'
    )


def _render_score_breakdown_html(node) -> str:
    """
    Render actionability score breakdown with weights, raw values, and contribution.

    Shows ALL factors (not just top 6) so users can see exactly where the
    score comes from and WHY each component scored high/low.
    """
    breakdown = getattr(node, 'signals', {}).get('actionability_breakdown', {})
    if not breakdown:
        return ""

    # Sort by contribution (highest first)
    sorted_factors = sorted(
        breakdown.items(), key=lambda x: x[1].get('contribution', 0), reverse=True
    )

    max_contribution = max((f[1].get('contribution', 0) for f in sorted_factors), default=0.01) or 0.01

    bar_items = []
    for name, info in sorted_factors:
        contribution = info.get('contribution', 0)
        weight = info.get('weight', 0)
        raw = info.get('raw', 0)
        normalized = info.get('normalized', raw)
        bar_width = max(2, int((contribution / max_contribution) * 100))
        bar_color = "#00ff88" if contribution > 0.08 else ("#ffa502" if contribution > 0.03 else "#444")
        display_name = name.replace('_', ' ').title()
        # Show weight% and raw value on hover via tooltip
        weight_pct = f"{weight:.0%}" if weight else ""
        raw_str = f"{raw:.2f}" if isinstance(raw, float) else str(raw)
        bar_items.append(
            f'<div style="display:flex; align-items:center; gap:4px; margin:2px 0;">'
            f'<span style="width:75px; font-size:10px; color:#888; text-align:right;" {_tt(name)}>{display_name}</span>'
            f'<span style="width:28px; font-size:9px; color:#555; text-align:right;">{weight_pct}</span>'
            f'<div style="flex:1; background:#1a1a2e; border-radius:3px; height:7px;">'
            f'<div style="width:{bar_width}%; background:{bar_color}; height:100%; border-radius:3px;"></div>'
            f'</div>'
            f'<span style="width:30px; font-size:9px; color:#aaa;">{contribution:.3f}</span>'
            f'</div>'
        )

    if not bar_items:
        return ""

    return (
        f'<div class="detail-section">'
        f'<div class="detail-section-title" {_tt("actionability")}>Actionability Breakdown</div>'
        f'<div style="display:flex; gap:4px; margin-bottom:3px; padding:0 0 2px 0; border-bottom:1px solid #222;">'
        f'<span style="width:75px; font-size:9px; color:#555; text-align:right;">Factor</span>'
        f'<span style="width:28px; font-size:9px; color:#555; text-align:right;">Wt</span>'
        f'<span style="flex:1; font-size:9px; color:#555;"></span>'
        f'<span style="width:30px; font-size:9px; color:#555;">Score</span>'
        f'</div>'
        f'{"".join(bar_items)}'
        f'</div>'
    )



# Try to import flow visualization
try:
    from streamlit_flow import streamlit_flow
    from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
    from streamlit_flow.state import StreamlitFlowState
    from streamlit_flow.layouts import TreeLayout
    FLOW_AVAILABLE = True
except ImportError:
    FLOW_AVAILABLE = False


def render_tree_node(
    node, tree, articles_map: dict, selected_nodes: list,
    selected_articles: list, node_index: int, parent_prefix: str = "",
    node_filter=None,
) -> int:
    """Recursively render a trend tree node with hierarchy visualization."""
    if node_filter and not node_filter(node):
        for child_id in node.children_ids:
            child_node = tree.nodes.get(str(child_id))
            if child_node:
                node_index = render_tree_node(
                    child_node, tree, articles_map, selected_nodes,
                    selected_articles, node_index, parent_prefix + "  ", node_filter
                )
        return node_index

    depth = node.depth
    has_children = bool(node.children_ids)
    depth_class = {1: "depth-major", 2: "depth-sub", 3: "depth-micro"}.get(depth, "depth-major")
    depth_label = {1: "MAJOR", 2: "SUB", 3: "MICRO"}.get(depth, "TREND")
    card_class = {1: "tree-card", 2: "tree-card tree-card-sub", 3: "tree-card tree-card-micro"}.get(depth, "tree-card")

    sig = node.signal_strength
    if sig == "strong":
        signal_class, signal_label = "signal-strong", "STRONG"
    elif sig == "weak":
        signal_class, signal_label = "signal-weak", "WEAK"
    else:
        signal_class, signal_label = "signal-noise", "NOISE"

    severity_val = node.severity.value.upper() if hasattr(node.severity, 'value') else str(node.severity).upper()
    safe_title = escape_for_html(node.trend_title)
    safe_summary = escape_for_html(node.trend_summary[:300] + "..." if len(node.trend_summary) > 300 else node.trend_summary)

    # Combine entities and affected companies (deduped), skip generic keywords
    entity_set = []
    seen_lower = set()
    for e in (getattr(node, 'key_entities', []) or []):
        if e.lower() not in seen_lower:
            seen_lower.add(e.lower())
            entity_set.append(e)
    for c in (getattr(node, 'affected_companies', []) or []):
        if c.lower() not in seen_lower:
            seen_lower.add(c.lower())
            entity_set.append(c)
    ent_html = ""
    if entity_set:
        ent_spans = ''.join([f'<span style="background: rgba(255,0,255,0.1); padding: 2px 6px; border-radius: 8px; font-size: 10px; margin-right: 4px;">{escape_for_html(e)}</span>' for e in entity_set[:8]])
        ent_html = f'<div style="margin-top: 6px;"><span style="font-size: 9px; color: #666; margin-right: 6px;" {_tt("entities")}>ENTITIES</span>{ent_spans}</div>'

    insight_html = ""
    actionable_insight = getattr(node, 'actionable_insight', '') or ''
    if actionable_insight:
        clean_insight = strip_html_tags(actionable_insight)
        safe_insight = escape_for_html(clean_insight)
        insight_html = (
            '<div style="margin-top: 10px; padding: 10px 12px; background: rgba(0, 255, 136, 0.08); border-left: 3px solid #00ff88; border-radius: 4px;">'
            f'<span style="font-size: 11px; font-weight: bold; color: #00ff88;" {_tt("why_this_matters")}>WHY THIS MATTERS</span>'
            f'<p style="color: #ccc; margin: 4px 0 0 0; font-size: 12px; line-height: 1.4;">{safe_insight}</p>'
            '</div>'
        )

    # Buying intent (6sense/Bombora style)
    buying_html = ""
    buying = getattr(node, 'buying_intent', {}) or {}
    if buying and buying.get('signal_type'):
        urgency_colors = {"immediate": "#ff4757", "short_term": "#ffa502", "medium_term": "#2ed573"}
        urg = buying.get('urgency', 'medium_term')
        urg_color = urgency_colors.get(urg, "#999")
        safe_hook = escape_for_html(buying.get('pitch_hook', '')[:250])
        safe_who = escape_for_html(buying.get('who_needs_help', '')[:150])
        safe_what = escape_for_html(buying.get('what_they_need', '')[:150])
        buying_html = (
            '<div style="margin-top: 8px; padding: 8px 12px; background: rgba(0, 212, 255, 0.06); border-left: 3px solid #00d4ff; border-radius: 4px;">'
            f'<span style="font-size: 11px; font-weight: bold; color: #00d4ff;" {_tt("buying_intent")}>BUYING INTENT</span>'
            f'<span style="font-size: 10px; background: {urg_color}22; color: {urg_color}; padding: 2px 6px; border-radius: 8px; margin-left: 8px;">{urg.replace("_", " ").upper()}</span>'
            f'<p style="color: #ccc; margin: 4px 0 2px 0; font-size: 12px;">{safe_hook}</p>'
            f'<p style="color: #888; margin: 2px 0; font-size: 11px;">Target: {safe_who}</p>'
            '</div>'
        )

    companies_html = ""  # Merged into ent_html above

    tree_icon = "📂" if has_children else "📄"
    children_hint = f" ({len(node.children_ids)} sub-topics)" if has_children else ""
    score_html = f'<span style="font-size: 10px; color: #666;">Score: {node.trend_score:.2f}</span>' if node.trend_score > 0 else ""

    # Use inline styles for indentation (more reliable than CSS classes in Streamlit)
    indent_px = (depth - 1) * 28
    border_style = f"border-left: 2px solid {'#444' if depth == 2 else '#555'}; padding-left: 12px;" if depth > 1 else ""

    with st.container():
        col1, col2 = st.columns([0.06, 0.94])
        with col1:
            checked = st.checkbox("Sel", key=f"tree_node_{node_index}_{str(node.id)[:8]}", value=(depth == 1), label_visibility="collapsed")
            if checked:
                selected_nodes.append(TrendData(
                    id=str(node.id), trend_title=node.trend_title, summary=node.trend_summary,
                    severity=node.severity,
                    industries_affected=[s.value if hasattr(s, 'value') else str(s) for s in node.primary_sectors],
                    source_links=[],
                    keywords=node.key_entities[:10] if node.key_entities else node.key_keywords[:10],
                ))

        with col2:
            card_html = (
                f'<div class="{card_class}" style="margin-left: {indent_px}px; {border_style}">'
                '<div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 8px;">'
                '<div style="flex: 1;">'
                f'<span class="depth-badge {depth_class}" {_tt("depth_badge")}>{depth_label}</span>'
                f'<span style="font-size: 16px; font-weight: 600; color: #e0e0e0;">{tree_icon} {safe_title}{children_hint}</span>'
                '</div>'
                '<div style="display: flex; gap: 8px; align-items: center;">'
                f'<span style="font-size: 11px; color: #888;" {_tt("article_count")}>{node.article_count} articles</span>'
                f'<span class="{signal_class}" {_tt("signal_badge")}>{signal_label}</span>'
                f'<span style="background: rgba(255,255,255,0.08); color: #aaa; padding: 3px 8px; border-radius: 10px; font-size: 10px;" {_tt("severity_badge")}>{severity_val}</span>'
                f'{score_html}'
                '</div></div>'
                f'<p style="color: #999; margin: 8px 0 4px 0; font-size: 13px; line-height: 1.4;">{safe_summary}</p>'
                f'{buying_html}{ent_html}{insight_html}'
                '</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

            node_articles = [articles_map.get(str(aid)) for aid in node.source_articles if str(aid) in articles_map]
            node_articles = [a for a in node_articles if a]
            if node_articles:
                src_names = sorted({a.source_name for a in node_articles})
                with st.expander(f"📄 {len(node_articles)} articles from {', '.join(src_names[:3])}{'...' if len(src_names) > 3 else ''}", expanded=False):
                    for j, art in enumerate(node_articles[:15]):
                        art_key = f"art_{node_index}_{j}_{str(art.id)[:8]}"
                        acol1, acol2 = st.columns([0.04, 0.96])
                        with acol1:
                            art_sel = st.checkbox("S", key=art_key, value=False, label_visibility="collapsed")
                        with acol2:
                            pub_date = art.published_at.strftime("%b %d") if hasattr(art, 'published_at') and art.published_at else ""
                            safe_art_title = escape_for_html(art.title[:90])
                            safe_url = html.escape(getattr(art, 'url', '') or '#')
                            src_name = escape_for_html(art.source_name) if art.source_name else ''
                            date_prefix = f'<span style="color: #666; font-size: 11px; margin-right: 8px;">{pub_date}</span>' if pub_date else ''
                            row_html = f'{date_prefix}<span style="color: #ccc;">{safe_art_title}</span><span style="color: #555; float: right; font-size: 11px;">{src_name}</span>'
                            if safe_url != '#':
                                st.markdown(f'<a href="{safe_url}" target="_blank" style="display: block; font-size: 12px; padding: 4px 0; text-decoration: none; border-bottom: 1px solid #222;">{row_html}</a>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div style="font-size: 12px; padding: 4px 0; border-bottom: 1px solid #222;">{row_html}</div>', unsafe_allow_html=True)
                        if art_sel:
                            selected_articles.append(art)

    node_index += 1
    if has_children:
        for child_id in node.children_ids:
            child_node = tree.nodes.get(str(child_id))
            if child_node:
                node_index = render_tree_node(child_node, tree, articles_map, selected_nodes, selected_articles, node_index, parent_prefix + "  ", node_filter)

    return node_index


def _close_detail_sidebar():
    """Callback to close the detail sidebar on single click."""
    if "selected_trend_id" in st.session_state:
        del st.session_state["selected_trend_id"]


def render_sliding_sidebar(node, tree, articles_map: dict):
    """Render sliding sidebar with trend details."""
    # Single-click close via on_click callback
    st.button("✕ Close", key="close_detail_sidebar", on_click=_close_detail_sidebar)

    # V7: Use AI-validated depth_label from TrendNode (falls back to positional)
    raw_depth = getattr(node, 'depth_label', None)
    if raw_depth:
        depth_label = raw_depth.upper() if isinstance(raw_depth, str) else str(raw_depth).split('.')[-1].upper()
    else:
        depth_label = {1: "MAJOR", 2: "SUB", 3: "MICRO"}.get(node.depth, "TREND")
    depth_class_map = {"MAJOR": "depth-major", "SUB": "depth-sub", "MICRO": "depth-micro", "MEGA": "depth-major"}
    depth_class = depth_class_map.get(depth_label, "depth-major")
    sig = str(node.signal_strength).upper()
    sig_class = {"STRONG": "signal-strong", "WEAK": "signal-weak", "NOISE": "signal-noise"}.get(sig, "signal-noise")
    severity_val = node.severity.value.upper() if hasattr(node.severity, "value") else str(node.severity).upper()
    safe_title = escape_for_html(node.trend_title)
    safe_summary = escape_for_html(node.trend_summary)

    # Merge entities + affected companies (deduped), skip generic keywords
    _entity_set = []
    _seen = set()
    for e in (node.key_entities or []):
        if e.lower() not in _seen:
            _seen.add(e.lower())
            _entity_set.append(e)
    for c in (getattr(node, 'affected_companies', []) or []):
        if c.lower() not in _seen:
            _seen.add(c.lower())
            _entity_set.append(c)
    ent_html = ""
    if _entity_set:
        ent_spans = "".join([f'<span class="entity-tag">{escape_for_html(e)}</span>' for e in _entity_set[:10]])
        ent_html = f'<div class="detail-section"><div class="detail-section-title" {_tt("entities")}>Key Entities</div>{ent_spans}</div>'

    insight_html = ""
    actionable_insight = getattr(node, "actionable_insight", "") or ""
    if actionable_insight:
        clean_insight = strip_html_tags(actionable_insight)
        safe_insight = escape_for_html(clean_insight)
        insight_html = f'<div class="insight-box"><div class="insight-label" {_tt("why_this_matters")}>💡 WHY THIS MATTERS</div><div class="insight-text">{safe_insight}</div></div>'

    # V2: Sparkline from temporal histogram
    sparkline_html = ""
    temporal_hist = getattr(node, 'temporal_histogram', []) or []
    if temporal_hist:
        sparkline_svg = render_sparkline_svg(temporal_hist)
        momentum = getattr(node, 'momentum_label', '') or ''
        momentum_colors = {"accelerating": "#00ff88", "decelerating": "#ff4757", "spiking": "#ffa502", "steady": "#888"}
        momentum_color = momentum_colors.get(momentum, "#888")
        momentum_badge = f' <span style="font-size:9px; background:{momentum_color}22; color:{momentum_color}; padding:1px 6px; border-radius:8px;" {_tt("momentum_badge")}>{momentum.upper()}</span>' if momentum else ""
        # Momentum prediction if available
        prediction = getattr(node, 'signals', {}).get('momentum_prediction', '')
        pred_html = ""
        if prediction and prediction != "insufficient_data":
            pred_colors = {"likely_growing": "#00ff88", "likely_fading": "#ff4757", "stable": "#888"}
            pred_color = pred_colors.get(prediction, "#888")
            pred_label = prediction.replace('_', ' ').title()
            pred_html = f'<div style="font-size:10px; color:{pred_color}; margin-top:2px;" {_tt("forecast")}>Forecast: {pred_label}</div>'
        sparkline_html = (
            '<div class="detail-section">'
            f'<div class="detail-section-title" {_tt("trend_velocity")}>Trend Velocity{momentum_badge}</div>'
            f'<div style="padding: 4px 0;">{sparkline_svg}</div>'
            f'{pred_html}'
            '</div>'
        )

    # Trend Score with breakdown
    ts_breakdown = (getattr(node, 'signals', {}) or {}).get('trend_score_breakdown', {})
    ts_detail_html = ""
    if ts_breakdown:
        ts_items = []
        for fname, fdata in sorted(ts_breakdown.items(), key=lambda x: x[1].get('contribution', 0), reverse=True):
            contrib = fdata.get('contribution', 0)
            weight = fdata.get('weight', 0)
            raw = fdata.get('raw', 0)
            display = fname.replace('_', ' ').title()
            bar_w = max(2, int(min(100, raw * 100)))
            ts_items.append(
                f'<div style="display:flex; align-items:center; gap:4px; margin:1px 0;">'
                f'<span style="width:70px; font-size:9px; color:#666; text-align:right;">{display} ({weight:.0%})</span>'
                f'<div style="flex:1; background:#1a1a2e; border-radius:2px; height:6px;">'
                f'<div style="width:{bar_w}%; background:#00d4ff; height:100%; border-radius:2px;"></div></div>'
                f'<span style="width:30px; font-size:9px; color:#888;">{raw:.2f}</span>'
                f'</div>'
            )
        ts_detail_html = f'<div style="margin-top:4px;">{"".join(ts_items)}</div>'

    scores_html = (
        '<div class="detail-section">'
        '<div class="detail-section-title">Metrics</div>'
        '<div class="scores-grid">'
        f'<div class="score-item"><div class="score-value" style="color: #00d4ff;">{node.trend_score:.2f}</div><div class="score-label" {_tt("trend_score")}>Trend Score</div></div>'
        f'<div class="score-item"><div class="score-value" style="color: #00ff88;">{node.actionability_score:.2f}</div><div class="score-label" {_tt("actionability")}>Actionability</div></div>'
        f'<div class="score-item"><div class="score-value" style="color: #ffa502;">{node.source_diversity:.0%}</div><div class="score-label" {_tt("source_diversity")}>Source Diversity</div></div>'
        '</div>'
        f'{ts_detail_html}'
        '</div>'
    )

    # Confidence section
    confidence_html = ""
    conf = getattr(node, 'confidence', None)
    if conf and hasattr(conf, 'score'):
        conf_score = conf.score
        conf_color = "#00ff88" if conf_score >= 0.7 else "#ffa502" if conf_score >= 0.5 else "#ff4757"
        conf_level = getattr(conf, 'level', 'moderate')
        factor_items = ""
        if hasattr(conf, 'factors') and conf.factors:
            factor_items = "".join(
                f'<div style="padding:1px 0; font-size:10px; color:#888;">{"+" if "+" in f or "present" in f.lower() or "high" in f.lower() or "strong" in f.lower() or "rich" in f.lower() or "detailed" in f.lower() else "-"} {escape_for_html(f)}</div>'
                for f in conf.factors[:8]
            )
        confidence_html = (
            '<div class="detail-section">'
            f'<div class="detail-section-title" style="color: {conf_color};">Confidence</div>'
            f'<div style="display:flex; align-items:baseline; gap:8px; margin-bottom:4px;">'
            f'<span style="color:{conf_color}; font-size:20px; font-weight:700;">{conf_score:.0%}</span>'
            f'<span style="color:#888; font-size:11px;">{conf_level.replace("_", " ").title()}</span>'
            f'</div>'
            f'{factor_items}'
            '</div>'
        )

    # Cross-trend bridges section
    bridges_html = ""
    bridges = (getattr(node, 'signals', {}) or {}).get('cross_trend_bridges', [])
    if bridges:
        bridge_items = []
        for b in bridges[:5]:
            is_source = b.get('source_node_id') == str(node.id)
            linked_title = b.get('target_title', 'Unknown') if is_source else b.get('source_title', 'Unknown')
            strength = b.get('strength', 0)
            shared = b.get('bridge_entities', [])[:4]
            shared_str = ", ".join(shared) if shared else "shared entities"
            s_color = "#00ff88" if strength > 0.3 else "#ffa502" if strength > 0.15 else "#888"
            bridge_items.append(
                f'<div style="padding:4px 6px; background:rgba(124,58,237,0.08); border-left:2px solid #7c3aed; border-radius:0 4px 4px 0; margin:3px 0;">'
                f'<div style="font-size:11px; color:#e0e0e0;">{escape_for_html(linked_title[:60])}</div>'
                f'<div style="font-size:10px; color:#888;">via {escape_for_html(shared_str)} '
                f'<span style="color:{s_color};">(strength: {strength:.2f})</span></div>'
                f'</div>'
            )
        bridges_html = (
            '<div class="detail-section">'
            '<div class="detail-section-title" style="color: #7c3aed;">Cross-Trend Connections</div>'
            f'{"".join(bridge_items)}'
            '</div>'
        )

    # V7: AI Council validation section
    ai_validation_html = ""
    validation_reasoning = getattr(node, 'validation_reasoning', '') or ''
    importance = getattr(node, 'importance_score', 0.0) or 0.0
    validated_event = getattr(node, 'validated_event_type', '') or ''
    event_reasoning = getattr(node, 'event_type_reasoning', '') or ''

    if validation_reasoning or importance > 0:
        imp_color = "#00ff88" if importance >= 0.7 else "#ffa502" if importance >= 0.4 else "#ff4757"
        validated_event_html = ""
        if validated_event:
            evt_label = validated_event.replace("_", " ").title()
            validated_event_html = (
                f'<div style="margin-top:4px;">'
                f'<span style="color: #7c3aed; font-size: 11px; font-weight: 600;" {_tt("event_type")}>Event Type:</span> '
                f'<span style="color: #e0e0e0; font-size: 11px;">{escape_for_html(evt_label)}</span>'
                f'</div>'
            )
            if event_reasoning:
                validated_event_html += f'<div style="color: #888; font-size: 10px; margin-top: 2px;">{escape_for_html(event_reasoning[:150])}</div>'

        ai_validation_html = (
            '<div class="detail-section">'
            f'<div class="detail-section-title" style="color: #00d4ff;" {_tt("ai_council")}>AI Council Assessment</div>'
            f'<div style="margin-bottom: 6px;">'
            f'<span style="color: #aaa; font-size: 11px;" {_tt("business_importance")}>Business Importance:</span> '
            f'<span style="color: {imp_color}; font-size: 14px; font-weight: 700;">{importance:.0%}</span>'
            f'</div>'
            f'<div style="color: #ccc; font-size: 12px; line-height: 1.5; padding: 6px; background: rgba(0,212,255,0.05); border-radius: 6px; border-left: 3px solid {imp_color};">'
            f'{escape_for_html(validation_reasoning[:300])}</div>'
            f'{validated_event_html}'
            '</div>'
        )

    # V7: Cluster quality section (V4: now includes OSS + sales-actionability grade)
    cluster_quality_html = ""
    quality = (getattr(node, 'signals', {}) or {}).get('cluster_quality', {})
    if quality and isinstance(quality, dict):
        grade = quality.get('quality_grade', '')
        grade_colors = {"A": "#00ff88", "B": "#2ed573", "C": "#ffa502", "D": "#ff6348", "F": "#ff4757"}
        g_color = grade_colors.get(grade, "#888")
        coherence = quality.get('coherence_score', 0)
        entity_con = quality.get('entity_consistency', 0)
        quality_reasoning = quality.get('quality_reasoning', '')
        # V4: OSS + new composite fields
        oss = quality.get('specificity_score', None)
        composite = quality.get('composite_score', None)
        cmi = quality.get('cmi_relevance', None)

        # OSS bar
        oss_html = ""
        if oss is not None:
            oss_color = "#00ff88" if oss >= 0.6 else "#ffa502" if oss >= 0.3 else "#ff4757"
            oss_bar = int(oss * 100)
            oss_label = "High" if oss >= 0.6 else "Medium" if oss >= 0.3 else "Low"
            oss_html = (
                f'<div style="margin-top:6px;">'
                f'<div style="display:flex; justify-content:space-between; font-size:10px; color:#888; margin-bottom:2px;">'
                f'<span>Specificity (OSS)</span>'
                f'<span style="color:{oss_color};">{oss:.2f} — {oss_label}</span>'
                f'</div>'
                f'<div style="background:#1a1a2e; border-radius:3px; height:5px;">'
                f'<div style="width:{oss_bar}%; background:{oss_color}; height:100%; border-radius:3px;"></div>'
                f'</div>'
                f'</div>'
            )

        cluster_quality_html = (
            '<div class="detail-section">'
            '<div class="detail-section-title">Cluster Quality</div>'
            '<div class="scores-grid">'
            f'<div class="score-item"><div class="score-value" style="color: {g_color}; font-size: 22px;">{grade}</div><div class="score-label" {_tt("grade")}>Grade</div></div>'
            f'<div class="score-item"><div class="score-value" style="color: #00d4ff;">{coherence:.2f}</div><div class="score-label" {_tt("coherence")}>Coherence</div></div>'
            f'<div class="score-item"><div class="score-value" style="color: #ffa502;">{entity_con:.0%}</div><div class="score-label" {_tt("entity_match")}>Entity Match</div></div>'
            + (f'<div class="score-item"><div class="score-value" style="color: #a29bfe;">{composite:.2f}</div><div class="score-label">Composite</div></div>' if composite is not None else '')
            + (f'<div class="score-item"><div class="score-value" style="color: #fd79a8;">{cmi:.2f}</div><div class="score-label">CMI</div></div>' if cmi is not None else '')
            + '</div>'
            + oss_html
            + f'<div style="color: #888; font-size: 10px; margin-top: 6px; line-height: 1.4;">{escape_for_html(quality_reasoning[:250])}</div>'
            '</div>'
        )

    # V7: Dynamic credibility section
    credibility_html = ""
    signals = getattr(node, 'signals', {}) or {}
    dyn_cred = signals.get('dynamic_credibility', 0)
    base_auth = signals.get('base_authority', 0)
    cross_cite = signals.get('cross_citation', 0)
    originality = signals.get('originality_score', 0)
    if dyn_cred > 0:
        cred_color = "#00ff88" if dyn_cred >= 0.8 else "#ffa502" if dyn_cred >= 0.6 else "#ff4757"
        credibility_html = (
            '<div class="detail-section">'
            '<div class="detail-section-title">Source Credibility</div>'
            '<div class="scores-grid">'
            f'<div class="score-item"><div class="score-value" style="color: {cred_color};">{dyn_cred:.2f}</div><div class="score-label" {_tt("dynamic_score")}>Dynamic Score</div></div>'
            f'<div class="score-item"><div class="score-value" style="color: #aaa;">{base_auth:.2f}</div><div class="score-label" {_tt("base_tier")}>Base Tier</div></div>'
            f'<div class="score-item"><div class="score-value" style="color: #aaa;">{cross_cite:.2f}</div><div class="score-label" {_tt("cross_citation")}>Cross-Citation</div></div>'
            '</div>'
            f'<div style="color: #666; font-size: 10px; margin-top: 4px;">Formula: 0.40×tier + 0.25×citation + 0.20×originality({originality:.2f}) + 0.15×agreement</div>'
            '</div>'
        )

    # V3: Score breakdown
    breakdown_html = _render_score_breakdown_html(node)

    children_html = ""
    if node.children_ids:
        child_items = []
        for cid in node.children_ids[:6]:
            child = tree.nodes.get(str(cid))
            if child:
                safe_child_title = escape_for_html(truncate_text(child.trend_title, 50))
                child_count = child.article_count
                child_sev = child.severity.value.upper() if hasattr(child.severity, 'value') else str(child.severity).upper()
                sev_colors = {"CRITICAL": "#ff4757", "HIGH": "#ffa502", "MEDIUM": "#888", "LOW": "#666"}
                sev_color = sev_colors.get(child_sev, "#666")
                child_items.append(
                    f'<div class="subtopic-item">'
                    f'<span style="color: #ccc;">{safe_child_title}</span>'
                    f'<span style="float: right; font-size: 10px; color: #666;">{child_count} articles</span>'
                    f'</div>'
                )
        if child_items:
            children_html = f'<div class="detail-section"><div class="detail-section-title">Sub-topics ({len(node.children_ids)})</div>{"".join(child_items)}</div>'

    node_articles = [articles_map.get(str(aid)) for aid in node.source_articles if str(aid) in articles_map]
    node_articles = [a for a in node_articles if a]
    articles_html = ""
    if node_articles:
        art_items = []
        for art in node_articles:
            safe_art_title = escape_for_html(truncate_text(art.title, 65))
            safe_url = html.escape(getattr(art, 'url', '') or '#')
            src = escape_for_html(art.source_name) if art.source_name else ''
            pub_date = art.published_at.strftime("%b %d") if hasattr(art, 'published_at') and art.published_at else ""
            date_prefix = f'<span style="color: #666; font-size: 10px; margin-right: 8px; white-space: nowrap;">{pub_date}</span>' if pub_date else ''
            row_inner = f'{date_prefix}<span style="color: #ccc;">{safe_art_title}</span><span style="color: #555; float: right; font-size: 10px; white-space: nowrap; margin-left: 8px;">{src}</span>'
            if safe_url != '#':
                art_items.append(f'<a href="{safe_url}" target="_blank" style="display: block; padding: 7px 6px; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 11px; text-decoration: none; border-radius: 4px;">{row_inner}</a>')
            else:
                art_items.append(f'<div style="padding: 7px 6px; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 11px;">{row_inner}</div>')
        articles_html = f'<div class="detail-section"><div class="detail-section-title">Source Articles ({len(node_articles)})</div>{"".join(art_items)}</div>'

    # Buying intent section (6sense/Bombora style)
    buying_html = ""
    buying = getattr(node, 'buying_intent', {}) or {}
    if buying and buying.get('signal_type'):
        urgency_colors = {"immediate": "#ff4757", "short_term": "#ffa502", "medium_term": "#2ed573"}
        urg = buying.get('urgency', 'medium_term')
        urg_color = urgency_colors.get(urg, "#999")
        buying_html = (
            '<div class="detail-section">'
            f'<div class="detail-section-title" style="color: #00d4ff;" {_tt("buying_intent")}>Buying Intent</div>'
            '<div style="margin-bottom: 4px;">'
            f'<span style="color: #aaa; font-size: 11px;">Signal:</span> <span style="color: #e0e0e0; font-size: 12px;">{escape_for_html(buying.get("signal_type", "").replace("_", " ").title())}</span>'
            f'<span style="font-size: 10px; background: {urg_color}22; color: {urg_color}; padding: 2px 6px; border-radius: 8px; margin-left: 8px;">{urg.replace("_", " ").upper()}</span>'
            '</div>'
            f'<div style="color: #ccc; font-size: 12px; margin: 4px 0;">{escape_for_html(buying.get("pitch_hook", ""))}</div>'
            f'<div style="color: #888; font-size: 11px;">Target: {escape_for_html(buying.get("who_needs_help", ""))}</div>'
            f'<div style="color: #888; font-size: 11px;">Need: {escape_for_html(buying.get("what_they_need", ""))}</div>'
            '</div>'
        )

    # 5W1H section
    w5h1_html = ""
    w5h1 = getattr(node, 'event_5w1h', {}) or {}
    if w5h1 and any(w5h1.values()):
        w5h1_items = []
        labels = [("who", "Who"), ("what", "What"), ("whom", "Whom"), ("when", "When"), ("where", "Where"), ("why", "Why")]
        for key, label in labels:
            val = w5h1.get(key, '')
            if val and val.lower() not in ('not specified', 'n/a', 'unknown', 'none'):
                w5h1_items.append(f'<div style="padding: 2px 0;"><span style="color: #7c3aed; font-weight: 600; font-size: 11px;">{label}:</span> <span style="color: #aaa; font-size: 11px;">{escape_for_html(val[:100])}</span></div>')
        if w5h1_items:
            w5h1_html = f'<div class="detail-section"><div class="detail-section-title" style="color: #7c3aed;" {_tt("event_5w1h")}>Event Analysis (5W1H)</div>{"".join(w5h1_items)}</div>'

    # Causal chain section
    chain_html = ""
    chain = getattr(node, 'causal_chain', []) or []
    if chain:
        chain_items = ''.join([f'<div style="padding: 4px 0 4px 12px; border-left: 2px solid #ffa502; margin-left: 4px; color: #aaa; font-size: 11px; line-height: 1.4;">{escape_for_html(str(c))}</div>' for c in chain[:6]])
        chain_html = f'<div class="detail-section"><div class="detail-section-title" style="color: #ffa502;" {_tt("causal_chain")}>Causal Chain</div>{chain_items}</div>'

    sidebar_html = (
        '<div class="detail-sidebar" id="trend-detail-sidebar">'
        '<div class="detail-sidebar-close" id="sidebar-close-x">✕</div>'
        '<div class="detail-sidebar-header">'
        f'<h3 class="detail-sidebar-title">{safe_title}</h3>'
        '<div class="detail-sidebar-badges">'
        f'<span class="depth-badge {depth_class}" {_tt("depth_badge")}>{depth_label}</span>'
        f'<span class="{sig_class}" {_tt("signal_badge")}>{sig}</span>'
        f'<span style="background: rgba(255,255,255,0.08); color: #aaa; padding: 3px 8px; border-radius: 10px; font-size: 10px;" {_tt("severity_badge")}>{severity_val}</span>'
        f'<span style="background: rgba(0,212,255,0.15); color: #00d4ff; padding: 3px 8px; border-radius: 10px; font-size: 10px;" {_tt("article_count")}>{node.article_count} articles</span>'
        '</div></div>'
        f'<div class="detail-sidebar-summary">{safe_summary}</div>'
        f'{ai_validation_html}{buying_html}{insight_html}{sparkline_html}{w5h1_html}{chain_html}{ent_html}{scores_html}{confidence_html}{bridges_html}{cluster_quality_html}{credibility_html}{breakdown_html}{children_html}{articles_html}'
        '</div>'
    )
    st.markdown(sidebar_html, unsafe_allow_html=True)

    # Wire the HTML X button to the Streamlit close button via JS
    st_components.html("""<script>
(function(){
try {
    var doc = window.parent.document;
    var x = doc.getElementById('sidebar-close-x');
    if (!x) return;
    x.addEventListener('click', function(){
        var btns = doc.querySelectorAll('button');
        for (var i = 0; i < btns.length; i++) {
            if (btns[i].textContent.trim() === '✕ Close') {
                btns[i].click(); return;
            }
        }
    });
    // Hide the Streamlit button since the X is inside the sidebar
    var btns = doc.querySelectorAll('button');
    for (var i = 0; i < btns.length; i++) {
        if (btns[i].textContent.trim() === '✕ Close') {
            var el = btns[i].closest('[data-testid="stElementContainer"]');
            if (el) el.style.display = 'none';
            break;
        }
    }
} catch(e) { /* fallback: user can use the Streamlit button */ }
})();
</script>""", height=0)

    # Interactive sub-topic navigation (Streamlit buttons below content)
    if node.children_ids:
        child_nodes = [tree.nodes.get(str(cid)) for cid in node.children_ids[:6] if tree.nodes.get(str(cid))]
        if child_nodes:
            st.markdown("**Explore Sub-topics:**")
            nav_cols = st.columns(min(len(child_nodes), 3))
            for i, child in enumerate(child_nodes):
                with nav_cols[i % min(len(child_nodes), 3)]:
                    if st.button(
                        f"{truncate_text(child.trend_title, 35)}",
                        key=f"subtopic_nav_{child.id}",
                        use_container_width=True,
                    ):
                        st.session_state["selected_trend_id"] = str(child.id)
                        st.rerun()

    # Back to parent navigation
    if node.parent_id:
        parent_node = tree.nodes.get(str(node.parent_id))
        if parent_node:
            if st.button(f"← Back to {truncate_text(parent_node.trend_title, 40)}", key="subtopic_back_nav"):
                st.session_state["selected_trend_id"] = str(node.parent_id)
                st.rerun()


def build_flow_state(tree, node_filter=None):
    """Convert TrendTree to StreamlitFlowState for visualization."""
    if not FLOW_AVAILABLE or not tree or not tree.nodes:
        return None

    nodes = []
    edges = []
    depth_colors = {
        1: {"bg": "#1e3a5f", "border": "#00d4ff", "text": "#e0e0e0"},
        2: {"bg": "#2d1e5f", "border": "#7c3aed", "text": "#d0d0e0"},
        3: {"bg": "#1e1e4f", "border": "#6366f1", "text": "#c0c0d0"},
    }
    signal_colors = {"strong": "#2ed573", "weak": "#ffa502", "noise": "#57606f"}

    for node_id_str, node in tree.nodes.items():
        if node_filter and not node_filter(node):
            continue
        depth = min(node.depth, 3)
        colors = depth_colors.get(depth, depth_colors[1])
        sig_color = signal_colors.get(str(node.signal_strength), signal_colors["noise"])
        has_parent = node.parent_id is not None
        has_children = bool(node.children_ids)
        node_type = "input" if not has_parent else ("output" if not has_children else "default")

        depth_label = {1: "MAJOR", 2: "SUB", 3: "MICRO"}.get(node.depth, "TREND")
        sig = str(node.signal_strength).upper()
        title = truncate_text(node.trend_title, 35)
        content = f"**{title}**\n\n`{depth_label}` `{sig}` `{node.article_count} articles`"

        flow_node = StreamlitFlowNode(
            id=node_id_str, pos=(0, 0), data={"content": content},
            node_type=node_type, source_position="right", target_position="left", draggable=True,
            style={"background": colors["bg"], "color": colors["text"], "border": f"2px solid {colors['border']}",
                   "borderRadius": "8px", "padding": "10px", "minWidth": "180px", "maxWidth": "220px", "fontSize": "12px",
                   "boxShadow": f"0 4px 12px rgba(0, 0, 0, 0.3), 0 0 0 2px {sig_color}33"},
        )
        nodes.append(flow_node)

    for node_id_str, node in tree.nodes.items():
        if node_filter and not node_filter(node):
            continue
        for child_id in node.children_ids:
            child_id_str = str(child_id)
            child_node = tree.nodes.get(child_id_str)
            if child_node and (not node_filter or node_filter(child_node)):
                edge = StreamlitFlowEdge(id=f"{node_id_str}-{child_id_str}", source=node_id_str, target=child_id_str,
                                         animated=False, edge_type="smoothstep", style={"stroke": "#555", "strokeWidth": 2},
                                         marker_end={"type": "arrowclosed"})
                edges.append(edge)

    return StreamlitFlowState(nodes=nodes, edges=edges) if nodes else None


def render_bubble_map(tree, articles_map: dict, node_filter=None):
    """Render bubble map visualization."""
    import plotly.graph_objects as go
    import math

    filtered_nodes = [n for n in tree.nodes.values() if not node_filter or node_filter(n)]
    if not filtered_nodes:
        st.info("No trends match the current filters.")
        return []

    filtered_nodes.sort(key=lambda n: n.article_count, reverse=True)
    signal_colors = {"strong": "#2ed573", "weak": "#ffa502", "noise": "#57606f"}
    max_articles = max(n.article_count for n in filtered_nodes)
    positions, radii = [], []

    for i, node in enumerate(filtered_nodes):
        ratio = math.sqrt(node.article_count / max_articles) if max_articles > 0 else 0.5
        radius = 30 + ratio * 90
        radii.append(radius)
        angle = i * 0.8
        r = 50 + i * 25 + radius * 0.5
        positions.append((r * math.cos(angle), r * math.sin(angle)))

    x_vals = [p[0] for p in positions]
    y_vals = [p[1] for p in positions]
    sizes = [r * 2 for r in radii]
    colors = [signal_colors.get(str(n.signal_strength).lower(), "#57606f") for n in filtered_nodes]
    hover_texts = [f"<b>{n.trend_title}</b><br>Signal: {str(n.signal_strength).upper()}<br>Articles: {n.article_count}" for n in filtered_nodes]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode="markers+text",
        marker=dict(size=sizes, color=colors, line=dict(width=2, color="#1a1a3e"), opacity=0.85),
        text=[f"<b>{truncate_text(n.trend_title, 18)}</b><br>{n.article_count} articles" for n in filtered_nodes],
        textposition="middle center", textfont=dict(size=10, color="white"),
        hovertext=hover_texts, hoverinfo="text", customdata=[str(n.id) for n in filtered_nodes],
    ))
    fig.update_layout(showlegend=False, margin=dict(t=20, l=20, r=20, b=20),
                      paper_bgcolor="rgba(10, 10, 26, 1)", plot_bgcolor="rgba(10, 10, 26, 1)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"), height=600)

    selected = st.plotly_chart(fig, use_container_width=True, key="trend_bubble_map", on_select="rerun", selection_mode="points")
    if selected and selected.selection and selected.selection.point_indices:
        idx = selected.selection.point_indices[0]
        if idx < len(filtered_nodes):
            st.session_state["selected_trend_id"] = str(filtered_nodes[idx].id)

    if "selected_trend_id" in st.session_state:
        selected_node = tree.nodes.get(st.session_state["selected_trend_id"])
        if selected_node and (not node_filter or node_filter(selected_node)):
            render_sliding_sidebar(selected_node, tree, articles_map)
    return []


def render_flow_tree(tree, articles_map: dict, node_filter=None):
    """Render horizontal flow tree visualization."""
    if not FLOW_AVAILABLE:
        st.warning("Flow visualization not available. Install: pip install streamlit-flow-component")
        return []

    flow_key = "trend_flow_state"
    if flow_key not in st.session_state or st.session_state.get("_flow_tree_id") != id(tree):
        flow_state = build_flow_state(tree, node_filter)
        if flow_state:
            st.session_state[flow_key] = flow_state
            st.session_state["_flow_tree_id"] = id(tree)

    flow_state = st.session_state.get(flow_key)
    if not flow_state:
        st.info("No trends to display in flow view.")
        return []

    st.markdown("**Click a trend node to see details**")
    updated_state = streamlit_flow(
        key="trend_tree_flow", state=flow_state, layout=TreeLayout(direction="right", node_node_spacing=100),
        height=550, fit_view=True, show_controls=True, show_minimap=True, allow_new_edges=False,
        get_node_on_click=True, get_edge_on_click=False, pan_on_drag=True, allow_zoom=True, min_zoom=0.3,
        hide_watermark=True, style={"background": "linear-gradient(135deg, #0a0a1a 0%, #12122a 100%)", "borderRadius": "10px", "border": "1px solid #333"},
    )
    st.session_state[flow_key] = updated_state

    selected_id = updated_state.selected_id if updated_state else None
    if selected_id and selected_id in tree.nodes:
        st.session_state["selected_trend_id"] = selected_id

    if "selected_trend_id" in st.session_state:
        selected_node = tree.nodes.get(st.session_state["selected_trend_id"])
        if selected_node and (not node_filter or node_filter(selected_node)):
            render_sliding_sidebar(selected_node, tree, articles_map)
    return []


# ══════════════════════════════════════════════════════════════════════════════
# V6: SECTOR HEATMAP — Meltwater ranking grid style
# ══════════════════════════════════════════════════════════════════════════════

def render_sector_heatmap(tree, node_filter=None):
    """
    Render sector x signal strength heatmap.

    Shows concentration of strong/weak/noise trends per sector.
    Validated by: Meltwater — "ranking grid to spotlight the most
    significant changes across brands, products, people, locations."
    """
    import plotly.graph_objects as go
    from collections import defaultdict

    if not tree or not tree.nodes:
        return

    # Collect sector → signal counts
    sector_signals: dict = defaultdict(lambda: {"strong": 0, "weak": 0, "noise": 0})

    for node in tree.nodes.values():
        if node_filter and not node_filter(node):
            continue
        sig = str(node.signal_strength).lower()
        if sig not in ("strong", "weak", "noise"):
            sig = "noise"
        sectors = node.primary_sectors
        if not sectors:
            sectors = ["Unclassified"]
        for s in sectors:
            sector_name = s.value if hasattr(s, 'value') else str(s)
            sector_signals[sector_name][sig] += 1

    if not sector_signals:
        return

    # Sort sectors by total trend count
    sorted_sectors = sorted(
        sector_signals.items(),
        key=lambda x: sum(x[1].values()),
        reverse=True
    )[:12]  # Cap at 12 sectors

    sector_names = [s[0] for s in sorted_sectors]
    signal_types = ["strong", "weak", "noise"]
    z_data = [[sorted_sectors[i][1][sig] for i in range(len(sorted_sectors))] for sig in signal_types]

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=sector_names,
        y=["Strong", "Weak", "Noise"],
        colorscale=[[0, "#0a0a1a"], [0.5, "#1a4a6e"], [1, "#00d4ff"]],
        showscale=False,
        text=z_data,
        texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
    ))

    fig.update_layout(
        height=180,
        margin=dict(t=10, l=60, r=10, b=30),
        paper_bgcolor="rgba(10, 10, 26, 0)",
        plot_bgcolor="rgba(10, 10, 26, 0)",
        xaxis=dict(tickangle=-45, tickfont=dict(size=10, color="#888")),
        yaxis=dict(tickfont=dict(size=10, color="#888")),
    )

    st.plotly_chart(fig, use_container_width=True, key="sector_heatmap")
