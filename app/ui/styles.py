"""
CSS styles for the CMI Sales Agent Streamlit app.
"""

import streamlit as st


def apply_custom_styles():
    """Apply custom CSS styles to the Streamlit app."""
    st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
    }

    /* Cards */
    .trend-card {
        background: linear-gradient(145deg, #1e1e3f, #2a2a5a);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #00d4ff;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.1);
    }

    .company-card {
        background: linear-gradient(145deg, #1e3f1e, #2a5a2a);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #00ff88;
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.1);
    }

    .contact-card {
        background: linear-gradient(145deg, #3f1e3f, #5a2a5a);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #ff00ff;
        box-shadow: 0 4px 15px rgba(255, 0, 255, 0.1);
    }

    .email-card {
        background: linear-gradient(145deg, #3f3f1e, #5a5a2a);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #ffcc00;
        box-shadow: 0 4px 15px rgba(255, 204, 0, 0.1);
    }

    /* Severity badges */
    .badge-high {
        background: #ff4757;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }

    .badge-medium {
        background: #ffa502;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }

    .badge-low {
        background: #2ed573;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }

    /* Progress steps */
    .step-active {
        color: #00d4ff;
        font-weight: bold;
    }

    .step-completed {
        color: #00ff88;
    }

    /* Sidebar nav buttons for completed steps — look like green text links */
    [data-testid="stSidebar"] button[kind="secondary"] {
        background: transparent !important;
        border: none !important;
        color: #00ff88 !important;
        text-align: left !important;
        padding: 2px 0 !important;
        font-size: 14px !important;
        box-shadow: none !important;
        justify-content: flex-start !important;
    }
    [data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: rgba(0, 255, 136, 0.08) !important;
        text-decoration: underline !important;
    }

    .step-pending {
        color: #666;
    }

    /* Stats */
    .stat-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }

    .stat-number {
        font-size: 2.5em;
        font-weight: bold;
        color: #00d4ff;
    }

    .stat-label {
        color: #888;
        font-size: 0.9em;
    }

    /* Signal strength badges */
    .signal-strong {
        background: #2ed573;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
    }

    .signal-weak {
        background: #ffa502;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
    }

    .signal-noise {
        background: #57606f;
        color: #ccc;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
    }

    /* Pipeline stats bar */
    .pipeline-stats {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 12px 20px;
        margin: 10px 0 20px 0;
        display: flex;
        gap: 30px;
        align-items: center;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .pipeline-stat {
        text-align: center;
    }

    .pipeline-stat-value {
        font-size: 1.6em;
        font-weight: bold;
        color: #00d4ff;
    }

    .pipeline-stat-label {
        font-size: 0.75em;
        color: #888;
    }

    /* Tree visualization */
    .tree-node {
        margin-left: 0;
        padding: 0;
    }

    .tree-node-depth-1 { margin-left: 0; }
    .tree-node-depth-2 { margin-left: 28px; border-left: 2px solid #333; padding-left: 12px; }
    .tree-node-depth-3 { margin-left: 56px; border-left: 2px solid #444; padding-left: 12px; }

    .tree-card {
        background: linear-gradient(145deg, #1e1e3f, #2a2a5a);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
        border-left: 4px solid #00d4ff;
        transition: all 0.2s ease;
    }

    .tree-card:hover {
        border-left-color: #00ff88;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.15);
    }

    .tree-card-sub {
        background: linear-gradient(145deg, #1a1a35, #252550);
        border-left-color: #7c3aed;
    }

    .tree-card-micro {
        background: linear-gradient(145deg, #151530, #1e1e45);
        border-left-color: #6366f1;
    }

    .depth-badge {
        font-size: 10px;
        padding: 2px 8px;
        border-radius: 10px;
        margin-right: 8px;
    }

    .depth-major { background: #00d4ff33; color: #00d4ff; }
    .depth-sub { background: #7c3aed33; color: #a78bfa; }
    .depth-micro { background: #6366f133; color: #818cf8; }

    .tree-toggle {
        cursor: pointer;
        font-size: 14px;
        color: #888;
        margin-right: 8px;
    }

    /* Sidebar styling for trend details */
    [data-testid="stSidebar"] .stButton button {
        margin-bottom: 10px;
    }

    .keyword-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.08);
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        margin: 2px 4px 2px 0;
        color: #ccc;
    }

    .entity-tag {
        display: inline-block;
        background: rgba(59, 130, 246, 0.2);
        border: 1px solid rgba(59, 130, 246, 0.4);
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        margin: 2px 4px 2px 0;
        color: #93c5fd;
    }

    .insight-box {
        background: rgba(0, 255, 136, 0.08);
        border-left: 3px solid #00ff88;
        border-radius: 4px;
        padding: 12px 14px;
        margin-top: 10px;
    }

    .insight-label {
        font-size: 11px;
        font-weight: bold;
        color: #00ff88;
        margin-bottom: 6px;
    }

    .insight-text {
        color: #ccc;
        font-size: 13px;
        line-height: 1.5;
    }

    .scores-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 15px;
        margin-top: 10px;
    }

    .score-item {
        text-align: center;
    }

    .score-value {
        font-size: 1.4em;
        font-weight: bold;
    }

    .score-label {
        font-size: 10px;
        color: #666;
    }

    /* View toggle buttons */
    .view-toggle {
        display: flex;
        gap: 8px;
        margin-bottom: 15px;
    }

    .view-toggle-btn {
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 13px;
        cursor: pointer;
        border: 1px solid rgba(255, 255, 255, 0.15);
        background: rgba(255, 255, 255, 0.05);
        color: #aaa;
        transition: all 0.2s ease;
    }

    .view-toggle-btn.active {
        background: rgba(0, 212, 255, 0.2);
        border-color: #00d4ff;
        color: #00d4ff;
    }

    /* Sliding detail sidebar overlay */
    .detail-sidebar {
        position: fixed;
        top: 0;
        right: 0;
        width: 550px;
        height: 100vh;
        background: linear-gradient(180deg, #12122a 0%, #1a1a3e 100%);
        border-left: 2px solid #00d4ff;
        box-shadow: -10px 0 40px rgba(0, 0, 0, 0.5);
        z-index: 9999;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 20px;
        animation: slideIn 0.3s ease-out;
    }

    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    .detail-sidebar-close {
        position: absolute;
        top: 15px;
        right: 15px;
        background: rgba(255, 255, 255, 0.1);
        border: none;
        color: #aaa;
        font-size: 24px;
        cursor: pointer;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .detail-sidebar-close:hover {
        background: rgba(255, 0, 0, 0.2);
        color: #ff6b6b;
    }

    .detail-sidebar-header {
        margin-bottom: 20px;
        padding-top: 35px;
        padding-right: 40px;
    }

    .detail-sidebar-title {
        font-size: 1.3em;
        font-weight: 600;
        color: #e0e0e0;
        margin: 0 0 12px 0;
        line-height: 1.3;
    }

    .detail-sidebar-badges {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 15px;
    }

    .detail-sidebar-summary {
        color: #bbb;
        font-size: 13px;
        line-height: 1.6;
        margin-bottom: 15px;
    }

    .detail-section {
        margin-top: 15px;
        padding-top: 15px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }

    .detail-section-title {
        font-size: 11px;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    /* Step indicator bar */
    .step-bar {
        display: flex;
        gap: 4px;
        padding: 10px 16px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        margin-bottom: 16px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        align-items: center;
    }

    .step-bar-item {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 13px;
        transition: all 0.2s ease;
    }

    .step-bar-item.active {
        background: rgba(0, 212, 255, 0.15);
        color: #00d4ff;
        font-weight: 600;
    }

    .step-bar-item.completed {
        color: #2ed573;
    }

    .step-bar-item.pending {
        color: #555;
    }

    .step-bar-separator {
        color: #333;
        margin: 0 2px;
        align-self: center;
    }

    /* Impact section labels */
    .impact-section {
        margin-bottom: 14px;
    }

    .impact-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }

    .impact-label-direct { color: #ff4757; }
    .impact-label-indirect { color: #ffa502; }
    .impact-label-pain { color: #ff6b81; }
    .impact-label-consulting { color: #00d4ff; }
    .impact-label-verticals { color: #a78bfa; }
    .impact-label-services { color: #2ed573; }

    .impact-item {
        padding: 3px 0;
        color: #ccc;
        font-size: 13px;
    }

    .impact-item::before {
        content: "•";
        margin-right: 8px;
        font-size: 8px;
    }

    /* Result count badge */
    .result-count {
        display: inline-block;
        background: rgba(0, 212, 255, 0.15);
        color: #00d4ff;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 8px;
    }

    /* Compact log area */
    .log-section [data-testid="stExpander"] {
        margin-top: 4px;
        margin-bottom: 0;
    }

    /* Source article link hover */
    .detail-sidebar a[target="_blank"],
    .stMarkdown a[target="_blank"] {
        transition: background 0.15s ease, color 0.15s ease;
        border-radius: 4px;
        padding-left: 4px;
        padding-right: 4px;
    }

    .detail-sidebar a[target="_blank"]:hover,
    .stMarkdown a[target="_blank"]:hover {
        background: rgba(0, 212, 255, 0.08) !important;
    }

    .detail-sidebar a[target="_blank"]:hover span:first-child {
        color: #00d4ff !important;
    }

    /* Detail sidebar spacing improvements */
    .detail-section {
        margin-top: 18px;
        padding-top: 18px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }

    .detail-section-title {
        font-size: 11px;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }

    .detail-sidebar-summary {
        color: #bbb;
        font-size: 13px;
        line-height: 1.7;
        margin-bottom: 18px;
    }

    /* Sub-topic item in sidebar */
    .subtopic-item {
        padding: 6px 10px;
        color: #bbb;
        font-size: 12px;
        border-radius: 6px;
        cursor: pointer;
        transition: background 0.15s ease;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }

    .subtopic-item:hover {
        background: rgba(255, 255, 255, 0.05);
    }

    /* Related trend item */
    .related-item {
        padding: 4px 8px;
        border-radius: 6px;
        transition: background 0.15s ease;
    }

    .related-item:hover {
        background: rgba(255, 255, 255, 0.05);
    }

    /* Entity tag spacing */
    .entity-tag {
        display: inline-block;
        background: rgba(255, 0, 255, 0.1);
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        margin: 3px 4px 3px 0;
        color: #e0a0ff;
    }

    .keyword-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.08);
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        margin: 3px 4px 3px 0;
        color: #ccc;
    }

    /* ── Metric Tooltips ─────────────────────────────── */
    [data-tooltip] {
        position: relative;
        cursor: help;
        border-bottom: 1px dotted rgba(255,255,255,0.25);
    }
    /* Badges already have distinct styling — no dotted underline */
    .depth-badge[data-tooltip],
    .signal-strong[data-tooltip],
    .signal-weak[data-tooltip],
    .signal-noise[data-tooltip],
    .severity-badge[data-tooltip],
    span[data-tooltip][style*="background"] {
        border-bottom: none;
    }
    /* Section titles with tooltips — no underline */
    .detail-section-title[data-tooltip] {
        border-bottom: none;
    }
    [data-tooltip]::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: calc(100% + 8px);
        left: 50%;
        transform: translateX(-50%);
        background: #0d0d1f;
        color: #ccc;
        font-size: 12.5px;
        font-weight: 400;
        line-height: 1.6;
        letter-spacing: 0.01em;
        text-transform: none !important;
        text-align: left;
        white-space: normal;
        word-wrap: break-word;
        overflow-wrap: break-word;
        min-width: 200px;
        max-width: 380px;
        width: max-content;
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #00d4ff44;
        box-shadow: 0 4px 20px rgba(0,0,0,0.8);
        z-index: 99999;
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.18s ease;
    }
    [data-tooltip]:hover::after { opacity: 1; }

    /* Keep tooltips on-screen when near left/right edges (main content) */
    .scores-grid .score-item:first-child [data-tooltip]::after { left: 0; transform: none; }
    .scores-grid .score-item:last-child [data-tooltip]::after { left: auto; right: 0; transform: none; }

    /* ── Detail sidebar tooltip overrides ──────────── */
    /* Show BELOW the element so scrollable container doesn't clip.
       Hard-cap width to fit the 550px panel (40px padding = 510px usable). */
    .detail-sidebar [data-tooltip]::after {
        bottom: auto;
        top: calc(100% + 6px);
        left: 0;
        right: auto;
        transform: none;
        max-width: min(440px, calc(100vw - 60px));
        min-width: 160px;
    }

    /* Badge row in sidebar: all anchor left, capped width keeps them in bounds */
    .detail-sidebar-badges [data-tooltip]::after {
        max-width: 280px;
        left: 0;
        right: auto;
        transform: none;
    }

    /* Scores grid inside sidebar — position per column so tooltips never overflow */
    .detail-sidebar .scores-grid [data-tooltip]::after {
        max-width: 320px;
    }
    .detail-sidebar .scores-grid .score-item:first-child [data-tooltip]::after {
        left: 0;
        right: auto;
        transform: none;
    }
    .detail-sidebar .scores-grid .score-item:nth-child(2) [data-tooltip]::after {
        left: 50%;
        right: auto;
        transform: translateX(-50%);
    }
    .detail-sidebar .scores-grid .score-item:last-child [data-tooltip]::after {
        left: auto;
        right: 0;
        transform: none;
    }

    /* Breakdown bars: show below the label, anchor left, narrower */
    .breakdown-row [data-tooltip]::after {
        bottom: auto;
        top: calc(100% + 4px);
        left: 0;
        right: auto;
        transform: none;
        max-width: 340px;
    }

    /* Section titles in sidebar (AI Council, 5W1H, etc.) */
    .detail-sidebar .detail-section-title[data-tooltip]::after {
        left: 0;
        transform: none;
        max-width: min(440px, calc(100vw - 60px));
    }

    /* ── Tree card tooltip overrides ───────────────── */
    .tree-card [data-tooltip]::after {
        max-width: 340px;
    }
    .tree-card div[style*="display: flex"] [data-tooltip]::after {
        left: auto;
        right: 0;
        transform: none;
    }
    .tree-card .depth-badge[data-tooltip]::after {
        left: 0;
        right: auto;
        transform: none;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
