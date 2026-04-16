"""
Dashboard Theme — 3D light-on-dark trading dashboard.
Bright elevated cards on a medium-dark background with depth shadows.
Import and call apply_theme() at the top of every page.
"""
import streamlit as st


# ── Color Palette ─────────────────────────────────────────────────────────
COLORS = {
    "bg": "#1a1f2e",
    "card": "#ffffff",
    "card_alt": "#f1f5f9",
    "accent": "#6366f1",
    "accent2": "#06b6d4",
    "green": "#10b981",
    "red": "#ef4444",
    "orange": "#f59e0b",
    "text": "#e2e8f0",
    "text_dark": "#1e293b",
    "text_muted": "#94a3b8",
    "chart_bg": "#232a3b",
}

# ── Plotly template ───────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#232a3b",
    font=dict(color="#cbd5e1", family="Inter, system-ui, sans-serif", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1")),
    hoverlabel=dict(bgcolor="#ffffff", font_color="#1e293b", bordercolor="#e2e8f0"),
)


def apply_theme():
    """Inject the 3D light-on-dark CSS."""
    st.markdown(_CSS, unsafe_allow_html=True)


def styled_metric(label: str, value: str, delta: str = "", delta_color: str = "normal"):
    """Render a 3D elevated metric card with light background."""
    delta_class = ""
    if delta:
        if delta_color == "green" or (delta_color == "normal" and not delta.startswith("-")):
            delta_class = "metric-delta-green"
        elif delta_color == "red" or (delta_color == "normal" and delta.startswith("-")):
            delta_class = "metric-delta-red"
        else:
            delta_class = "metric-delta-muted"

    delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="card-3d metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def status_badge(text: str, variant: str = "default"):
    """Render a 3D pill badge. variant: success, danger, warning, info, default."""
    st.markdown(f'<span class="badge-3d badge-{variant}">{text}</span>', unsafe_allow_html=True)


def section_header(title: str, subtitle: str = ""):
    """Render a styled section header."""
    sub = f'<span class="section-subtitle">{subtitle}</span>' if subtitle else ""
    st.markdown(f'<div class="section-header"><h3>{title}</h3>{sub}</div>', unsafe_allow_html=True)


def apply_plotly_theme(fig):
    """Apply the chart theme to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ── Master CSS ────────────────────────────────────────────────────────────
_CSS = """
<style>
/* ── Fonts ──────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Variables ──────────────────────────────────────────────────── */
:root {
    --bg: #1a1f2e;
    --bg-raised: #232a3b;
    --card: #ffffff;
    --card-alt: #f1f5f9;
    --accent: #6366f1;
    --accent-light: #818cf8;
    --accent2: #06b6d4;
    --green: #10b981;
    --red: #ef4444;
    --orange: #f59e0b;
    --text-light: #e2e8f0;
    --text-dark: #1e293b;
    --text-muted-dark: #64748b;
    --text-muted-light: #94a3b8;

    /* 3D shadow layers */
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.2), 0 2px 4px rgba(0,0,0,0.15);
    --shadow-md: 0 4px 8px rgba(0,0,0,0.2), 0 8px 16px rgba(0,0,0,0.15), 0 1px 0 rgba(255,255,255,0.05) inset;
    --shadow-lg: 0 8px 16px rgba(0,0,0,0.25), 0 16px 32px rgba(0,0,0,0.2), 0 1px 0 rgba(255,255,255,0.08) inset;
    --shadow-xl: 0 12px 24px rgba(0,0,0,0.3), 0 24px 48px rgba(0,0,0,0.2), 0 1px 0 rgba(255,255,255,0.1) inset;
    --shadow-float: 0 20px 40px rgba(0,0,0,0.35), 0 2px 0 rgba(255,255,255,0.1) inset;

    /* Inset shadow for depth */
    --shadow-inset: inset 0 2px 4px rgba(0,0,0,0.15), inset 0 -1px 0 rgba(255,255,255,0.03);
}

/* ── Global ─────────────────────────────────────────────────────── */
.stApp {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background: linear-gradient(160deg, #1a1f2e 0%, #151929 50%, #1e2540 100%) !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em;
}

/* ── 3D Light cards (white on dark) ─────────────────────────────── */
.card-3d {
    background: linear-gradient(145deg, #ffffff, #f1f5f9);
    border: 1px solid rgba(255,255,255,0.8);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 12px;
    box-shadow: var(--shadow-lg);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    transform: translateY(0) perspective(600px) rotateX(0);
}

.card-3d:hover {
    transform: translateY(-4px) perspective(600px) rotateX(1deg);
    box-shadow: var(--shadow-xl);
}

/* ── Glass card (dark translucent) ──────────────────────────────── */
.glass-card {
    background: linear-gradient(145deg, rgba(35,42,59,0.9), rgba(26,31,46,0.7));
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08);
    border-top: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 12px;
    box-shadow: var(--shadow-md);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-lg);
    border-color: rgba(99,102,241,0.3);
}

/* ── Metric cards (3D light) ────────────────────────────────────── */
.metric-card {
    text-align: center;
    padding: 18px 14px;
}

.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-muted-dark);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 1.45rem;
    font-weight: 800;
    color: var(--text-dark);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.03em;
    text-shadow: 0 1px 0 rgba(255,255,255,0.8);
}

.metric-delta {
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 6px;
    font-family: 'JetBrains Mono', monospace;
}

.metric-delta-green { color: var(--green); }
.metric-delta-red { color: var(--red); }
.metric-delta-muted { color: var(--text-muted-dark); }

/* ── 3D Badges ──────────────────────────────────────────────────── */
.badge-3d {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 24px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    box-shadow: var(--shadow-sm);
    transition: all 0.2s ease;
}

.badge-3d:hover { transform: translateY(-1px); box-shadow: var(--shadow-md); }

.badge-success {
    background: linear-gradient(145deg, #10b981, #059669);
    color: white;
    border: 1px solid #10b981;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2);
}

.badge-danger {
    background: linear-gradient(145deg, #ef4444, #dc2626);
    color: white;
    border: 1px solid #ef4444;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2);
}

.badge-warning {
    background: linear-gradient(145deg, #f59e0b, #d97706);
    color: white;
    border: 1px solid #f59e0b;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2);
}

.badge-info {
    background: linear-gradient(145deg, #6366f1, #4f46e5);
    color: white;
    border: 1px solid #6366f1;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2);
}

.badge-default {
    background: linear-gradient(145deg, #475569, #334155);
    color: #e2e8f0;
    border: 1px solid #475569;
}

/* ── Section headers ────────────────────────────────────────────── */
.section-header {
    margin: 28px 0 18px 0;
    padding-bottom: 14px;
    border-bottom: 2px solid rgba(99,102,241,0.3);
}

.section-header h3 {
    margin: 0;
    color: var(--text-light);
    font-size: 1.2rem;
    text-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

.section-subtitle {
    font-size: 0.82rem;
    color: var(--text-muted-light);
    margin-top: 4px;
    display: block;
}

/* ── Hero header (3D floating) ──────────────────────────────────── */
.hero {
    background: linear-gradient(145deg, #ffffff, #e8ecf4);
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 24px;
    padding: 36px;
    margin-bottom: 28px;
    text-align: center;
    box-shadow: var(--shadow-float);
    transform: perspective(800px) rotateX(1deg);
    transition: all 0.3s ease;
}

.hero:hover {
    transform: perspective(800px) rotateX(0deg) translateY(-2px);
    box-shadow: 0 24px 48px rgba(0,0,0,0.4);
}

.hero h1 {
    background: linear-gradient(135deg, #6366f1, #06b6d4, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.4rem !important;
    margin-bottom: 6px !important;
    text-shadow: none;
}

.hero p {
    color: var(--text-muted-dark);
    font-size: 0.92rem;
    margin: 0;
    font-weight: 500;
}

/* ── Sidebar ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e2536, #232a3b) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    box-shadow: 4px 0 16px rgba(0,0,0,0.3) !important;
}

/* ── Streamlit native metric → 3D style ─────────────────────────── */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #ffffff, #f1f5f9);
    border: 1px solid rgba(255,255,255,0.8);
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: var(--shadow-md);
    transition: all 0.25s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted-dark) !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 800 !important;
    color: var(--text-dark) !important;
}

[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
}

/* ── Expander → 3D container ────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(35,42,59,0.6);
    border: 1px solid rgba(255,255,255,0.08);
    border-top: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}

/* ── DataFrame → elevated container ─────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: var(--shadow-md);
}

/* ── 3D Buttons ─────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 12px;
    font-weight: 700;
    letter-spacing: 0.02em;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: var(--shadow-sm);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

.stButton > button:active {
    transform: translateY(1px);
    box-shadow: 0 1px 2px rgba(0,0,0,0.2);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(145deg, #6366f1, #4f46e5) !important;
    border: 1px solid #6366f1 !important;
    color: white !important;
    font-weight: 700 !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    box-shadow: 0 4px 12px rgba(99,102,241,0.4), var(--shadow-sm);
}

.stButton > button[kind="primary"]:hover {
    box-shadow: 0 8px 24px rgba(99,102,241,0.5), var(--shadow-md);
    transform: translateY(-3px);
}

.stButton > button[kind="primary"]:active {
    transform: translateY(0);
    box-shadow: 0 2px 6px rgba(99,102,241,0.3);
}

/* ── Tabs → 3D ──────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(35,42,59,0.6);
    border-radius: 14px;
    padding: 5px;
    box-shadow: var(--shadow-inset);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    box-shadow: var(--shadow-sm);
}

/* ── Input fields → subtle 3D ───────────────────────────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    border-radius: 10px !important;
    box-shadow: var(--shadow-inset);
}

/* ── Scrollbar ──────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(148,163,184,0.25);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(148,163,184,0.45); }

/* ── Signal colors ──────────────────────────────────────────────── */
.signal-buy { color: var(--green); font-weight: 700; }
.signal-sell { color: var(--red); font-weight: 700; }
.signal-hold { color: var(--orange); font-weight: 700; }
.signal-pass { color: var(--text-muted-light); }

/* ── Divider ────────────────────────────────────────────────────── */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent);
    margin: 24px 0;
}

/* ── Live pulse ─────────────────────────────────────────────────── */
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 4px var(--green); }
    50% { opacity: 0.5; box-shadow: 0 0 12px var(--green); }
}

.live-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse 2s infinite;
    margin-right: 6px;
    vertical-align: middle;
}

/* ── 3D floating animation for cards ────────────────────────────── */
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-3px); }
}

.float-anim {
    animation: float 4s ease-in-out infinite;
}
</style>
"""
