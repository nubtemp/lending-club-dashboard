"""
LC Credit Intelligence Platform — v3.0
Internal Fintech Credit Risk Dashboard
Powered by XGBoost · Streamlit · Plotly
"""

import os
import glob
import hashlib
import warnings

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Loan Investment Dashboard | LendingClub AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM  (Bloomberg / Refinitiv dark-fintech palette)
# ══════════════════════════════════════════════════════════════════════════════
THEME = {
    "bg":         "#080e1c",
    "surface":    "#0d1424",
    "surface2":   "#121c30",
    "surface3":   "#19253d",
    "border":     "#1b3358",
    "border2":    "#274d7a",
    "text_pri":   "#e2e8f0",
    "text_sec":   "#8fa8c8",
    "text_muted": "#3d5470",
    "accent":     "#3b82f6",
    "accent_lt":  "#60a5fa",
    "gold":       "#f59e0b",
    "gold_lt":    "#fbbf24",
    "green":      "#10b981",
    "green_lt":   "#34d399",
    "red":        "#ef4444",
    "red_lt":     "#f87171",
    "amber":      "#f59e0b",
    "purple":     "#8b5cf6",
    "teal":       "#06b6d4",
    "chart_bg":   "#0d1424",
    "chart_grid": "#1b3358",
    "chart_font": "#3d5470",
}

# ══════════════════════════════════════════════════════════════════════════════
#  CSS  —  Professional fintech dark theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
}}
.stApp {{ background-color: {THEME['bg']}; color: {THEME['text_pri']}; }}
[data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"],
[data-testid="block-container"] {{ background-color: {THEME['bg']} !important; }}

/* Inputs */
.stTextInput input, .stNumberInput input,
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {{
    background-color: {THEME['surface']} !important;
    color: {THEME['text_pri']} !important;
    border-color: {THEME['border2']} !important;
}}
div[data-baseweb="select"] svg {{ fill: {THEME['text_sec']}; }}
.stSlider > div > div > div {{ background: {THEME['border2']}; }}
.stAlert {{ background: {THEME['surface2']}; color: {THEME['text_pri']}; border-color: {THEME['border']}; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: {THEME['surface']};
    border-right: 1px solid {THEME['border']};
}}
[data-testid="stSidebar"] * {{ color: {THEME['text_pri']}; }}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] p {{
    color: {THEME['text_sec']} !important;
    font-size: 0.73rem !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
[data-testid="stSidebar"] .stSuccess,
[data-testid="stSidebar"] .stInfo {{
    background: {THEME['surface2']};
    border-color: {THEME['border']};
    color: {THEME['text_pri']};
}}

/* Nav bar */
.nav-bar {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 0 14px;
    border-bottom: 1px solid {THEME['border2']};
    margin-bottom: 22px;
}}
.nav-brand {{ display: flex; align-items: baseline; gap: 12px; }}
.nav-title {{
    font-size: 1.05rem; font-weight: 700; color: {THEME['accent']};
    letter-spacing: -0.01em;
}}
.nav-sub {{
    font-size: 0.63rem; color: {THEME['text_muted']};
    font-weight: 400; text-transform: uppercase; letter-spacing: 0.06em;
}}
.nav-badge {{
    font-size: 0.62rem; font-weight: 700; color: {THEME['accent']};
    background: rgba(59,130,246,0.10); border: 1px solid rgba(59,130,246,0.22);
    border-radius: 4px; padding: 3px 9px; letter-spacing: 0.05em;
}}

/* Section headers */
.sec-head {{
    font-size: 0.63rem; font-weight: 700; color: {THEME['accent']};
    text-transform: uppercase; letter-spacing: 0.13em;
    padding-bottom: 8px; border-bottom: 1px solid {THEME['border']};
    margin: 30px 0 14px;
}}

/* KPI grid */
.kpi-grid {{
    display: grid; grid-template-columns: repeat(5, 1fr);
    gap: 1px; background: {THEME['border']};
    border: 1px solid {THEME['border']}; border-radius: 6px;
    overflow: hidden; margin-bottom: 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
}}
.kpi-cell {{ background: {THEME['surface2']}; padding: 16px 18px; }}
.kpi-label {{
    font-size: 0.60rem; font-weight: 700; color: {THEME['text_muted']};
    text-transform: uppercase; letter-spacing: 0.11em; margin-bottom: 7px;
}}
.kpi-value {{
    font-size: 1.42rem; font-weight: 700; color: {THEME['text_pri']};
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.02em; line-height: 1.1;
}}
.kpi-delta {{
    font-size: 0.67rem; margin-top: 5px; color: {THEME['text_muted']};
    font-family: 'JetBrains Mono', monospace;
}}
.kpi-delta.pos {{ color: {THEME['green_lt']}; font-weight: 600; }}
.kpi-delta.neg {{ color: {THEME['red_lt']}; font-weight: 600; }}

/* Executive Summary card */
.exec-card {{
    background: linear-gradient(135deg, {THEME['surface2']} 0%, {THEME['surface3']} 100%);
    border: 1px solid {THEME['border2']};
    border-left: 3px solid {THEME['gold']};
    border-radius: 6px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.25);
}}
.exec-header {{
    display: flex; align-items: center; gap: 10px; margin-bottom: 12px;
}}
.exec-tag {{
    font-size: 0.60rem; font-weight: 700; color: {THEME['gold']};
    text-transform: uppercase; letter-spacing: 0.13em;
    background: rgba(245,158,11,0.10); border: 1px solid rgba(245,158,11,0.22);
    border-radius: 3px; padding: 2px 8px;
}}
.exec-body {{
    font-size: 0.86rem; color: {THEME['text_sec']}; line-height: 1.75;
    margin-bottom: 14px;
}}
.exec-body strong {{ color: {THEME['text_pri']}; font-weight: 600; }}
.exec-insight-grid {{
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;
    margin-top: 2px;
}}
.exec-insight-item {{
    background: rgba(8,14,28,0.60); border: 1px solid {THEME['border']};
    border-radius: 4px; padding: 10px 14px;
}}
.exec-insight-label {{
    font-size: 0.58rem; color: {THEME['text_muted']};
    text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: 4px;
}}
.exec-insight-value {{
    font-size: 0.82rem; color: {THEME['text_pri']}; font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}}
.exec-insight-sub {{
    font-size: 0.66rem; color: {THEME['text_muted']}; margin-top: 2px;
    font-family: 'JetBrains Mono', monospace;
}}

/* Feature description cards */
.feat-card {{
    background: {THEME['surface2']}; border: 1px solid {THEME['border']};
    border-radius: 4px; padding: 9px 12px; margin-bottom: 5px;
}}
.feat-card-name {{
    font-size: 0.72rem; font-weight: 600; color: {THEME['accent_lt']};
    margin-bottom: 3px;
}}
.feat-card-rank {{
    font-size: 0.60rem; font-weight: 700; color: {THEME['gold']};
    font-family: 'JetBrains Mono', monospace;
    float: right;
}}
.feat-card-desc {{
    font-size: 0.73rem; color: {THEME['text_sec']}; line-height: 1.5;
}}

/* Status strip */
.status-strip {{
    display: flex; align-items: center; gap: 20px;
    background: {THEME['surface']}; border: 1px solid {THEME['border']};
    border-left: 3px solid {THEME['green']};
    border-radius: 5px; padding: 8px 15px;
    font-size: 0.69rem; color: {THEME['text_muted']};
    margin-bottom: 18px; font-family: 'JetBrains Mono', monospace;
}}
.status-dot {{
    width: 6px; height: 6px; border-radius: 50%;
    background: {THEME['green']}; display: inline-block; margin-right: 5px;
}}
.status-item span {{ color: {THEME['text_pri']}; font-weight: 600; }}

/* Filter strip */
.filter-strip {{
    display: flex; align-items: center; gap: 8px;
    background: rgba(59,130,246,0.05); border: 1px solid rgba(59,130,246,0.15);
    border-radius: 4px; padding: 6px 12px;
    font-size: 0.68rem; color: {THEME['text_muted']};
    margin-bottom: 14px; font-family: 'JetBrains Mono', monospace;
}}
.filter-badge {{
    background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.25);
    border-radius: 3px; padding: 1px 7px; color: {THEME['accent_lt']};
    font-size: 0.63rem; font-weight: 600;
}}

/* Risk pills */
.risk-pill {{
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 0.66rem; font-weight: 700; padding: 2px 9px;
    border-radius: 3px; font-family: 'JetBrains Mono', monospace;
}}
.risk-low  {{ background: #052e16; color: {THEME['green_lt']};  border: 1px solid #166534; }}
.risk-mid  {{ background: #1c1408; color: {THEME['amber']};     border: 1px solid #92400e; }}
.risk-high {{ background: #1c0808; color: {THEME['red_lt']};    border: 1px solid #7f1d1d; }}

/* Info box */
.info-box {{
    background: rgba(59,130,246,0.06); border: 1px solid rgba(59,130,246,0.16);
    border-left: 2px solid {THEME['accent']};
    border-radius: 4px; padding: 8px 13px;
    font-size: 0.76rem; color: {THEME['text_sec']}; margin-bottom: 12px;
}}

/* Sidebar section divider */
.sb-section {{
    font-size: 0.59rem; font-weight: 700; color: {THEME['text_muted']};
    text-transform: uppercase; letter-spacing: 0.13em;
    padding: 12px 0 5px; border-top: 1px solid {THEME['border']};
    margin-top: 8px;
}}

/* Buttons */
.stButton > button {{
    background-color: transparent !important; color: {THEME['text_pri']} !important;
    border: 1.5px solid {THEME['border2']} !important;
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    border-radius: 5px !important; transition: all 0.15s !important;
}}
.stButton > button:hover {{
    background-color: rgba(59,130,246,0.10) !important;
    border-color: {THEME['accent']} !important; color: {THEME['accent_lt']} !important;
}}
.stDownloadButton > button {{
    background-color: transparent !important; color: {THEME['text_pri']} !important;
    border: 1.5px solid {THEME['border2']} !important;
    font-weight: 600 !important; border-radius: 5px !important;
}}
.stDownloadButton > button:hover {{
    background-color: rgba(59,130,246,0.10) !important;
    border-color: {THEME['accent']} !important;
}}
.stDataFrame {{ border: 1px solid {THEME['border']} !important; border-radius: 4px !important; }}

/* Force dark regardless of system theme */
[data-theme="light"] .stApp,
[data-theme="light"] [data-testid="stAppViewContainer"],
[data-theme="light"] [data-testid="stMainBlockContainer"],
[data-theme="light"] [data-testid="block-container"] {{
    background-color: {THEME['bg']} !important; color: {THEME['text_pri']} !important;
}}
@media (prefers-color-scheme: light) {{
    .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stMainBlockContainer"],
    [data-testid="block-container"] {{
        background-color: {THEME['bg']} !important; color: {THEME['text_pri']} !important;
    }}
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════
PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=THEME["chart_bg"],
    plot_bgcolor=THEME["chart_bg"],
    font=dict(family="Inter", color=THEME["chart_font"], size=11),
    margin=dict(t=44, b=28, l=12, r=12),
    xaxis=dict(
        gridcolor=THEME["chart_grid"], linecolor=THEME["border"],
        tickfont=dict(size=10, color=THEME["chart_font"]),
        title_font=dict(size=11, color=THEME["text_sec"]),
        showline=True, mirror=False,
    ),
    yaxis=dict(
        gridcolor=THEME["chart_grid"], linecolor=THEME["border"],
        tickfont=dict(size=10, color=THEME["chart_font"]),
        title_font=dict(size=11, color=THEME["text_sec"]),
        showline=False,
    ),
    title_font=dict(size=13, color=THEME["text_pri"], family="Inter"),
    title_x=0,
    hoverlabel=dict(
        bgcolor=THEME["surface3"], bordercolor=THEME["border2"],
        font=dict(family="JetBrains Mono", size=11, color=THEME["text_pri"]),
    ),
)
_LEG = dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
            font=dict(size=10, color=THEME["text_sec"]))
LEGEND_V = dict(legend={**_LEG})
LEGEND_H = dict(legend={**_LEG, "orientation": "h", "y": -0.28})


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
REQUIRED  = ["loan_amnt", "int_rate", "annual_inc", "dti", "loan_status"]
OPTIONAL  = ["grade", "term", "purpose", "emp_length", "home_ownership",
             "open_acc", "revol_util", "total_acc", "addr_state",
             "fico_range_low", "fico_range_high", "delinq_2yrs", "inq_last_6mths"]
N_SAMPLE  = 80_000
STRATS    = [
    "All Loans", "Low-Risk Loans",
    "High-Interest Loans", "Optimized (Max Projected Profit)"
]

# Human-readable financial labels for feature names
FEATURE_LABELS = {
    "loan_amnt":      "Loan Size",
    "int_rate":       "Interest Rate",
    "annual_inc":     "Annual Income",
    "dti":            "Debt-to-Income Ratio",
    "revol_util":     "Revolving Utilization",
    "open_acc":       "Open Credit Lines",
    "total_acc":      "Total Credit Accounts",
    "term_months":    "Loan Term",
    "grade_num":      "Credit Grade",
    "owns_home":      "Homeownership",
    "fico_avg":       "FICO Score",
    "delinq_2yrs":    "Delinquencies (2yr)",
    "inq_last_6mths": "Recent Inquiries (6mo)",
    "inc_to_loan":    "Income Coverage Ratio",
}

# Financial context for each feature (analyst notes panel)
FEATURE_DESC = {
    "loan_amnt":      "Larger loans mean more money at stake if the borrower stops paying. Bigger loans can lead to bigger losses.",
    "int_rate":       "Higher interest rates mean higher profit potential — but lenders charge more precisely because these borrowers are riskier.",
    "annual_inc":     "Borrowers with higher incomes are generally better able to keep up with loan payments, even if life gets difficult.",
    "dti":            "This compares a borrower's debt to their income. High debt relative to income makes it harder to repay — a key warning sign.",
    "revol_util":     "How much of their available credit a borrower is using. Using most of it (above 70%) often signals financial stress.",
    "open_acc":       "The number of active credit accounts. Too many can suggest the borrower is over-extended across multiple lenders.",
    "total_acc":      "Borrowers with a longer credit history (more accounts over time) tend to be more financially stable and predictable.",
    "term_months":    "Longer-term loans (60 months vs 36) carry more risk because there's a longer window for things to go wrong.",
    "grade_num":      "LendingClub's own risk rating for each borrower (A = safest, G = riskiest). This is the single strongest predictor of default.",
    "owns_home":      "Homeowners tend to be more financially stable than renters, and historically default on loans less often.",
    "fico_avg":       "A credit score used by lenders. Higher scores (740+) indicate a strong track record of repaying debts on time.",
    "delinq_2yrs":    "Past missed payments are a strong signal of future missed payments. Prior delinquency is one of the top risk indicators.",
    "inq_last_6mths": "Multiple recent credit checks suggest the borrower has been actively seeking new credit, which can indicate financial difficulty.",
    "inc_to_loan":    "How large the loan is relative to the borrower's annual income. A smaller ratio means they can more comfortably repay.",
}


# ══════════════════════════════════════════════════════════════════════════════
#  SAMPLE DATA GENERATOR  (demo mode — no upload required)
# ══════════════════════════════════════════════════════════════════════════════
def generate_sample_data(n: int = 6_000) -> pd.DataFrame:
    """Generate synthetic LendingClub-style sample data for demo exploration."""
    rng = np.random.default_rng(42)
    grades = rng.choice(
        ["A","B","C","D","E","F","G"], n,
        p=[0.18, 0.25, 0.22, 0.16, 0.10, 0.06, 0.03]
    )
    grade_rate    = {"A":0.065,"B":0.095,"C":0.125,"D":0.155,"E":0.185,"F":0.215,"G":0.245}
    grade_default = {"A":0.04, "B":0.07, "C":0.12, "D":0.18, "E":0.25, "F":0.32, "G":0.40}

    loan_amnt  = (rng.uniform(1_000, 40_000, n) / 100).round() * 100
    int_rate   = np.array([grade_rate[g] + rng.uniform(-0.012, 0.012) for g in grades])
    annual_inc = np.exp(rng.normal(10.8, 0.55, n)).round(-2)
    dti        = np.clip(rng.normal(18, 9, n), 2, 48).round(1)

    def_prob   = np.array([grade_default[g] for g in grades])
    def_prob  += rng.uniform(-0.04, 0.04, n)
    def_prob   = np.clip(def_prob, 0.01, 0.90)

    outcome    = rng.random(n)
    loan_status = np.where(
        outcome < def_prob,
        rng.choice(["Charged Off","Default","Late (31-120 days)"], n),
        "Fully Paid"
    )

    terms   = rng.choice(["36 months","60 months"], n, p=[0.60, 0.40])
    purposes = rng.choice(
        ["debt_consolidation","credit_card","home_improvement","other","medical","car"],
        n, p=[0.45, 0.25, 0.12, 0.08, 0.05, 0.05]
    )
    states   = rng.choice(
        ["CA","TX","NY","FL","IL","PA","OH","GA","NC","MI"], n
    )
    fico_low   = rng.integers(620, 800, n)
    home_own   = rng.choice(["RENT","OWN","MORTGAGE"], n, p=[0.40, 0.15, 0.45])
    revol_util = np.clip(rng.normal(48, 22, n), 1, 99).round(1)
    open_acc   = rng.integers(3, 25, n)
    total_acc  = open_acc + rng.integers(0, 20, n)
    delinq     = rng.choice([0,1,2,3], n, p=[0.75, 0.15, 0.07, 0.03])
    inq        = rng.choice([0,1,2,3,4], n, p=[0.45, 0.30, 0.15, 0.07, 0.03])

    return pd.DataFrame({
        "loan_amnt":      loan_amnt,
        "int_rate":       (int_rate * 100).round(2),
        "annual_inc":     annual_inc,
        "dti":            dti,
        "loan_status":    loan_status,
        "grade":          grades,
        "term":           terms,
        "purpose":        purposes,
        "addr_state":     states,
        "fico_range_low": fico_low,
        "fico_range_high":fico_low + 4,
        "home_ownership": home_own,
        "revol_util":     revol_util,
        "open_acc":       open_acc,
        "total_acc":      total_acc,
        "delinq_2yrs":    delinq,
        "inq_last_6mths": inq,
    })


# ══════════════════════════════════════════════════════════════════════════════
#  DATA DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════
APP_DIR = os.path.dirname(os.path.abspath(__file__))


def find_local_csv() -> "str | None":
    patterns = [
        os.path.join(APP_DIR, "accepted*.csv"),
        os.path.join(APP_DIR, "lending*.csv"),
        os.path.join(APP_DIR, "loan*.csv"),
        os.path.join(APP_DIR, "**", "accepted*.csv"),
        os.path.join(APP_DIR, "**", "lending*.csv"),
    ]
    for pat in patterns:
        matches = [m for m in glob.glob(pat, recursive=True) if os.path.isfile(m)]
        big = [m for m in matches if os.path.getsize(m) > 1_000_000]
        if big:
            return sorted(big, key=os.path.getsize, reverse=True)[0]
        if matches:
            return matches[0]
    return None


LOCAL_CSV = find_local_csv()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Part 1: controls that don't need data
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        f"<div style='padding:12px 0 4px;font-size:0.88rem;font-weight:700;"
        f"color:{THEME['text_pri']};letter-spacing:-0.01em;'>"
        f"💳 Loan Investment Dashboard</div>"
        f"<div style='font-size:0.63rem;color:{THEME['text_muted']};padding-bottom:12px;"
        f"border-bottom:1px solid {THEME['border']};'>AI-Powered · LendingClub Data · v3.0</div>",
        unsafe_allow_html=True,
    )

    # Data source
    st.markdown("<div class='sb-section'>Data Source</div>", unsafe_allow_html=True)
    if LOCAL_CSV:
        fname = os.path.basename(LOCAL_CSV)
        fsize = os.path.getsize(LOCAL_CSV) / 1e9
        st.success(
            f"**Auto-detected:** `{fname[:38]}`\n\n"
            f"`{fsize:.2f} GB` · sampled {N_SAMPLE:,} rows"
        )
        use_local  = st.toggle("Use local dataset", value=True)
        use_sample = False
        uploaded   = None if use_local else st.file_uploader("Upload CSV", type=["csv"])
    else:
        use_local  = False
        use_sample = st.toggle(
            "Explore with Sample Data",
            value=True,
            help="Try the full dashboard using built-in demo data — no file upload needed.",
        )
        if use_sample:
            st.info("Using built-in demo data. Upload your own LendingClub CSV to analyze real loans.")
            uploaded = None
        else:
            st.caption("Upload a LendingClub loan CSV to analyze your own data.")
            uploaded = st.file_uploader("Upload LendingClub CSV", type=["csv"])

    # Portfolio parameters
    st.markdown("<div class='sb-section'>Portfolio Settings</div>", unsafe_allow_html=True)
    investment = st.number_input(
        "Total Investment ($)",
        min_value=10_000, max_value=100_000_000,
        value=1_000_000, step=10_000, format="%d",
        help="The total dollar amount you want to invest across all selected loans.",
    )

    risk_tolerance = st.slider(
        "Risk Tolerance", 0, 100, 40,
        help=(
            "How much risk you're willing to accept. "
            "Low = only the safest loans. High = includes riskier, higher-interest loans."
        ),
    )
    risk_ceiling = 0.10 + (risk_tolerance / 100) * 0.40
    if risk_tolerance < 33:   r_cls, r_lbl = "risk-low",  "CONSERVATIVE"
    elif risk_tolerance < 66: r_cls, r_lbl = "risk-mid",  "MODERATE"
    else:                     r_cls, r_lbl = "risk-high", "AGGRESSIVE"
    st.markdown(
        f"<div style='margin-top:4px;'>"
        f"<span class='risk-pill {r_cls}'>● {r_lbl}</span>"
        f"<span style='font-size:0.64rem;color:{THEME['text_muted']};margin-left:8px;'>"
        f"max risk: {risk_ceiling:.0%}</span></div>",
        unsafe_allow_html=True,
    )

    # Strategy
    st.markdown("<div class='sb-section'>Investment Strategy</div>", unsafe_allow_html=True)
    strategy = st.selectbox(
        "Strategy",
        STRATS,
        index=3,
        label_visibility="collapsed",
        help="Choose how you want the model to select loans for your portfolio.",
    )

    # Dynamic strategy explanation
    STRAT_EXPLAIN = {
        "All Loans":                        ("MODERATE RISK",   THEME["amber"],  "Invests across all loans within your risk limit. Broad diversification, middle-of-the-road returns."),
        "Low-Risk Loans":                   ("LOWER RISK",      THEME["green"],  "Focuses only on loans with the lowest chance of not being repaid. Safer, but with lower projected profit."),
        "High-Interest Loans":              ("HIGHER RISK",     THEME["red_lt"], "Targets loans with higher interest rates for greater profit potential. More risk of losses."),
        "Optimized (Max Projected Profit)":  ("BEST BALANCE",    THEME["accent"], "Our model picks the loans most likely to deliver the best profit relative to risk. Recommended for most users."),
    }
    s_tag, s_color, s_desc = STRAT_EXPLAIN.get(strategy, ("", THEME["text_sec"], ""))
    st.markdown(
        f"<div style='background:rgba(0,0,0,0.25);border:1px solid {THEME['border']};"
        f"border-left:2px solid {s_color};border-radius:4px;padding:8px 12px;margin-top:6px;'>"
        f"<div style='font-size:0.58rem;font-weight:700;color:{s_color};"
        f"text-transform:uppercase;letter-spacing:0.09em;margin-bottom:4px;'>{s_tag}</div>"
        f"<div style='font-size:0.74rem;color:{THEME['text_sec']};line-height:1.6;'>{s_desc}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Model info
    st.markdown("<div class='sb-section'>About the Model</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:0.72rem;color:{THEME['text_muted']};line-height:1.9;'>"
        f"Trained on <span style='color:{THEME['text_sec']};'>{N_SAMPLE:,} historical loans</span><br>"
        f"Predicts which loans are likely to default<br>"
        f"Accuracy score shown in the dashboard below"
        f"</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_and_clean(source, n: int = N_SAMPLE) -> pd.DataFrame:
    """Load and sanitize LendingClub CSV from file path, upload, or sample generator."""
    if source == "SAMPLE":
        df = generate_sample_data(6_000)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        def_kw = ["charged off", "default", "late", "grace period"]
        df["default"] = df["loan_status"].str.lower().apply(
            lambda x: 1 if any(k in str(x) for k in def_kw) else 0
        )
        return df.reset_index(drop=True)
    try:
        if isinstance(source, str):
            fsize = os.path.getsize(source)
            if fsize > 200 * 1024 * 1024:
                df = pd.read_csv(source, nrows=n, low_memory=False,
                                 encoding="utf-8", encoding_errors="replace")
            else:
                df = pd.read_csv(source, low_memory=False,
                                 encoding="utf-8", encoding_errors="replace")
                if len(df) > n:
                    df = df.sample(n, random_state=42)
        else:
            df = pd.read_csv(source, low_memory=False)
            if len(df) > n:
                df = df.sample(n, random_state=42)
    except PermissionError:
        st.error(
            "**Permission Denied** — the file may be locked by OneDrive.\n\n"
            "Copy the CSV to a local folder outside OneDrive and rerun, "
            "or upload it directly via the sidebar."
        )
        return pd.DataFrame()
    except IsADirectoryError:
        st.error("**Path is a directory, not a file.** Ensure the CSV file is in the app folder.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"**Could not read file:** {e}")
        return pd.DataFrame()

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        st.error(f"**Missing required columns:** {missing}")
        return pd.DataFrame()

    # Normalise interest rate to decimal
    if df["int_rate"].dtype == object:
        df["int_rate"] = pd.to_numeric(
            df["int_rate"].astype(str).str.replace("%", "").str.strip(), errors="coerce"
        )
    if df["int_rate"].median() > 1:
        df["int_rate"] = df["int_rate"] / 100.0

    # Binary default label
    def_kw = ["charged off", "default", "late", "grace period"]
    df["default"] = df["loan_status"].str.lower().apply(
        lambda x: 1 if any(k in str(x) for k in def_kw) else 0
    )

    for col in ["loan_amnt", "annual_inc", "dti"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["loan_amnt", "int_rate", "annual_inc", "dti"])
    df = df[df["loan_amnt"] > 0]
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> "tuple[pd.DataFrame, list[str]]":
    """Build numeric feature matrix from raw columns."""
    feats = ["loan_amnt", "int_rate", "annual_inc", "dti"]

    if "revol_util" in df.columns:
        df["revol_util"] = pd.to_numeric(
            df["revol_util"].astype(str).str.replace("%", ""), errors="coerce"
        ).fillna(50)
        feats.append("revol_util")

    for col in ["open_acc", "total_acc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(10)
            feats.append(col)

    if "term" in df.columns:
        df["term_months"] = (
            df["term"].astype(str).str.extract(r"(\d+)").astype(float).fillna(36)
        )
        feats.append("term_months")

    if "grade" in df.columns:
        df["grade_num"] = (
            df["grade"].map({"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7}).fillna(4)
        )
        feats.append("grade_num")

    if "home_ownership" in df.columns:
        df["owns_home"] = (
            df["home_ownership"].str.upper().isin(["OWN", "MORTGAGE"]).astype(int)
        )
        feats.append("owns_home")

    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_avg"] = (
            pd.to_numeric(df["fico_range_low"],  errors="coerce").fillna(670) +
            pd.to_numeric(df["fico_range_high"], errors="coerce").fillna(680)
        ) / 2
        feats.append("fico_avg")

    if "delinq_2yrs" in df.columns:
        df["delinq_2yrs"] = pd.to_numeric(df["delinq_2yrs"], errors="coerce").fillna(0)
        feats.append("delinq_2yrs")

    if "inq_last_6mths" in df.columns:
        df["inq_last_6mths"] = pd.to_numeric(df["inq_last_6mths"], errors="coerce").fillna(1)
        feats.append("inq_last_6mths")

    df["inc_to_loan"] = (df["annual_inc"] / df["loan_amnt"]).clip(0, 100)
    feats.append("inc_to_loan")

    df[feats] = (
        df[feats]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(df[feats].apply(pd.to_numeric, errors="coerce").median())
    )
    return df, feats


@st.cache_resource(show_spinner=False)
def train_model(_data_hash: str, df_json: str):
    """Train XGBoost model and return model + diagnostics (cached by data hash)."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
    from xgboost import XGBClassifier

    df = pd.read_json(df_json)
    df, feats = engineer_features(df)
    X, y = df[feats], df["default"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.80,
        colsample_bytree=0.80,
        scale_pos_weight=max((y == 0).sum() / max((y == 1).sum(), 1), 1),
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_tr, y_tr)

    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc    = roc_auc_score(y_te, y_prob)
    cm     = confusion_matrix(y_te, y_pred).tolist()   # [[TN,FP],[FN,TP]]
    fpr, tpr, _ = roc_curve(y_te, y_prob)

    return (
        model, feats, auc,
        int(len(df)), float(y.mean()),
        cm, fpr.tolist(), tpr.tolist()
    )


def score_loans(model, feats: list, df: pd.DataFrame) -> pd.DataFrame:
    """Attach default probability and expected return metrics to each loan."""
    df, _ = engineer_features(df)
    df["default_prob"]        = model.predict_proba(df[feats].fillna(0))[:, 1]
    df["interest_income"]     = df["int_rate"] * df["loan_amnt"]
    df["expected_loss"]       = df["default_prob"] * df["loan_amnt"]
    df["expected_return"]     = df["interest_income"] - df["expected_loss"]
    df["expected_return_pct"] = df["expected_return"] / df["loan_amnt"]
    return df


def build_portfolio(
    df: pd.DataFrame, strat: str, risk_tol: int, capital: float
) -> "tuple[pd.DataFrame, dict]":
    """Select loans per strategy and compute allocation + summary metrics."""
    df   = df.copy()
    ceil = 0.10 + (risk_tol / 100) * 0.40

    if strat == "All Loans":
        sel = df[df["default_prob"] <= ceil]
    elif strat == "Low-Risk Loans":
        sel = df[df["default_prob"] <= 0.12]
    elif strat == "High-Interest Loans":
        sel = df[
            (df["int_rate"] >= df["int_rate"].quantile(0.75)) &
            (df["default_prob"] <= ceil)
        ]
    else:  # Optimized (Max Projected Profit)
        sel = (
            df[df["expected_return"] > 0]
            .sort_values("expected_return", ascending=False)
            .head(int(len(df) * 0.40))
        )

    if sel.empty:
        return sel, {}

    sel = sel.copy()
    w   = sel["expected_return"].clip(lower=0)
    wt  = w.sum()
    sel["allocation"] = w / wt * capital if wt > 0 else capital / len(sel)

    inv  = sel["allocation"].sum()
    gain = (sel["expected_return_pct"] * sel["allocation"]).sum()
    dr   = sel["default"].mean() if "default" in sel.columns else sel["default_prob"].mean()
    ap   = sel["default_prob"].mean()

    return sel, {
        "n_loans":            len(sel),
        "total_invested":     inv,
        "expected_return_$":  gain,
        "expected_roi_%":     gain / inv * 100 if inv else 0,
        "default_rate_%":     dr * 100,
        "avg_int_rate_%":     sel["int_rate"].mean() * 100,
        "avg_default_prob_%": ap * 100,
    }


@st.cache_data(show_spinner=False)
def compare_all_strategies(
    df_json: str, _data_hash: str, risk_tol: int, capital: float
) -> dict:
    """Compute metrics for all four strategies at current settings (cached)."""
    df = pd.read_json(df_json)
    return {
        s: build_portfolio(df, s, risk_tol, capital)[1]
        for s in STRATS
    }


def apply_filters(
    df: pd.DataFrame,
    grades: list, purposes: list, states: list
) -> pd.DataFrame:
    """Narrow loan universe to user-selected dimensions."""
    mask = pd.Series(True, index=df.index)
    if grades and "grade" in df.columns:
        mask &= df["grade"].isin(grades)
    if purposes and "purpose" in df.columns:
        mask &= df["purpose"].isin(purposes)
    if states and "addr_state" in df.columns:
        mask &= df["addr_state"].isin(states)
    return df[mask].copy()


def build_executive_summary(
    strat_metrics: dict, base_dr: float, auc: float, investment: float
) -> "tuple[str, dict]":
    """Generate business-language narrative for the executive summary panel."""
    if not strat_metrics:
        return "", {}

    best_name = max(strat_metrics, key=lambda s: strat_metrics[s].get("expected_roi_%", 0))
    best_m    = strat_metrics[best_name]
    all_m     = strat_metrics.get("All Loans", {})
    all_roi   = all_m.get("expected_roi_%", 0) if all_m else 0

    opt_dr          = best_m.get("default_rate_%", base_dr * 100)
    risk_reduction  = (base_dr * 100 - opt_dr) / (base_dr * 100) * 100 if base_dr > 0 else 0
    roi_excess      = best_m.get("expected_roi_%", 0) - all_roi
    model_quality   = "strong" if auc >= 0.75 else "good" if auc >= 0.70 else "moderate"
    short_name      = best_name.replace(" (Max Projected Profit)", "")

    narrative = (
        f"Our model analyzed <strong>{best_m.get('n_loans', 0):,} loans</strong> "
        f"(prediction accuracy: <strong>{auc:.4f}</strong> — {model_quality}) "
        f"and recommends the <strong>{short_name}</strong> strategy as the best fit for your settings. "
        f"Investing <strong>${investment:,.0f}</strong> is projected to generate "
        f"<strong>{best_m.get('expected_roi_%', 0):+.2f}% in returns</strong>, "
        f"with only <strong>{best_m.get('default_rate_%', 0):.1f}% of loans</strong> expected to not be repaid — "
        f"a <strong>{risk_reduction:.0f}% improvement</strong> over the historical average of {base_dr * 100:.1f}%."
    )

    if "Optimized" in best_name:
        rationale = "The model selects only loans where the projected profit outweighs the risk of loss — the best balance of safety and return."
    elif "Low-Risk" in best_name:
        rationale = "Only loans with a very low chance of default are included. Safer, but with lower profit potential."
    elif "High-Interest" in best_name:
        rationale = "Targets the highest-yielding loans while staying within your risk limit. Higher profit, but more loans may not be repaid."
    else:
        rationale = "Spreads your investment broadly across all qualifying loans within your risk setting. Maximum diversification."

    return narrative, {
        "optimal":    short_name,
        "roi_excess": f"{roi_excess:+.2f}% vs All Loans",
        "risk_cut":   f"−{risk_reduction:.0f}% vs avg",
        "rationale":  rationale,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DETERMINE DATA SOURCE
# ══════════════════════════════════════════════════════════════════════════════
if use_sample:
    source = "SAMPLE"
elif use_local and LOCAL_CSV:
    source = LOCAL_CSV
else:
    source = uploaded

# ── Navigation bar ────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="nav-bar">
    <div class="nav-brand">
        <span class="nav-title">Loan Investment Dashboard</span>
        <span class="nav-sub">Powered by Machine Learning · LendingClub Data</span>
    </div>
    <span class="nav-badge">AI-Powered · v3.0</span>
</div>
""", unsafe_allow_html=True)

# ── How This Works ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{THEME['surface2']};border:1px solid {THEME['border']};
            border-left:3px solid {THEME['accent']};border-radius:6px;
            padding:16px 22px;margin-bottom:20px;">
    <div style="font-size:0.62rem;font-weight:700;color:{THEME['accent']};
                text-transform:uppercase;letter-spacing:0.12em;margin-bottom:10px;">
        How This Works
    </div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;">
        <div style="font-size:0.78rem;color:{THEME['text_sec']};line-height:1.6;">
            <span style="color:{THEME['text_pri']};font-weight:600;display:block;margin-bottom:3px;">
                1. We analyze loans
            </span>
            Historical LendingClub data covering thousands of real loans is loaded and cleaned automatically.
        </div>
        <div style="font-size:0.78rem;color:{THEME['text_sec']};line-height:1.6;">
            <span style="color:{THEME['text_pri']};font-weight:600;display:block;margin-bottom:3px;">
                2. AI scores each loan
            </span>
            A machine learning model estimates the chance each loan will not be repaid, based on borrower history.
        </div>
        <div style="font-size:0.78rem;color:{THEME['text_sec']};line-height:1.6;">
            <span style="color:{THEME['text_pri']};font-weight:600;display:block;margin-bottom:3px;">
                3. We simulate strategies
            </span>
            We test four investment approaches and show you the projected profit and risk for each one.
        </div>
        <div style="font-size:0.78rem;color:{THEME['text_sec']};line-height:1.6;">
            <span style="color:{THEME['text_pri']};font-weight:600;display:block;margin-bottom:3px;">
                4. You choose your approach
            </span>
            Set your investment amount and risk comfort level in the sidebar, then explore the results below.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── No-data landing ───────────────────────────────────────────────────────────
if source is None:
    st.markdown(f"""
    <div style="background:{THEME['surface']};border:1px solid {THEME['border']};
                border-radius:6px;padding:40px 44px;text-align:center;margin-top:8px;
                box-shadow:0 2px 12px rgba(0,0,0,0.3);">
        <div style="font-size:1.0rem;font-weight:600;color:{THEME['text_pri']};margin-bottom:10px;">
            No Data Loaded
        </div>
        <div style="font-size:0.84rem;color:{THEME['text_sec']};max-width:480px;
                    margin:0 auto;line-height:1.8;">
            To explore the dashboard with demo data, enable
            <strong style="color:{THEME['accent_lt']};">Explore with Sample Data</strong>
            in the sidebar — no upload needed.<br><br>
            To analyze real loan data, upload a LendingClub CSV via the sidebar.
            Dataset available at
            <a href="https://www.kaggle.com/datasets/wordsforthewise/lending-club"
               target="_blank" style="color:{THEME['accent_lt']};font-weight:500;">
               Kaggle — LendingClub</a>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='sec-head'>Required Columns</div>", unsafe_allow_html=True)
        for c in REQUIRED:
            st.markdown(f"`{c}`")
    with c2:
        st.markdown("<div class='sec-head'>Recommended Columns</div>", unsafe_allow_html=True)
        for c in OPTIONAL:
            st.markdown(f"`{c}`")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD → TRAIN → SCORE
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading dataset…"):
    df_raw = load_and_clean(source, n=N_SAMPLE)
if df_raw.empty:
    st.stop()

with st.spinner("Training XGBoost model — first run only, cached thereafter…"):
    data_hash = hashlib.md5(
        pd.util.hash_pandas_object(df_raw, index=False).values.tobytes()
    ).hexdigest()
    model, feats, auc, n_train, base_dr, cm_data, roc_fpr, roc_tpr = train_model(
        data_hash, df_raw.to_json()
    )

df_scored = score_loans(model, feats, df_raw.copy())


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Part 2: interactive filters (populated after data loads)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<div class='sb-section'>Filter Loans</div>", unsafe_allow_html=True)
    st.caption("Narrow the loans used in your portfolio")

    grade_opts   = (sorted(df_scored["grade"].dropna().unique().tolist())
                    if "grade" in df_scored.columns else [])
    purpose_opts = (sorted(df_scored["purpose"].dropna().unique().tolist())
                    if "purpose" in df_scored.columns else [])
    state_opts   = (sorted(df_scored["addr_state"].dropna().unique().tolist())
                    if "addr_state" in df_scored.columns else [])

    grade_filter   = st.multiselect("Loan Grade",   grade_opts,   default=[],
                                    key="grade_filt",   placeholder="All grades")
    purpose_filter = st.multiselect("Loan Purpose",  purpose_opts, default=[],
                                    key="purpose_filt", placeholder="All purposes")
    state_filter   = st.multiselect("State",         state_opts,   default=[],
                                    key="state_filt",   placeholder="All states")


# ══════════════════════════════════════════════════════════════════════════════
#  APPLY FILTERS → BUILD PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════
df_filtered = apply_filters(df_scored, grade_filter, purpose_filter, state_filter)

if df_filtered.empty:
    st.warning("No loans match the current filter combination. Expand your filter selections.")
    st.stop()

sel_df, metrics = build_portfolio(df_filtered, strategy, risk_tolerance, investment)

if sel_df.empty:
    st.warning("No loans pass the current risk ceiling. Increase the Risk Tolerance slider.")
    st.stop()

# Strategy comparison (cached)
strat_metrics = compare_all_strategies(
    df_filtered.to_json(), data_hash, risk_tolerance, investment
)

# ── Status strip ──────────────────────────────────────────────────────────────
if source == "SAMPLE":
    src_name = "Demo Sample Data"
elif isinstance(source, str):
    src_name = os.path.basename(source)
else:
    src_name = getattr(source, "name", "uploaded file")

filters_active = bool(grade_filter or purpose_filter or state_filter)
filter_label = "FILTERED" if filters_active else "FULL PORTFOLIO"

st.markdown(f"""
<div class="status-strip">
    <span><span class="status-dot"></span>Live</span>
    <span class="status-item">Data: <span>{src_name[:45]}</span></span>
    <span class="status-item">Loans analyzed: <span>{n_train:,}</span></span>
    <span class="status-item">Loans in view: <span>{len(df_filtered):,}</span></span>
    <span class="status-item">Historical default rate: <span>{base_dr:.1%}</span></span>
    <span class="status-item">Strategy: <span>{strategy.replace(' (Max Projected Profit)','')}</span></span>
    <span class="status-item">Scope: <span>{filter_label}</span></span>
</div>
""", unsafe_allow_html=True)

# Sample data notice
if source == "SAMPLE":
    st.markdown(
        f"<div class='info-box'>"
        f"<strong style='color:{THEME['accent_lt']};'>Demo Mode:</strong> "
        f"You're viewing built-in sample data. Results are illustrative. "
        f"Upload a real LendingClub CSV from the sidebar to analyze actual loans."
        f"</div>",
        unsafe_allow_html=True,
    )

# Filter active notice
if filters_active:
    active_parts = []
    if grade_filter:   active_parts.append(f"Grade: {', '.join(grade_filter)}")
    if purpose_filter: active_parts.append(f"Purpose: {len(purpose_filter)} selected")
    if state_filter:   active_parts.append(f"State: {len(state_filter)} selected")
    st.markdown(
        f"<div class='filter-strip'>🔽 Active filters: "
        + " &nbsp;·&nbsp; ".join(f"<span class='filter-badge'>{p}</span>" for p in active_parts)
        + "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec-head'>Portfolio Overview</div>", unsafe_allow_html=True)

narrative, insights = build_executive_summary(strat_metrics, base_dr, auc, investment)

if narrative:
    st.markdown(f"""
    <div class="exec-card">
        <div class="exec-header">
            <span class="exec-tag">📋 AI-Generated Insight</span>
        </div>
        <div class="exec-body">{narrative}</div>
        <div class="exec-insight-grid">
            <div class="exec-insight-item">
                <div class="exec-insight-label">Best Strategy</div>
                <div class="exec-insight-value">{insights.get('optimal','—')}</div>
                <div class="exec-insight-sub">by projected return</div>
            </div>
            <div class="exec-insight-item">
                <div class="exec-insight-label">Extra Profit vs Baseline</div>
                <div class="exec-insight-value">{insights.get('roi_excess','—')}</div>
                <div class="exec-insight-sub">above investing in all loans</div>
            </div>
            <div class="exec-insight-item">
                <div class="exec-insight-label">Default Risk Reduction</div>
                <div class="exec-insight-value">{insights.get('risk_cut','—')}</div>
                <div class="exec-insight-sub">fewer defaults vs avg</div>
            </div>
        </div>
        <div style="margin-top:12px;padding:9px 12px;background:rgba(245,158,11,0.06);
                    border:1px solid rgba(245,158,11,0.14);border-radius:4px;">
            <span style="font-size:0.62rem;color:{THEME['gold']};font-weight:700;
                         text-transform:uppercase;letter-spacing:0.08em;">
                Why we recommend this:
            </span>
            <span style="font-size:0.77rem;color:{THEME['text_sec']};margin-left:8px;">
                {insights.get('rationale','—')}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════
roi_cls = "pos" if metrics["expected_roi_%"] >= 0 else "neg"
dr_cls  = "neg" if metrics["default_rate_%"] > 20 else ("pos" if metrics["default_rate_%"] < 10 else "")
auc_cls = "pos" if auc >= 0.70 else "neg"

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-cell">
    <div class="kpi-label">Capital Invested</div>
    <div class="kpi-value">${metrics['total_invested']:,.0f}</div>
    <div class="kpi-delta">{metrics['n_loans']:,} loans selected</div>
  </div>
  <div class="kpi-cell">
    <div class="kpi-label">Projected Profit</div>
    <div class="kpi-value">${metrics['expected_return_$']:,.0f}</div>
    <div class="kpi-delta {roi_cls}">{'+' if metrics['expected_roi_%'] >= 0 else ''}{metrics['expected_roi_%']:.2f}% projected return</div>
  </div>
  <div class="kpi-cell">
    <div class="kpi-label">Loans at Risk of Default</div>
    <div class="kpi-value">{metrics['default_rate_%']:.1f}%</div>
    <div class="kpi-delta {dr_cls}">avg risk score: {metrics['avg_default_prob_%']:.1f}%</div>
  </div>
  <div class="kpi-cell">
    <div class="kpi-label">Avg Interest Rate</div>
    <div class="kpi-value">{metrics['avg_int_rate_%']:.2f}%</div>
    <div class="kpi-delta">weighted across portfolio</div>
  </div>
  <div class="kpi-cell">
    <div class="kpi-label">Prediction Accuracy</div>
    <div class="kpi-value">{auc:.4f}</div>
    <div class="kpi-delta {auc_cls}">{'Strong' if auc >= 0.75 else 'Good' if auc >= 0.70 else 'Moderate'} model accuracy</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  EXAMPLE SCENARIO CARD
# ══════════════════════════════════════════════════════════════════════════════
_ex_inv     = investment
_ex_profit  = metrics.get("expected_return_$", 0)
_ex_roi     = metrics.get("expected_roi_%", 0)
_ex_default = metrics.get("default_rate_%", 0)
_ex_loans   = metrics.get("n_loans", 0)
_ex_loss_est = _ex_inv * (_ex_default / 100)
_ex_net     = _ex_profit - _ex_loss_est
_strat_short = strategy.replace(" (Max Projected Profit)", "")

st.markdown(f"""
<div style="background:linear-gradient(135deg,{THEME['surface2']} 0%,{THEME['surface3']} 100%);
            border:1px solid {THEME['border2']};border-left:3px solid {THEME['gold']};
            border-radius:6px;padding:18px 24px;margin-bottom:22px;">
    <div style="font-size:0.60rem;font-weight:700;color:{THEME['gold']};
                text-transform:uppercase;letter-spacing:0.12em;margin-bottom:10px;">
        Example Scenario — What Happens With Your Investment
    </div>
    <div style="font-size:0.84rem;color:{THEME['text_sec']};margin-bottom:14px;line-height:1.7;">
        If you invest <strong style="color:{THEME['text_pri']};">${_ex_inv:,.0f}</strong>
        using the <strong style="color:{THEME['accent_lt']};">{_strat_short}</strong> strategy,
        your money is spread across <strong style="color:{THEME['text_pri']};">{_ex_loans:,} loans</strong>.
        Based on historical data and our model, here is what we project:
    </div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">
        <div style="background:rgba(0,0,0,0.3);border:1px solid {THEME['border']};
                    border-radius:4px;padding:12px 14px;text-align:center;">
            <div style="font-size:0.60rem;color:{THEME['text_muted']};text-transform:uppercase;
                        letter-spacing:0.08em;margin-bottom:6px;">Money Invested</div>
            <div style="font-size:1.1rem;font-weight:700;color:{THEME['text_pri']};
                        font-family:'JetBrains Mono',monospace;">${_ex_inv:,.0f}</div>
            <div style="font-size:0.68rem;color:{THEME['text_muted']};margin-top:3px;">your starting capital</div>
        </div>
        <div style="background:rgba(0,0,0,0.3);border:1px solid {THEME['border']};
                    border-radius:4px;padding:12px 14px;text-align:center;">
            <div style="font-size:0.60rem;color:{THEME['text_muted']};text-transform:uppercase;
                        letter-spacing:0.08em;margin-bottom:6px;">Projected Profit</div>
            <div style="font-size:1.1rem;font-weight:700;
                        color:{THEME['green_lt'] if _ex_profit >= 0 else THEME['red_lt']};
                        font-family:'JetBrains Mono',monospace;">
                {'+' if _ex_profit >= 0 else ''}${_ex_profit:,.0f}
            </div>
            <div style="font-size:0.68rem;color:{THEME['text_muted']};margin-top:3px;">
                {'+' if _ex_roi >= 0 else ''}{_ex_roi:.2f}% return
            </div>
        </div>
        <div style="background:rgba(0,0,0,0.3);border:1px solid {THEME['border']};
                    border-radius:4px;padding:12px 14px;text-align:center;">
            <div style="font-size:0.60rem;color:{THEME['text_muted']};text-transform:uppercase;
                        letter-spacing:0.08em;margin-bottom:6px;">Estimated Losses</div>
            <div style="font-size:1.1rem;font-weight:700;color:{THEME['amber']};
                        font-family:'JetBrains Mono',monospace;">${_ex_loss_est:,.0f}</div>
            <div style="font-size:0.68rem;color:{THEME['text_muted']};margin-top:3px;">
                from loans not repaid ({_ex_default:.1f}%)
            </div>
        </div>
        <div style="background:rgba(0,0,0,0.3);border:1px solid {THEME['border']};
                    border-radius:4px;padding:12px 14px;text-align:center;">
            <div style="font-size:0.60rem;color:{THEME['text_muted']};text-transform:uppercase;
                        letter-spacing:0.08em;margin-bottom:6px;">Loans in Portfolio</div>
            <div style="font-size:1.1rem;font-weight:700;color:{THEME['text_pri']};
                        font-family:'JetBrains Mono',monospace;">{_ex_loans:,}</div>
            <div style="font-size:0.68rem;color:{THEME['text_muted']};margin-top:3px;">
                avg ${_ex_inv/_ex_loans:,.0f} per loan
            </div>
        </div>
    </div>
    <div style="margin-top:10px;font-size:0.70rem;color:{THEME['text_muted']};line-height:1.5;">
        These are projections based on historical patterns, not guarantees.
        Actual results will vary based on economic conditions and borrower behavior.
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — STRATEGY COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec-head'>Strategy Comparison</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='info-box'>All four strategies are evaluated on the same set of loans at your "
    "current risk and investment settings. Your active strategy is highlighted.</div>",
    unsafe_allow_html=True,
)

rows = []
for s, m in strat_metrics.items():
    if m:
        rows.append({
            "Strategy":          s,
            "Loans Selected":    f"{m['n_loans']:,}",
            "Expected ROI":      f"{m['expected_roi_%']:+.2f}%",
            "Expected Return":   f"${m['expected_return_$']:,.0f}",
            "Default Rate":      f"{m['default_rate_%']:.1f}%",
            "Avg Interest Rate": f"{m['avg_int_rate_%']:.2f}%",
            "Avg Default Prob":  f"{m['avg_default_prob_%']:.1f}%",
        })

if rows:
    comp = pd.DataFrame(rows)

    def _hl_row(row):
        active = row["Strategy"] == strategy
        c = (f"background-color:rgba(59,130,246,0.14);"
             f"color:{THEME['accent_lt']};font-weight:600") if active else ""
        return [c] * len(row)

    st.dataframe(
        comp.style.apply(_hl_row, axis=1),
        use_container_width=True, hide_index=True
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — RISK DISTRIBUTION & LOAN-LEVEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<div class='sec-head'>Risk Insights</div>",
    unsafe_allow_html=True
)
ch1, ch2 = st.columns(2)

with ch1:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df_filtered["default_prob"], nbinsx=60, name="Filtered Universe",
        marker_color=THEME["border2"], opacity=0.5, marker_line=dict(width=0),
    ))
    fig.add_trace(go.Histogram(
        x=sel_df["default_prob"], nbinsx=60,
        name=strategy.split("(")[0].strip(),
        marker_color=THEME["accent"], opacity=0.82, marker_line=dict(width=0),
    ))
    fig.update_layout(
        title="Chance of Not Being Repaid — Distribution",
        xaxis_title="Chance of Loan Defaulting (0 = safe, 1 = high risk)",
        yaxis_title="Number of Loans",
        barmode="overlay",
        **PLOT_LAYOUT, **LEGEND_H,
    )
    st.plotly_chart(fig, use_container_width=True)

with ch2:
    samp = sel_df.sample(min(5_000, len(sel_df)), random_state=1)
    color_col = "grade_num" if "grade_num" in samp.columns else "int_rate"
    fig2 = px.scatter(
        samp, x="default_prob", y="expected_return_pct",
        color=color_col, size="loan_amnt", size_max=9,
        color_continuous_scale=[[0, "#dbeafe"], [0.5, "#2563eb"], [1, "#1e3a8a"]],
        labels={
            "default_prob":        "Chance of Default",
            "expected_return_pct": "Projected Profit (%)",
            color_col: "Grade" if color_col == "grade_num" else "Int Rate",
        },
        title="Risk vs. Projected Profit",
    )
    fig2.update_layout(**PLOT_LAYOUT, **LEGEND_V)
    fig2.update_coloraxes(colorbar=dict(
        tickfont=dict(size=9, color=THEME["chart_font"]),
        title_font=dict(size=10, color=THEME["chart_font"]),
    ))
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — STRATEGY PERFORMANCE & GRADE BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<div class='sec-head'>Strategy Performance</div>",
    unsafe_allow_html=True
)
ch3, ch4 = st.columns(2)

with ch3:
    sn, rv, dv = [], [], []
    for s, m in strat_metrics.items():
        if m:
            sn.append(s.replace(" (Max Projected Profit)", ""))
            rv.append(round(m["expected_roi_%"], 2))
            dv.append(round(m["default_rate_%"], 2))

    active_short = strategy.replace(" (Max Projected Profit)", "")
    bar_colors  = [THEME["accent"] if s == active_short else THEME["border2"] for s in sn]
    bar_borders = [THEME["accent"] if s == active_short else THEME["border"]  for s in sn]

    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Bar(
        x=sn, y=rv, name="Expected ROI (%)",
        marker=dict(color=bar_colors, line=dict(color=bar_borders, width=1)),
        opacity=0.9,
    ), secondary_y=False)
    fig3.add_trace(go.Scatter(
        x=sn, y=dv, name="Default Rate (%)",
        mode="lines+markers",
        line=dict(color=THEME["red_lt"], width=2),
        marker=dict(size=7, color=THEME["red_lt"],
                    line=dict(color=THEME["chart_bg"], width=2)),
    ), secondary_y=True)
    fig3.update_yaxes(title_text="Projected Return (%)", secondary_y=False,
                      gridcolor=THEME["chart_grid"], tickfont=dict(size=10))
    fig3.update_yaxes(title_text="Default Rate (%)", secondary_y=True,
                      gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10))
    fig3.update_layout(title="Projected Return vs. Default Rate by Strategy",
                       **PLOT_LAYOUT, **LEGEND_H)
    st.plotly_chart(fig3, use_container_width=True)

with ch4:
    if "grade" in sel_df.columns:
        grp = (
            sel_df.groupby("grade", observed=True)
            .agg(avg_roi=("expected_return_pct", "mean"),
                 avg_def=("default_prob", "mean"),
                 count=("loan_amnt", "count"))
            .reset_index().sort_values("grade")
        )
        grp["avg_roi_pct"] = (grp["avg_roi"] * 100).round(2)
        fig4 = go.Figure(go.Bar(
            x=grp["grade"], y=grp["avg_roi_pct"],
            text=grp["count"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            textfont=dict(size=10, color=THEME["text_sec"]),
            marker=dict(
                color=grp["avg_def"],
                colorscale=[[0, "#16a34a"], [0.5, "#d97706"], [1, "#b91c1c"]],
                showscale=True,
                colorbar=dict(title="Default Prob",
                              tickfont=dict(size=9, color=THEME["chart_font"])),
            ),
        ))
        fig4.update_layout(title="Projected Profit by Loan Grade",
                           xaxis_title="Grade (A = safest, G = riskiest)", yaxis_title="Avg Profit (%)",
                           **PLOT_LAYOUT, **LEGEND_V)
    else:
        sel_df["rate_bucket"] = pd.cut(
            sel_df["int_rate"] * 100,
            bins=[0, 8, 12, 16, 20, 100],
            labels=["<8%", "8–12%", "12–16%", "16–20%", ">20%"]
        )
        grp = (
            sel_df.groupby("rate_bucket", observed=True)
            .agg(avg_roi=("expected_return_pct", "mean"), count=("loan_amnt", "count"))
            .reset_index()
        )
        grp["avg_roi_pct"] = (grp["avg_roi"] * 100).round(2)
        fig4 = go.Figure(go.Bar(
            x=grp["rate_bucket"].astype(str), y=grp["avg_roi_pct"],
            text=grp["count"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            marker_color=THEME["accent"],
        ))
        fig4.update_layout(title="Projected Profit by Interest Rate Range",
                           xaxis_title="Interest Rate Range", yaxis_title="Avg Profit (%)",
                           **PLOT_LAYOUT, **LEGEND_V)
    st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — FEATURE IMPORTANCE + ANALYST NOTES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec-head'>What Drives Loan Risk</div>",
            unsafe_allow_html=True)

fi_raw = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
fi_pct = fi_raw / fi_raw.sum() * 100  # percentage of total importance
fi_labeled = fi_pct.copy()
fi_labeled.index = [FEATURE_LABELS.get(f, f) for f in fi_pct.index]
fi_labeled_sorted = fi_labeled.sort_values()

col_fi, col_desc = st.columns([3, 2])

with col_fi:
    n_top = min(12, len(fi_labeled_sorted))
    fi_plot = fi_labeled_sorted.tail(n_top)

    # Color: top 3 = gold, top 6 = accent, rest = muted
    n = len(fi_plot)
    bar_colors = []
    for i in range(n):
        if i >= n - 3:   bar_colors.append(THEME["gold"])
        elif i >= n - 6: bar_colors.append(THEME["accent"])
        else:            bar_colors.append(THEME["border2"])

    fig5 = go.Figure(go.Bar(
        x=fi_plot.values,
        y=fi_plot.index,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color="rgba(0,0,0,0)", width=0)),
        text=[f"{v:.1f}%" for v in fi_plot.values],
        textposition="outside",
        textfont=dict(size=9, color=THEME["text_sec"]),
        hovertemplate="%{y}<br>Importance: %{x:.2f}%<extra></extra>",
    ))
    fig5.update_layout(
        title="Top Factors That Predict Loan Default",
        xaxis_title="Influence on Prediction (%)",
        height=420,
        **PLOT_LAYOUT, **LEGEND_V,
    )
    st.plotly_chart(fig5, use_container_width=True)

with col_desc:
    st.markdown(
        f"<div style='font-size:0.62rem;font-weight:700;color:{THEME['accent']};"
        f"text-transform:uppercase;letter-spacing:0.12em;padding-bottom:8px;"
        f"border-bottom:1px solid {THEME['border']};margin-bottom:10px;'>"
        f"Top Risk Factors — Plain English</div>",
        unsafe_allow_html=True,
    )
    # Show top 6 features with descriptions
    top_feats = fi_raw.sort_values(ascending=False).head(6)
    for rank, (feat_raw, importance) in enumerate(top_feats.items(), 1):
        label = FEATURE_LABELS.get(feat_raw, feat_raw)
        desc  = FEATURE_DESC.get(feat_raw, "Contributes to default prediction.")
        pct   = fi_pct[feat_raw]
        rank_color = THEME["gold"] if rank <= 3 else THEME["accent"]
        st.markdown(f"""
        <div class="feat-card">
            <div class="feat-card-name">
                {label}
                <span class="feat-card-rank" style="color:{rank_color};">
                    #{rank} · {pct:.1f}%
                </span>
            </div>
            <div class="feat-card-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — MODEL DIAGNOSTICS: ROC CURVE + CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec-head'>How Accurate Is Our Model</div>", unsafe_allow_html=True)

ch_roc, ch_cm = st.columns(2)

with ch_roc:
    # ROC Curve
    fpr_arr = np.array(roc_fpr)
    tpr_arr = np.array(roc_tpr)

    fig_roc = go.Figure()
    # Shaded AUC area
    fig_roc.add_trace(go.Scatter(
        x=fpr_arr, y=tpr_arr,
        mode="lines",
        line=dict(color=THEME["accent"], width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba(59,130,246,0.12)",
        name=f"ROC (AUC = {auc:.4f})",
        hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
    ))
    # Random baseline
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color=THEME["border2"], width=1.5, dash="dot"),
        name="Random Baseline",
        hoverinfo="skip",
    ))
    # Skill labels
    fig_roc.add_annotation(
        x=0.6, y=0.4,
        text=f"AUC = {auc:.4f}<br>({'Strong' if auc >= 0.75 else 'Good' if auc >= 0.70 else 'Moderate'})",
        showarrow=False,
        font=dict(size=12, color=THEME["gold"], family="JetBrains Mono"),
        bgcolor=THEME["surface2"],
        bordercolor=THEME["border2"],
        borderwidth=1,
        borderpad=6,
    )
    fig_roc.update_layout(
        title="Model Accuracy Curve — How Well We Detect Defaults",
        xaxis_title="False Alarm Rate (loans flagged but not actually risky)",
        yaxis_title="Detection Rate (risky loans correctly caught)",
        height=380,
        **PLOT_LAYOUT, **LEGEND_H,
    )
    st.plotly_chart(fig_roc, use_container_width=True)

with ch_cm:
    # Confusion Matrix heatmap
    tn, fp = cm_data[0]
    fn, tp = cm_data[1]
    total  = tn + fp + fn + tp

    cm_labels  = [["TN", "FP"], ["FN", "TP"]]
    cm_values  = [[tn, fp], [fn, tp]]
    cm_pcts    = [[f"{tn/total:.1%}", f"{fp/total:.1%}"],
                  [f"{fn/total:.1%}", f"{tp/total:.1%}"]]
    cm_text    = [
        [f"<b>True Negative</b><br>{tn:,}<br>{tn/total:.1%}", f"<b>False Positive</b><br>{fp:,}<br>{fp/total:.1%}"],
        [f"<b>False Negative</b><br>{fn:,}<br>{fn/total:.1%}", f"<b>True Positive</b><br>{tp:,}<br>{tp/total:.1%}"],
    ]
    cm_color   = [[tn, fp], [fn, tp]]

    fig_cm = go.Figure(go.Heatmap(
        z=cm_color,
        x=["Predicted: Paid", "Predicted: Default"],
        y=["Actual: Paid", "Actual: Default"],
        text=cm_text,
        texttemplate="%{text}",
        textfont=dict(size=11, family="JetBrains Mono"),
        colorscale=[[0, THEME["surface3"]], [1, THEME["accent"]]],
        showscale=False,
        hoverinfo="text",
    ))
    fig_cm.update_layout(
        title="Prediction Results — Did the Model Get It Right?",
        height=380,
        **{k: v for k, v in PLOT_LAYOUT.items() if k != "xaxis"},
        xaxis=dict(**PLOT_LAYOUT["xaxis"], side="bottom"),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # Derived metrics below the confusion matrix
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy  = (tp + tn) / total if total > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    for col, label, val, tip in [
        (m1, "Overall Accuracy",   f"{accuracy:.3f}",  "How often the model correctly predicted whether a loan would default"),
        (m2, "Precision",          f"{precision:.3f}", "Of loans flagged as risky, how many actually defaulted"),
        (m3, "Default Detection",  f"{recall:.3f}",    "Of loans that actually defaulted, how many the model caught"),
        (m4, "Balanced Score",     f"{f1:.3f}",        "Combines precision and detection rate into a single score"),
    ]:
        col.markdown(f"""
        <div style="background:{THEME['surface2']};border:1px solid {THEME['border']};
                    border-radius:4px;padding:10px 12px;text-align:center;">
            <div style="font-size:0.58rem;color:{THEME['text_muted']};text-transform:uppercase;
                        letter-spacing:0.09em;margin-bottom:4px;">{label}</div>
            <div style="font-size:1.1rem;font-weight:700;color:{THEME['text_pri']};
                        font-family:'JetBrains Mono',monospace;">{val}</div>
            <div style="font-size:0.62rem;color:{THEME['text_muted']};margin-top:2px;">{tip}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — PORTFOLIO RISK EXPOSURE BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec-head'>How Your Money Is Spread</div>",
            unsafe_allow_html=True)

# Risk tier classification
sel_copy = sel_df.copy()
conditions = [
    sel_copy["default_prob"] < 0.10,
    sel_copy["default_prob"] < 0.20,
]
choices    = ["Low  (<10%)", "Medium  (10–20%)"]
sel_copy["risk_tier"] = np.select(conditions, choices, default="High  (>20%)")

tier_order  = ["Low  (<10%)", "Medium  (10–20%)", "High  (>20%)"]
tier_colors = [THEME["green"], THEME["amber"], THEME["red_lt"]]

ch_exp1, ch_exp2, ch_exp3 = st.columns(3)

# ── Donut: loan count by risk tier
with ch_exp1:
    tier_cnt = sel_copy["risk_tier"].value_counts().reindex(tier_order, fill_value=0)
    fig_donut = go.Figure(go.Pie(
        labels=tier_cnt.index,
        values=tier_cnt.values,
        hole=0.58,
        marker=dict(colors=tier_colors, line=dict(color=THEME["bg"], width=2)),
        textinfo="percent",
        textfont=dict(size=11, family="JetBrains Mono"),
        hovertemplate="%{label}<br>Loans: %{value:,}<br>Share: %{percent}<extra></extra>",
    ))
    fig_donut.add_annotation(
        text=f"<b>{len(sel_copy):,}</b><br><span style='font-size:9px'>Loans</span>",
        x=0.5, y=0.5, font=dict(size=14, color=THEME["text_pri"], family="JetBrains Mono"),
        showarrow=False
    )
    fig_donut.update_layout(
        title="Loans by Risk Tier",
        height=320, margin=dict(t=44, b=10, l=10, r=10),
        **{k: v for k, v in PLOT_LAYOUT.items() if k not in ("margin",)},
        **LEGEND_V,
    )
    st.plotly_chart(fig_donut, use_container_width=True)

# ── Bar: capital allocation by risk tier
with ch_exp2:
    tier_alloc = (
        sel_copy.groupby("risk_tier", observed=True)["allocation"]
        .sum().reindex(tier_order, fill_value=0)
    )
    fig_alloc = go.Figure(go.Bar(
        x=tier_alloc.index,
        y=tier_alloc.values,
        marker=dict(color=tier_colors, line=dict(color=THEME["bg"], width=1)),
        text=[f"${v:,.0f}" for v in tier_alloc.values],
        textposition="outside",
        textfont=dict(size=9, color=THEME["text_sec"]),
        hovertemplate="%{x}<br>Allocated: $%{y:,.0f}<extra></extra>",
    ))
    fig_alloc.update_layout(
        title="Capital Allocated by Risk Tier",
        yaxis_title="Allocation ($)",
        height=320,
        **PLOT_LAYOUT, **LEGEND_V,
    )
    st.plotly_chart(fig_alloc, use_container_width=True)

# ── Grouped bar: expected gain vs expected loss by grade (or tier)
with ch_exp3:
    if "grade" in sel_copy.columns:
        grp_exp = (
            sel_copy.groupby("grade", observed=True)
            .agg(gain=("interest_income", "sum"),
                 loss=("expected_loss", "sum"))
            .reset_index().sort_values("grade")
        )
        x_axis = grp_exp["grade"]
    else:
        grp_exp = (
            sel_copy.groupby("risk_tier", observed=True)
            .agg(gain=("interest_income", "sum"),
                 loss=("expected_loss", "sum"))
            .reset_index()
        )
        x_axis = grp_exp["risk_tier"]

    fig_exp = go.Figure()
    fig_exp.add_trace(go.Bar(
        name="Expected Interest Income",
        x=x_axis, y=grp_exp["gain"],
        marker_color=THEME["green"],
        opacity=0.85,
        hovertemplate="%{x}<br>Income: $%{y:,.0f}<extra></extra>",
    ))
    fig_exp.add_trace(go.Bar(
        name="Expected Loss (Charge-off)",
        x=x_axis, y=grp_exp["loss"],
        marker_color=THEME["red_lt"],
        opacity=0.85,
        hovertemplate="%{x}<br>Loss: $%{y:,.0f}<extra></extra>",
    ))
    fig_exp.update_layout(
        title="Income vs. Expected Loss",
        yaxis_title="Amount ($)",
        barmode="group",
        height=320,
        **PLOT_LAYOUT, **LEGEND_H,
    )
    st.plotly_chart(fig_exp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — CUMULATIVE RETURN CURVE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec-head'>Projected Returns Across Your Portfolio</div>", unsafe_allow_html=True)

ranked        = sel_df.sort_values("expected_return", ascending=False).reset_index(drop=True)
ranked["cum"] = ranked["expected_return"].cumsum()
ranked["n"]   = range(1, len(ranked) + 1)

# Find 80/20 cutoff
cutoff_80 = int(len(ranked) * 0.20)
y_at_80   = ranked.iloc[cutoff_80]["cum"] if cutoff_80 < len(ranked) else 0

fig6 = go.Figure()
fig6.add_trace(go.Scatter(
    x=ranked["n"], y=ranked["cum"],
    mode="lines",
    line=dict(color=THEME["accent_lt"], width=2.5),
    fill="tozeroy",
    fillcolor="rgba(59,130,246,0.07)",
    name="Cumulative Return",
    hovertemplate="Loan #%{x:,}<br>Cum. Return: $%{y:,.0f}<extra></extra>",
))
# 80/20 reference line
fig6.add_vline(x=cutoff_80, line=dict(color=THEME["gold"], width=1.5, dash="dash"),
               annotation_text="Top 20% loans", annotation_font_color=THEME["gold"],
               annotation_font_size=10)
fig6.add_hline(y=0, line=dict(color=THEME["border2"], width=1, dash="dot"))
fig6.update_layout(
    title="Cumulative Projected Profit (Best Loans First)",
    xaxis_title="Loans included, ranked from highest to lowest projected profit",
    yaxis_title="Total Projected Profit ($)",
    **PLOT_LAYOUT, **LEGEND_V,
)
st.plotly_chart(fig6, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — LOAN PURPOSE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if "purpose" in sel_df.columns:
    st.markdown("<div class='sec-head'>Profit by Loan Purpose</div>", unsafe_allow_html=True)
    purp = (
        sel_df.groupby("purpose", observed=True)
        .agg(avg_ret=("expected_return_pct", "mean"),
             count=("loan_amnt", "count"),
             avg_def=("default_prob", "mean"))
        .reset_index()
        .sort_values("avg_ret", ascending=True)
        .tail(15)
    )
    purp["avg_ret_pct"] = (purp["avg_ret"] * 100).round(2)
    fig7 = go.Figure(go.Bar(
        x=purp["avg_ret_pct"], y=purp["purpose"], orientation="h",
        text=purp["count"].apply(lambda x: f"{x:,}"),
        textposition="outside",
        textfont=dict(size=10, color=THEME["text_sec"]),
        marker=dict(
            color=purp["avg_def"],
            colorscale=[[0, "#16a34a"], [0.5, "#d97706"], [1, "#b91c1c"]],
            showscale=True,
            colorbar=dict(title="Default Prob",
                          tickfont=dict(size=9, color=THEME["chart_font"])),
        ),
    ))
    fig7.update_layout(
        title="Average Expected Return by Loan Purpose",
        xaxis_title="Avg Return (%)",
        height=420, **PLOT_LAYOUT, **LEGEND_V,
    )
    st.plotly_chart(fig7, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — TOP LOAN OPPORTUNITIES TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec-head'>Top Loan Picks</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='info-box'>The top 300 loans ranked by projected profit, with your invested "
    "amount allocated across them. Higher-risk loans are color-coded accordingly.</div>",
    unsafe_allow_html=True,
)

BASE  = ["loan_amnt", "int_rate", "dti", "annual_inc",
         "default_prob", "expected_return", "expected_return_pct", "allocation"]
EXTRA = [c for c in ["grade", "term", "purpose", "addr_state", "loan_status"]
         if c in sel_df.columns]

top = (
    sel_df[EXTRA + BASE]
    .sort_values("expected_return", ascending=False)
    .head(300)
    .reset_index(drop=True)
)

fmt = top.copy()
if "int_rate"            in fmt: fmt["int_rate"]            = (fmt["int_rate"] * 100).round(2).astype(str) + "%"
if "default_prob"        in fmt: fmt["default_prob"]        = (fmt["default_prob"] * 100).round(1).astype(str) + "%"
if "expected_return_pct" in fmt: fmt["expected_return_pct"] = (fmt["expected_return_pct"] * 100).round(2).astype(str) + "%"
for c in ["loan_amnt", "annual_inc"]:
    if c in fmt: fmt[c] = fmt[c].apply(lambda x: f"${x:,.0f}")
for c in ["expected_return", "allocation"]:
    if c in fmt: fmt[c] = fmt[c].apply(lambda x: f"${x:,.2f}")
fmt.columns = [c.replace("_", " ").title() for c in fmt.columns]

st.dataframe(fmt, use_container_width=True, hide_index=True, height=420)

# Download button
csv_bytes = (
    sel_df[EXTRA + BASE]
    .sort_values("expected_return", ascending=False)
    .to_csv(index=False)
    .encode("utf-8")
)
st.download_button(
    "⬇  Export Portfolio CSV",
    data=csv_bytes,
    file_name="lc_portfolio_selection.csv",
    mime="text/csv",
)


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="margin-top:52px;padding:14px 0;border-top:1px solid {THEME['border']};
            display:flex;justify-content:space-between;align-items:center;">
    <span style="font-size:0.70rem;color:{THEME['text_muted']};">
        Built using machine learning on historical LendingClub loan data.
    </span>
    <span style="font-size:0.68rem;color:{THEME['text_muted']};">
        For illustrative purposes only. Not financial advice.
    </span>
</div>
""", unsafe_allow_html=True)
