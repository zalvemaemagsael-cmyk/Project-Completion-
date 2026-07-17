"""
ASENXO — MSME Project Completion Dashboard (Model-Driven)
============================================================
Uses the trained Logistic Regression pipeline from MSME_CompletionModel.pkl
to estimate completion probability for MSME projects.
Dummy data is used for prototyping – replace with real database queries.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import plotly.express as px


# ── Page config ──────────────────────────────────────────────────
# NOTE: st.set_page_config() MUST be the very first Streamlit command
# executed in the script. Previously it was called *after* the hero
# st.markdown() below, which throws a StreamlitAPIException
# ("set_page_config() can only be called once per app, and must be
# called as the first Streamlit command") the moment Streamlit's page
# config isn't the default — this was a real deployment-breaking bug.
st.set_page_config(
    page_title="ASENXO | Completion Dashboard",
    page_icon="📊",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<div class="page-hero">
    <h1>📊 MSME Project Completion Dashboard</h1>
    <p>DOST SETUP 4.0 iFund Program — Western Visayas | Model-driven risk assessment (dummy data)</p>
</div>
""", unsafe_allow_html=True)


RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ── Global CSS / Theme ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=JetBrains+Mono:wght@300;400;500&family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #dde3ee;
    font-size: 15px;
    line-height: 1.65;
    -webkit-font-smoothing: antialiased;
}

/* ── App shell ── */
.stApp {
    background-color: #060b14;
    background-image:
        radial-gradient(ellipse 110% 55% at 5%  0%,   rgba(14,165,233,0.13)  0%, transparent 55%),
        radial-gradient(ellipse  70% 45% at 95% 5%,   rgba(99, 102,241,0.11) 0%, transparent 50%),
        radial-gradient(ellipse  60% 70% at 50% 100%, rgba(20,184,166,0.09)  0%, transparent 55%),
        radial-gradient(ellipse  40% 35% at 80% 55%,  rgba(244,63,94,0.06)   0%, transparent 50%);
    min-height: 100vh;
}
.main .block-container { padding: 0 2.4rem 3.5rem; max-width: 98%; }

.sidebar-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    margin: 0.6rem 0.8rem 0.9rem;
    background: linear-gradient(135deg, rgba(20,184,166,0.18), rgba(14,165,233,0.12));
    border: 1px solid rgba(20,184,166,0.28);
    border-radius: 6px;
    padding: 3px 9px;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #2dd4bf;
}
.info-badge, .stat-badge {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 9px;
    padding: 0.5rem 0.85rem;
    font-size: 0.75rem;
    color: #7a90a8;
    line-height: 1.55;
    margin: 0 0 5px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    display: inline-block;
}
.info-badge strong, .stat-badge span { color: #b8cce0; font-weight: 600; }

/* ── Page hero ── */
.page-hero {
    background: linear-gradient(135deg, rgba(14,165,233,0.09) 0%, rgba(99,102,241,0.08) 50%, rgba(20,184,166,0.06) 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 1.75rem 2rem;
    margin: 1.2rem 0 1.6rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.page-hero::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(14,165,233,0.4), rgba(99,102,241,0.4), transparent);
}
.page-hero::after {
    content: '';
    position: absolute; top: -60px; right: -60px; width: 260px; height: 260px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(14,165,233,0.10) 0%, transparent 65%);
    pointer-events: none;
}
.page-hero h1 {
    font-family: 'Fraunces', serif !important;
    font-size: 1.65rem !important;
    font-weight: 600 !important;
    color: #f0f6ff !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 0.3rem !important;
}
.page-hero p { color: #7a90a8; font-size: 0.875rem; margin: 0; line-height: 1.55; }

.section-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(14,165,233,0.09);
    border: 1px solid rgba(14,165,233,0.20);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; color: #38bdf8;
    margin-bottom: 0.9rem;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(255,255,255,0.052) 0%, rgba(255,255,255,0.022) 100%) !important;
    border: 1px solid rgba(255,255,255,0.075) !important;
    border-radius: 16px !important;
    padding: 1.1rem 1.3rem !important;
    backdrop-filter: blur(24px) saturate(1.4);
    transition: transform 0.22s ease, box-shadow 0.22s ease;
    position: relative; overflow: hidden;
}
div[data-testid="stMetric"]::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #0ea5e9, #6366f1, #14b8a6);
    opacity: 0.65;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 48px rgba(0,0,0,0.5), 0 0 0 1px rgba(14,165,233,0.20);
    border-color: rgba(14,165,233,0.22) !important;
}
div[data-testid="stMetric"] label { color: #7a90a8 !important; font-size: 0.72rem !important; font-weight: 700 !important; letter-spacing: 0.05em !important; text-transform: uppercase !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; font-size: 1.55rem !important; font-weight: 500 !important; color: #eaf1fb !important; }

/* ── Custom metric cards (risk cards) ── */
.metric-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    border-radius: 14px;
    padding: 18px 22px;
    border-left: 4px solid #4a9eff;
    border-top: 1px solid rgba(255,255,255,0.06);
    border-right: 1px solid rgba(255,255,255,0.06);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 8px;
    backdrop-filter: blur(20px);
}
.metric-card.green  { border-left-color: #2ecc71; }
.metric-card.red    { border-left-color: #e74c3c; }
.metric-card.yellow { border-left-color: #f39c12; }
.metric-card.blue   { border-left-color: #4a9eff; }
.metric-label { font-size: 12px; color: #7a90a8; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.04em; }
.metric-value { font-size: 28px; font-weight: 700; color: #f0f6ff; font-family: 'JetBrains Mono', monospace; }
.metric-sub   { font-size: 11px; color: #56698a; margin-top: 2px; }

.prob-bar-wrap { background: rgba(255,255,255,0.06); border-radius: 6px; height: 10px; overflow: hidden; margin: 8px 0 4px; }
.prob-bar { height: 100%; border-radius: 6px; transition: width 0.3s; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }
.badge-low    { background: rgba(46,204,113,0.15); color: #2ecc71; }
.badge-medium { background: rgba(243,156,18,0.15); color: #f39c12; }
.badge-high   { background: rgba(231,76,60,0.15); color: #e74c3c; }

.section-title { font-size: 16px; font-weight: 600; color: #c8d8ea; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 6px; margin: 20px 0 12px; }

/* ── Charts / Dataframes ── */
.stPlotlyChart {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.065);
    border-radius: 18px; padding: 12px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.28);
}
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.065);
    border-radius: 16px; overflow: hidden;
}

/* ── Form controls (dropdowns) ── */
div[data-testid="stSelectbox"] > div, div[data-testid="stMultiSelect"] > div {
    background: rgba(255,255,255,0.045) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.095) !important;
}

/* ── Alerts / headings / divider ── */
.stAlert { border-radius: 12px !important; border-left-width: 3px !important; font-size: 0.875rem !important; }
h1 { font-family: 'Fraunces', serif !important; font-weight: 600 !important; color: #f0f6ff !important; }
h2 { font-weight: 700 !important; font-size: 1.2rem !important; color: #c8d8ea !important; }
h3 { font-weight: 600 !important; font-size: 0.85rem !important; color: #7a90a8 !important; text-transform: uppercase; letter-spacing: 0.05em; }
h4 { color: #c8d8ea !important; }
hr { border: none !important; border-top: 1px solid rgba(255,255,255,0.065) !important; margin: 1.3rem 0 !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.025); }
::-webkit-scrollbar-thumb { background: rgba(14,165,233,0.28); border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ── Load model ──────────────────────────────────────────────────
MODEL_PATH = "MSME_CompletionModel.pkl"

@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(
            f"Model file '{MODEL_PATH}' not found. Make sure it's in the same "
            "directory this app is run from (or update MODEL_PATH)."
        )
        st.stop()
    except Exception as e:
        st.error(
            f"Could not load the model pickle: {e}\n\n"
            "This often means the pickle was created with a different "
            "scikit-learn version than the one installed here — re-pickle "
            "with a matching environment, or pin scikit-learn in "
            "requirements.txt to match the training environment."
        )
        st.stop()

model = load_model()

# ═══════════════════════════════════════════════════════════════
# PIPELINE CONTRACT — derived by inspecting MSME_CompletionModel.pkl
# ═══════════════════════════════════════════════════════════════
# The pickle is a sklearn Pipeline:
#   ColumnTransformer(
#       num = StandardScaler()        -> ['Project Cost']
#       cat = OneHotEncoder(handle_unknown='ignore')
#             -> ['Province', 'Sector', 'Type of Ownership',
#                 'Size of Enterprise', 'Has_Prior_Funding']
#   ) -> LogisticRegression(class_weight='balanced', penalty='l1', C=1)
#
# IMPORTANT: OneHotEncoder was fit with handle_unknown='ignore'. That means
# any category the encoder doesn't recognise (wrong casing, a typo, a
# string 'True' instead of a Python bool True, etc.) is NOT rejected — it
# is silently zeroed out across that feature's whole one-hot block, and
# predict_proba() still returns a confident-looking number. This is a
# *silent* failure mode: verified empirically, sending 'Has_Prior_Funding':
# 'True' (string) instead of True (bool) shifts completion probability by
# 8+ points with no error or warning. Column order does NOT matter (the
# ColumnTransformer selects by name), but column NAMES, dtypes, and
# category spellings must match exactly.
REQUIRED_COLUMNS = [
    "Province", "Sector", "Type of Ownership",
    "Size of Enterprise", "Project Cost", "Has_Prior_Funding",
]

# Categories exactly as seen in OneHotEncoder.categories_ (ground truth —
# this is what the model was actually trained on).
VALID_CATEGORIES = {
    "Province": ["Aklan", "Antique", "Capiz", "Guimaras", "Iloilo", "Negros Occidental"],
    "Sector": [
        "Agriculture/Marine/Aquaculture", "Food Processing", "Furniture",
        "Gifts, Decors, Handicrafts", "Horticulture & Agriculture",
        "Metals & Engineering", "Others (grouped)",
    ],
    "Type of Ownership": ["Cooperative", "Corporation", "Partnership", "Single"],
    "Size of Enterprise": ["medium", "micro", "small"],
    "Has_Prior_Funding": [False, True],
}
# NOTE ON A REAL MISMATCH FOUND IN THIS TASK'S SPEC:
# The task brief describes the Sector category as "Others", but the
# pickle's OneHotEncoder was actually fit on "Others (grouped)". If the
# dashboard (or any upstream source) ever sends plain "Others", the
# encoder will silently treat it as unknown -> zeroed one-hot block.
# The cleaning function below maps common "others" spellings to the
# model's real category so this can't happen silently.

# ── Standardization / cleaning ──────────────────────────────────
def _clean_text(value):
    """Strip whitespace and collapse internal whitespace; keep None for NaNs."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "n/a", "na"}:
        return None
    return " ".join(text.split())


def _standardize_size(value):
    text = _clean_text(value)
    if text is None:
        return None
    key = text.lower()
    mapping = {
        "micro": "micro", "small": "small", "medium": "medium", "med": "medium",
    }
    return mapping.get(key, key)  # fall back to lowercased text for validation to catch


def _standardize_ownership(value):
    text = _clean_text(value)
    if text is None:
        return None
    key = text.lower()
    mapping = {
        "sole/single proprietorship": "Single",
        "sole proprietorship": "Single",
        "single proprietorship": "Single",
        "single": "Single",
        "sole": "Single",
        "cooperative": "Cooperative",
        "coop": "Cooperative",
        "corporation": "Corporation",
        "corp": "Corporation",
        "partnership": "Partnership",
    }
    return mapping.get(key, text)  # preserve original casing if no rule matches


def _standardize_province(value):
    text = _clean_text(value)
    if text is None:
        return None
    lookup = {p.lower(): p for p in VALID_CATEGORIES["Province"]}
    return lookup.get(text.lower(), text)


def _standardize_sector(value):
    text = _clean_text(value)
    if text is None:
        return None
    lookup = {s.lower(): s for s in VALID_CATEGORIES["Sector"]}
    key = text.lower()
    if key in lookup:
        return lookup[key]
    if key in {"others", "other", "others (grouped)", "misc", "miscellaneous"}:
        return "Others (grouped)"
    return text


def _standardize_bool(value):
    """Convert any common truthy/falsy representation to a real Python bool."""
    if isinstance(value, bool):
        return value
    text = _clean_text(value)
    if text is None:
        return None
    key = text.lower()
    if key in {"true", "yes", "y", "1", "prior", "has funding"}:
        return True
    if key in {"false", "no", "n", "0", "none", "no prior"}:
        return False
    return value  # unrecognized — leave as-is so validation can flag it


def _standardize_cost(value):
    """Coerce a project cost value (possibly '1,500,000' or '₱500000') to float."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = _clean_text(value)
    if text is None:
        return None
    cleaned = text.replace("₱", "").replace(",", "").replace(" ", "")
    try:
        return float(cleaned)
    except ValueError:
        return value  # leave as-is so validation can flag the bad type


def standardize_row(raw_row: dict) -> dict:
    """
    Reusable preprocessing function that normalizes ONE incoming dashboard
    record so its values match what the trained pipeline expects.

    - Trims whitespace and collapses casing differences ('Micro' -> 'micro').
    - Maps known synonyms to the model's real training labels
      ('Sole/Single Proprietorship' -> 'Single', 'Others' -> 'Others (grouped)').
    - Converts prior-funding flags ('Yes'/'No'/'True') to real Python bools.
    - Coerces Project Cost to a numeric type.
    - Leaves missing values as None so validation can catch them explicitly
      rather than letting OneHotEncoder silently zero them out.

    Does NOT touch the model. Only transforms incoming data before it is
    handed to model.predict_proba().
    """
    cleaned = dict(raw_row)  # never mutate the caller's dict
    if "Province" in cleaned:
        cleaned["Province"] = _standardize_province(cleaned["Province"])
    if "Sector" in cleaned:
        cleaned["Sector"] = _standardize_sector(cleaned["Sector"])
    if "Type of Ownership" in cleaned:
        cleaned["Type of Ownership"] = _standardize_ownership(cleaned["Type of Ownership"])
    if "Size of Enterprise" in cleaned:
        cleaned["Size of Enterprise"] = _standardize_size(cleaned["Size of Enterprise"])
    if "Has_Prior_Funding" in cleaned:
        cleaned["Has_Prior_Funding"] = _standardize_bool(cleaned["Has_Prior_Funding"])
    if "Project Cost" in cleaned:
        cleaned["Project Cost"] = _standardize_cost(cleaned["Project Cost"])
    return cleaned


def validate_row(cleaned_row: dict):
    """
    Validate a CLEANED row before it is sent to the pipeline.
    Returns (errors, warnings) — both lists of human-readable strings.
    `errors` mean prediction should not proceed (missing/bad data).
    `warnings` mean prediction can proceed but the result may be degraded
    (e.g. an unknown category will be silently zeroed by the encoder).
    """
    errors, warnings = [], []

    missing = [c for c in REQUIRED_COLUMNS if c not in cleaned_row]
    if missing:
        errors.append(f"Missing required column(s): {', '.join(missing)}")
        return errors, warnings  # can't validate further without the columns

    for col in REQUIRED_COLUMNS:
        if cleaned_row[col] is None:
            errors.append(f"'{col}' is missing/empty after cleaning.")

    if not errors:
        cost = cleaned_row["Project Cost"]
        if not isinstance(cost, (int, float)) or isinstance(cost, bool):
            errors.append(f"'Project Cost' must be numeric, got {type(cost).__name__} ({cost!r}).")
        elif cost < 0:
            warnings.append(f"'Project Cost' is negative ({cost:,.0f}); check the source data.")

        prior = cleaned_row["Has_Prior_Funding"]
        if not isinstance(prior, bool):
            errors.append(
                f"'Has_Prior_Funding' must resolve to True/False, got {prior!r} — "
                "the OneHotEncoder will silently zero this feature otherwise."
            )

        for col in ["Province", "Sector", "Type of Ownership", "Size of Enterprise"]:
            val = cleaned_row.get(col)
            if val is not None and val not in VALID_CATEGORIES[col]:
                warnings.append(
                    f"'{col}' value {val!r} is not one of the trained categories "
                    f"{VALID_CATEGORIES[col]}. It will be treated as unknown and "
                    "zeroed out by the encoder (prediction will silently ignore this feature)."
                )

    return errors, warnings


# ── Helper functions ────────────────────────────────────────────
def predict_msme(raw_row: dict, debug: bool = False):
    """
    Full, reliable prediction path for a single MSME record:
        raw input -> cleaned input -> validation -> DataFrame in the
        pipeline's expected column order -> predict() / predict_proba()
        -> risk tier.

    Returns a dict with every intermediate stage (used for the debug panel)
    plus the final probabilities/class. Raises no exceptions for bad data —
    instead returns errors=[...] and prob=None so the caller can display a
    clear message instead of a silent/misleading prediction.
    """
    cleaned = standardize_row(raw_row)
    errors, warnings = validate_row(cleaned)

    result = {
        "raw": raw_row,
        "cleaned": cleaned,
        "errors": errors,
        "warnings": warnings,
        "final_df": None,
        "predicted_class": None,
        "prob_completed": None,
        "prob_not_completed": None,
        "risk_tier": None,
    }

    if errors:
        return result  # prevent a silent/incorrect prediction

    # Build the DataFrame with columns in the exact order the pipeline
    # expects (cosmetic for ColumnTransformer, but avoids any ambiguity
    # and makes debugging easier).
    X = pd.DataFrame([{col: cleaned[col] for col in REQUIRED_COLUMNS}])
    result["final_df"] = X

    proba = model.predict_proba(X)[0]
    pred_class = int(model.predict(X)[0])

    result["prob_not_completed"] = float(proba[0])
    result["prob_completed"] = float(proba[1])
    result["predicted_class"] = "Completed" if pred_class == 1 else "Not Completed"
    result["risk_tier"] = risk_tier(float(proba[1]))[0]

    if debug:
        return result
    return result


def get_completion_prob(row):
    """
    Backward-compatible wrapper: predict probability of completion (class 1)
    for a single row, running it through the same cleaning + validation path
    as every other prediction in this dashboard.
    Row should be a dict with keys matching the model's feature names.
    """
    result = predict_msme(row)
    if result["errors"]:
        # Dummy-data generation should never hit this, but fail loudly
        # instead of silently returning a meaningless number if it does.
        raise ValueError("Invalid row passed to get_completion_prob: " + "; ".join(result["errors"]))
    return result["prob_completed"]

def risk_tier(prob):
    # These cutoffs (0.60 / 0.40) are illustrative bands for the dashboard UI,
    # centered on the model's 0.50 default decision threshold, not a
    # threshold derived from a precision/recall or Youden's-J analysis
    # against held-out labels (we only have the fitted pickle here, no
    # labeled validation set to optimize against).
    #
    # Sanity-checked by scoring the pipeline across all 6,048 possible
    # category combinations (6 provinces x 7 sectors x 4 ownership types
    # x 3 sizes x 2 funding flags x 6 cost points): probabilities range
    # from ~1% to ~91%, model.predict()'s own 0.5 cut lines up with the
    # midpoint of these bands, and the three tiers split the space into
    # roughly 31% / 28% / 41% of combinations (Low/Medium/High) rather
    # than dumping everything into one bucket. That means 0.60/0.40 are a
    # reasonable *symmetric-around-0.5* starting point and not obviously
    # broken — but they still haven't been validated against actual
    # completion outcomes, so they should be revisited once a labeled
    # holdout set is available for a real threshold-tuning pass.
    if prob >= 0.60:
        return "Low", "badge-low"
    if prob >= 0.40:
        return "Medium", "badge-medium"
    return "High", "badge-high"

def tier_color(prob):
    return "#2ecc71" if prob >= 0.60 else ("#f39c12" if prob >= 0.40 else "#e74c3c")

# ── Dummy data generators ──────────────────────────────────────
# Categorical Data Dummy
PROVINCES = ["Aklan", "Antique", "Capiz", "Guimaras", "Iloilo", "Negros Occidental"]
SECTORS = [
    "Agriculture/Marine/Aquaculture",
    "Food Processing",
    "Furniture",
    "Gifts, Decors, Handicrafts",
    "Horticulture & Agriculture",
    "Metals & Engineering",
    "Others (grouped)"
]
OWNERSHIP = ["Cooperative", "Corporation", "Partnership", "Single"]
SIZES = ["medium", "micro", "small"]
RISK_TIERS = ["Low", "Medium", "High"]

def generate_msme_data(n, status="approved"):
    """
    Generate dummy MSME records.
    status: 'approved' or 'applicant' – only affects ID prefix.
    """
    rows = []
    for i in range(1, n + 1):
        province = random.choice(PROVINCES)
        year = random.randint(2018, 2025)
        sector = random.choice(SECTORS)
        ownership = random.choice(OWNERSHIP)
        size = random.choice(SIZES)
        project_cost = int(np.random.randint(150_000, 2_500_000))
        has_prior = random.choice([True, False])

        # NOTE: Year is intentionally excluded here/ dropped during training
        row_dict = {
            "Province": province,
            "Sector": sector,
            "Type of Ownership": ownership,
            "Size of Enterprise": size,
            "Project Cost": project_cost,
            "Has_Prior_Funding": has_prior
        }
        prob = get_completion_prob(row_dict)
        tier, _ = risk_tier(prob)

        prefix = "A" if status == "approved" else "P"
        rows.append({
            "ID": f"MSME-{prefix}{i:03d}",
            "Beneficiary": f"Enterprise {chr(64 + i % 26 + 1)}{i:02d}",
            "Province": province,
            "Year": year,
            "Sector": sector,
            "Type of Ownership": ownership,
            "Size": size,
            "Project Cost (₱)": project_cost,
            "Has Prior Funding": has_prior,
            "Completion Prob.": prob,
            "Risk Tier": tier,
        })
    return pd.DataFrame(rows)

@st.cache_data
def generate_dummy_data():
    approved = generate_msme_data(30, "approved")
    applicants = generate_msme_data(20, "applicant")
    return approved, applicants

approved_df, applicant_df = generate_dummy_data()

# ── Shared plotly template ──────────────────────────────────────
PLOT_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Plus Jakarta Sans, sans-serif', color='#7a90a8', size=12),
    margin=dict(l=52, r=28, t=52, b=48),
    xaxis=dict(gridcolor='rgba(255,255,255,0.045)', zeroline=False,
               title_font=dict(size=11, color='#546a82'), tickfont=dict(size=10, color='#546a82'),
               linecolor='rgba(255,255,255,0.07)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.045)', zeroline=False,
               title_font=dict(size=11, color='#546a82'), tickfont=dict(size=10, color='#546a82'),
               linecolor='rgba(255,255,255,0.07)'),
    legend=dict(bgcolor='rgba(6,11,20,0.7)', bordercolor='rgba(255,255,255,0.07)',
                borderwidth=1, font=dict(size=11, color='#94a3b8')),
    hoverlabel=dict(bgcolor='#0b1628', font_family='JetBrains Mono, monospace',
                     font_size=12, bordercolor='rgba(14,165,233,0.3)', font_color='#dde3ee'),
)
RISK_COLOR_MAP = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}

ALL_LABEL = "All"

# ── Top bar: brand header ────────────────────────────────────────
bh1, bh2 = st.columns([0.08, 0.92])
with bh1:
    st.image("https://www.dost.gov.ph/images/DOSTLogo.png", width=52)
with bh2:
    st.markdown("""
    <div style="padding-top:2px;">
        <span style="font-family:'Fraunces',serif;font-size:1.25rem;font-weight:600;color:#f0f6ff;">ASENXO</span>
        <span style="font-size:0.7rem;color:#4a6080;letter-spacing:0.08em;text-transform:uppercase;font-weight:600;margin-left:8px;">Completion Dashboard</span>
        <span class="sidebar-badge" style="margin-left:10px;">⚡ DOST SETUP 4.0 · iFund</span>
    </div>
    """, unsafe_allow_html=True)

# ── Debug mode toggle ────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    debug_mode = st.checkbox(
        "🐞 Debug mode",
        value=False,
        help="Show every stage of the prediction pipeline: raw input → "
             "cleaned input → validation → final DataFrame → predict() → "
             "predict_proba() → risk tier."
    )

# ── Filters (dropdowns, on-page) ─────────────────────────────────
st.markdown('<div class="section-pill">🔍 Filters</div>', unsafe_allow_html=True)
f1, f2, f3 = st.columns(3)
with f1:
    sel_province = st.selectbox("Province", [ALL_LABEL] + PROVINCES, index=0)
with f2:
    sel_sector = st.selectbox("Sector", [ALL_LABEL] + SECTORS, index=0)
with f3:
    sel_tier = st.selectbox("Risk Tier", [ALL_LABEL] + RISK_TIERS, index=0)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════
# NEW MSME PREDICTION (manual entry) — exercises the full cleaning /
# validation / prediction pipeline for a single record, not just the
# pre-generated dummy dataset.
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="section-pill">🧪 Predict a New MSME Profile</div>', unsafe_allow_html=True)
st.caption(
    "Manual-entry inputs intentionally include labels the way real intake "
    "forms often phrase them (e.g. 'Sole/Single Proprietorship', 'Micro'). "
    "They pass through the same standardization step as any other source "
    "before the model sees them."
)

p1, p2, p3 = st.columns(3)
with p1:
    new_province = st.selectbox("Province ", PROVINCES, key="new_province")
    new_sector = st.selectbox("Sector ", SECTORS, key="new_sector")
with p2:
    new_ownership = st.selectbox(
        "Type of Ownership ",
        ["Cooperative", "Corporation", "Partnership", "Sole/Single Proprietorship"],
        key="new_ownership",
    )
    new_size = st.selectbox("Size of Enterprise ", ["Micro", "Small", "Medium"], key="new_size")
with p3:
    new_cost = st.number_input(
        "Project Cost (₱) ", min_value=0, value=500_000, step=50_000, key="new_cost"
    )
    new_prior = st.selectbox("Has Prior Funding? ", ["Yes", "No"], key="new_prior")

if st.button("Predict Completion Probability", type="primary"):
    raw_input_row = {
        "Province": new_province,
        "Sector": new_sector,
        "Type of Ownership": new_ownership,
        "Size of Enterprise": new_size,
        "Project Cost": new_cost,
        "Has_Prior_Funding": new_prior,
    }
    result = predict_msme(raw_input_row, debug=True)

    if result["errors"]:
        for err in result["errors"]:
            st.error(f"❌ {err}")
    else:
        for warn in result["warnings"]:
            st.warning(f"⚠️ {warn}")

        prob_completed = result["prob_completed"]
        prob_not = result["prob_not_completed"]
        pred_class = result["predicted_class"]
        tier_label, tier_cls = risk_tier(prob_completed)
        color = tier_color(prob_completed)

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""<div class="metric-card green">
                <div class="metric-label">Probability of Completion</div>
                <div class="metric-value">{prob_completed*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class="metric-card red">
                <div class="metric-label">Probability of Non‑Completion</div>
                <div class="metric-value">{prob_not*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""<div class="metric-card" style="border-left-color:{color};">
                <div class="metric-label">Predicted Class</div>
                <div class="metric-value" style="font-size:20px;color:{color};">{pred_class}</div>
                <div class="metric-sub">Risk Tier: <span class="badge {tier_cls}">{tier_label}</span></div>
            </div>""", unsafe_allow_html=True)

        if debug_mode:
            with st.expander("🐞 Debug: full prediction pipeline", expanded=True):
                st.markdown("**1. Incoming raw data**")
                st.json(result["raw"])
                st.markdown("**2. Cleaned data** (after `standardize_row`)")
                st.json(result["cleaned"])
                st.markdown("**3. Final DataFrame sent to the model**")
                st.dataframe(result["final_df"], hide_index=True, use_container_width=True)
                st.markdown("**4. `model.predict()`**")
                st.code(pred_class, language="text")
                st.markdown("**5. `model.predict_proba()`**")
                st.code(f"[Not Completed: {prob_not:.4f}, Completed: {prob_completed:.4f}]", language="text")
                st.markdown("**6. Risk Tier**")
                st.code(tier_label, language="text")

st.markdown("---")

# 
province_list = PROVINCES if sel_province == ALL_LABEL else [sel_province]
sector_list = SECTORS if sel_sector == ALL_LABEL else [sel_sector]
tier_list = RISK_TIERS if sel_tier == ALL_LABEL else [sel_tier]

# Apply filters
filt_approved = approved_df[
    approved_df["Province"].isin(province_list) &
    approved_df["Sector"].isin(sector_list) &
    approved_df["Risk Tier"].isin(tier_list)
]
filt_applicant = applicant_df[
    applicant_df["Province"].isin(province_list) &
    applicant_df["Sector"].isin(sector_list) &
    applicant_df["Risk Tier"].isin(tier_list)
]

total_approved   = len(filt_approved)
total_applicants = len(filt_applicant)
high_risk_appr   = (filt_approved["Risk Tier"] == "High").sum()
high_risk_appl   = (filt_applicant["Risk Tier"] == "High").sum()
avg_prob_appr    = filt_approved["Completion Prob."].mean() if total_approved else 0
avg_prob_appl    = filt_applicant["Completion Prob."].mean() if total_applicants else 0

combined = pd.concat([
    filt_approved.assign(Group="Approved"),
    filt_applicant.assign(Group="Applicant"),
])
# ═══════════════════════════════════════════════════════════════
# PORTFOLIO KPIs
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="section-pill">🎯 Portfolio Snapshot</div>', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Approved MSMEs", total_approved)
with k2:
    st.metric("Applying MSMEs", total_applicants)
with k3:
    st.metric("High-Risk (Approved)", high_risk_appr,
              delta=f"-{high_risk_appr} need follow-up", delta_color="inverse")
with k4:
    st.metric("Avg Completion Prob. (Approved)", f"{avg_prob_appr*100:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# OVERVIEW CHARTS
# ═══════════════════════════════════════════════════════════════
c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="section-pill">🗺️ Risk Tier Distribution</div>', unsafe_allow_html=True)
    if not combined.empty:
        tier_counts = combined.groupby(["Group", "Risk Tier"]).size().reset_index(name="Count")
        fig = px.bar(tier_counts, x="Group", y="Count", color="Risk Tier",
                     color_discrete_map=RISK_COLOR_MAP, barmode="stack",
                     category_orders={"Risk Tier": ["Low", "Medium", "High"]},
                     title="Risk Tier by Group")
        fig.update_layout(**PLOT_LAYOUT, title_font=dict(size=13, color='#c8d8ea'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No records match the current filters.")

with c2:
    st.markdown('<div class="section-pill">🌏 Avg Completion Probability by Province</div>', unsafe_allow_html=True)
    if not combined.empty:
        prov_avg = combined.groupby("Province")["Completion Prob."].mean().reset_index()
        prov_avg = prov_avg.sort_values("Completion Prob.", ascending=False)
        fig2 = px.bar(prov_avg, x="Province", y="Completion Prob.",
                      color="Completion Prob.", color_continuous_scale="Teal",
                      title="Avg Completion Probability by Province")
        fig2.update_traces(hovertemplate="%{x}: %{y:.1%}")
        fig2.update_yaxes(tickformat=".0%")
        fig2.update_coloraxes(showscale=False)
        fig2.update_layout(**PLOT_LAYOUT, title_font=dict(size=13, color='#c8d8ea'))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No records match the current filters.")

st.markdown('<div class="section-pill">🏭 Completion Probability by Sector</div>', unsafe_allow_html=True)
if not combined.empty:
    sector_avg = combined.groupby("Sector")["Completion Prob."].mean().reset_index()
    sector_avg = sector_avg.sort_values("Completion Prob.", ascending=True)
    fig3 = px.bar(sector_avg, y="Sector", x="Completion Prob.", orientation="h",
                  color="Completion Prob.", color_continuous_scale="Bluyl",
                  title="Avg Completion Probability by Sector")
    fig3.update_traces(hovertemplate="%{y}: %{x:.1%}")
    fig3.update_xaxes(tickformat=".0%")
    fig3.update_coloraxes(showscale=False)
    fig3.update_layout(**PLOT_LAYOUT, title_font=dict(size=13, color='#c8d8ea'), height=380)
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No records match the current filters.")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════
# APPROVED / FUNDED MSMEs
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🏢 Approved / Funded MSMEs — Completion Probability</div>',
            unsafe_allow_html=True)
st.caption(
    "Completion probability is estimated from the project profile using the trained model. "
    "High risk (prob < 0.40) indicates profiles similar to historically non‑completing projects."
)

sa1, sa2, sa3 = st.columns(3)
with sa1:
    st.markdown(f"""<div class="metric-card green">
        <div class="metric-label">Avg Completion Probability</div>
        <div class="metric-value">{avg_prob_appr*100:.1f}%</div>
        <div class="metric-sub">Across {total_approved} approved MSMEs</div>
    </div>""", unsafe_allow_html=True)
with sa2:
    low_risk = (filt_approved["Risk Tier"] == "Low").sum()
    st.markdown(f"""<div class="metric-card blue">
        <div class="metric-label">Low‑Risk (Approved)</div>
        <div class="metric-value">{low_risk}</div>
        <div class="metric-sub">Prob ≥ 60%</div>
    </div>""", unsafe_allow_html=True)
with sa3:
    high_risk = (filt_approved["Risk Tier"] == "High").sum()
    st.markdown(f"""<div class="metric-card red">
        <div class="metric-label">High‑Risk (Approved)</div>
        <div class="metric-value">{high_risk}</div>
        <div class="metric-sub">Prob < 40% – monitor closely</div>
    </div>""", unsafe_allow_html=True)

st.markdown("#### Approved MSME Detail")
if filt_approved.empty:
    st.info("No approved MSMEs match the current filters.")
else:
    display_appr = filt_approved.copy()
    display_appr["Completion Prob."] = display_appr["Completion Prob."].apply(lambda x: f"{x*100:.1f}%")
    display_appr["Project Cost (₱)"] = display_appr["Project Cost (₱)"].apply(lambda x: f"₱{x:,.0f}")
    display_appr["Has Prior Funding"] = display_appr["Has Prior Funding"].apply(lambda x: "Yes" if x else "No")

    col_order = ["ID", "Beneficiary", "Province", "Year", "Sector",
                 "Type of Ownership", "Size", "Project Cost (₱)",
                 "Has Prior Funding", "Completion Prob.", "Risk Tier"]
    st.dataframe(display_appr[col_order], use_container_width=True, hide_index=True)

    st.markdown("#### 🔎 MSME Deep Dive")
    selected_id = st.selectbox(
        "Select an approved MSME to inspect:",
        filt_approved["ID"].tolist(),
        key="appr_select"
    )
    row = filt_approved[filt_approved["ID"] == selected_id].iloc[0]

    # Re-run the full pipeline for this row so the deep-dive shows every
    # confidence figure (not just the completion probability cached in
    # the dummy dataframe), plus the debug trail if debug mode is on.
    deep_dive_raw = {
        "Province": row["Province"],
        "Sector": row["Sector"],
        "Type of Ownership": row["Type of Ownership"],
        "Size of Enterprise": row["Size"],
        "Project Cost": row["Project Cost (₱)"],
        "Has_Prior_Funding": row["Has Prior Funding"],
    }
    deep_dive_result = predict_msme(deep_dive_raw, debug=True)
    prob = deep_dive_result["prob_completed"] if not deep_dive_result["errors"] else row["Completion Prob."]
    prob_not = deep_dive_result["prob_not_completed"] if not deep_dive_result["errors"] else (1 - prob)
    pred_class = deep_dive_result["predicted_class"] if not deep_dive_result["errors"] else None

    d1, d2 = st.columns(2)
    with d1:
        tier_label, tier_cls = risk_tier(prob)
        color = tier_color(prob)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Probability of Completion</div>
            <div class="metric-value" style="color:{color}">{prob*100:.1f}%</div>
            <div class="prob-bar-wrap"><div class="prob-bar"
                style="width:{prob*100:.1f}%; background:{color};"></div></div>
            <div class="metric-sub">Probability of Non‑Completion: {prob_not*100:.1f}%
                &nbsp;·&nbsp; Predicted Class: <strong>{pred_class or "—"}</strong>
            </div>
            <div class="metric-sub">Risk Tier:
                <span class="badge {tier_cls}">{tier_label}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        if debug_mode:
            with st.expander("🐞 Debug: pipeline trail for this record"):
                st.markdown("**1. Incoming raw data**")
                st.json(deep_dive_result["raw"])
                st.markdown("**2. Cleaned data**")
                st.json(deep_dive_result["cleaned"])
                if deep_dive_result["warnings"]:
                    for w in deep_dive_result["warnings"]:
                        st.warning(f"⚠️ {w}")
                if deep_dive_result["errors"]:
                    for e in deep_dive_result["errors"]:
                        st.error(f"❌ {e}")
                else:
                    st.markdown("**3. Final DataFrame sent to the model**")
                    st.dataframe(deep_dive_result["final_df"], hide_index=True, use_container_width=True)
                    st.markdown("**4–5. predict() / predict_proba()**")
                    st.code(
                        f"predict()       -> {pred_class}\n"
                        f"predict_proba() -> [Not Completed: {prob_not:.4f}, Completed: {prob:.4f}]",
                        language="text",
                    )
                    st.markdown("**6. Risk Tier**")
                    st.code(tier_label, language="text")

    with d2:
        st.markdown("**Profile**")
        profile_data = {
            "Field": ["Beneficiary", "Province", "Year", "Sector",
                      "Type of Ownership", "Size", "Project Cost", "Has Prior Funding"],
            "Value": [
                row["Beneficiary"], row["Province"], row["Year"],
                row["Sector"], row["Type of Ownership"], row["Size"],
                f"₱{row['Project Cost (₱)']:,}",
                "Yes" if row["Has Prior Funding"] else "No"
            ]
        }
        st.dataframe(pd.DataFrame(profile_data), hide_index=True, use_container_width=True)

    if prob >= 0.60:
        st.success("✅ Profile aligns with historical completions – low risk.")
    elif prob >= 0.40:
        st.warning("⚠️ Moderate risk – consider additional support or monitoring.")
    else:
        st.error("🔴 High risk – resembles non‑completing projects; Intense monitoring needed.")

st.markdown("---")

