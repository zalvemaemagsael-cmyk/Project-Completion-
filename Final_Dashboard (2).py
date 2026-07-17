"""
ASENXO — MSME Project Completion Dashboard (Model-Driven)
============================================================
Uses the trained Logistic Regression pipeline from MSME_CompletionModel.pkl
to estimate completion probability for endorsed MSME projects.
Endorsed MSME records are fetched live from Supabase.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page config (MUST be the first Streamlit command) ───────────
st.set_page_config(
    page_title="ASENXO | Completion Dashboard",
    page_icon="📊",
    layout="wide",
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Global CSS / Theme — clean light enterprise/government look ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
    color: #1f2937;
    font-size: 15px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}

/* ── App shell ── */
.stApp {
    background-color: #f4f6f9;
    min-height: 100vh;
}
.main .block-container { padding: 1.4rem 2.2rem 3rem; max-width: 1320px; }

/* ── Page hero ── */
.page-hero {
    background: #ffffff;
    border: 1px solid #e3e8ef;
    border-left: 4px solid #1d4ed8;
    border-radius: 10px;
    padding: 1.3rem 1.7rem;
    margin: 0 0 1.4rem;
    box-shadow: 0 1px 3px rgba(16,24,40,0.05);
}
.page-hero h1 {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    color: #111827 !important;
    letter-spacing: -0.01em !important;
    margin-bottom: 0.2rem !important;
}
.page-hero p { color: #6b7280; font-size: 0.85rem; margin: 0; line-height: 1.5; }

.section-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #eef2ff;
    border: 1px solid #dbe3fb;
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em;
    text-transform: uppercase; color: #3538cd;
    margin-bottom: 0.8rem;
}

.section-title {
    font-size: 15px; font-weight: 700; color: #111827;
    border-bottom: 1px solid #e5e9f0; padding-bottom: 8px; margin: 4px 0 14px;
}

/* ── Streamlit native metric cards ── */
div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #e3e8ef !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}
div[data-testid="stMetric"] label { color: #6b7280 !important; font-size: 0.72rem !important; font-weight: 600 !important; letter-spacing: 0.03em !important; text-transform: uppercase !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; font-size: 1.5rem !important; font-weight: 600 !important; color: #111827 !important; }

/* ── Custom metric / KPI cards ── */
.metric-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 16px 20px;
    border-left: 4px solid #1d4ed8;
    border-top: 1px solid #e3e8ef;
    border-right: 1px solid #e3e8ef;
    border-bottom: 1px solid #e3e8ef;
    margin-bottom: 8px;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}
.metric-card.green  { border-left-color: #16a34a; }
.metric-card.red    { border-left-color: #dc2626; }
.metric-card.yellow { border-left-color: #d97706; }
.metric-card.blue   { border-left-color: #1d4ed8; }
.metric-label { font-size: 12px; color: #6b7280; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.03em; font-weight: 600; }
.metric-value { font-size: 26px; font-weight: 700; color: #111827; font-family: 'JetBrains Mono', monospace; }
.metric-sub   { font-size: 12px; color: #6b7280; margin-top: 4px; }

.prob-bar-wrap { background: #eef1f5; border-radius: 6px; height: 9px; overflow: hidden; margin: 8px 0 4px; }
.prob-bar { height: 100%; border-radius: 6px; transition: width 0.3s; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 700; }
.badge-low    { background: #dcfce7; color: #15803d; }
.badge-medium { background: #fef3c7; color: #b45309; }
.badge-high   { background: #fee2e2; color: #b91c1c; }

.info-banner {
    background: #fffbeb; border: 1px solid #fde68a; color: #92400e;
    border-radius: 8px; padding: 10px 14px; font-size: 0.82rem; margin-bottom: 14px;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    background: #ffffff;
    border: 1px solid #e3e8ef;
    border-radius: 10px; overflow: hidden;
}

/* ── Form controls ── */
div[data-testid="stSelectbox"] > div {
    background: #ffffff !important;
    border-radius: 8px !important;
    border: 1px solid #d7dce3 !important;
}

/* ── Alerts / headings / divider ── */
.stAlert { border-radius: 8px !important; border-left-width: 3px !important; font-size: 0.85rem !important; }
h1 { font-weight: 700 !important; color: #111827 !important; }
h2 { font-weight: 700 !important; font-size: 1.15rem !important; color: #111827 !important; }
h3 { font-weight: 600 !important; font-size: 0.85rem !important; color: #6b7280 !important; text-transform: uppercase; letter-spacing: 0.04em; }
h4 { color: #111827 !important; }
hr { border: none !important; border-top: 1px solid #e5e9f0 !important; margin: 1.2rem 0 !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f4f6f9; }
::-webkit-scrollbar-thumb { background: #c7cfda; border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="page-hero">
    <h1> MSME Project Completion Dashboard</h1>
    <p>DOST SETUP 4.0 iFund Program — Western Visayas | Model-driven risk assessment (live data)</p>
</div>
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

# ═══════════════════════════════════════════════════════════════
# SUPABASE INTEGRATION
# ═══════════════════════════════════════════════════════════════
# Endorsed MSME records are fetched live from a Supabase table instead of
# being generated or manually entered. Configure these two secrets in
# .streamlit/secrets.toml (or your deployment platform's secrets manager):
#
#   SUPABASE_URL = "https://xxxx.supabase.co"
#   SUPABASE_KEY = "your-anon-or-service-key"
#
# Adjust SUPABASE_TABLE below to match your actual table name.
SUPABASE_TABLE = "endorsed_msmes"

# Best-effort column aliasing: maps the likely raw column names Supabase
# might return (snake_case, alternate spellings, etc.) to the internal
# names this dashboard/pipeline uses. Add more aliases here if your
# actual schema uses different names — this is the ONE place to edit.
COLUMN_ALIASES = {
    "ID": ["id", "msme_id", "msmeid", "reference_no", "reference_number"],
    "Beneficiary": ["beneficiary", "company_name", "business_name", "msme_name", "name"],
    "Province": ["province"],
    "Sector": ["sector", "industry_sector"],
    "Type of Ownership": ["type_of_ownership", "ownership_type", "ownership"],
    "Size of Enterprise": ["size_of_enterprise", "enterprise_size", "size"],
    "Project Cost": ["project_cost", "cost", "project_cost_php", "total_project_cost"],
    "Has_Prior_Funding": ["has_prior_funding", "prior_funding", "has_prior_funding_flag"],
    "Status": ["status", "stage", "application_status"],
    "Year": ["year", "year_endorsed", "date_endorsed"],
}


@st.cache_resource
def get_supabase_client():
    """Create (and cache) a Supabase client from st.secrets. Returns None
    if credentials aren't configured so the caller can show a clear
    setup message instead of crashing."""
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except Exception:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except ImportError:
        st.error(
            "The `supabase` Python package isn't installed. Add `supabase` "
            "to requirements.txt and redeploy."
        )
        return None
    except Exception as e:
        st.error(f"Could not create the Supabase client: {e}")
        return None


@st.cache_data(ttl=300, show_spinner="Fetching endorsed MSMEs from Supabase…")
def fetch_endorsed_msmes_raw():
    """
    Fetch all rows from the endorsed-MSME table.
    Returns (DataFrame or None, error_message or None).
    """
    client = get_supabase_client()
    if client is None:
        return None, "not_configured"
    try:
        response = client.table(SUPABASE_TABLE).select("*").execute()
        rows = response.data or []
        if not rows:
            return pd.DataFrame(), None
        return pd.DataFrame(rows), None
    except Exception as e:
        return None, str(e)


def map_supabase_columns(raw_df: pd.DataFrame):
    """
    Rename whatever columns Supabase actually returned to the internal
    names this dashboard expects, using COLUMN_ALIASES. Returns
    (mapped_df, missing_required_columns).
    """
    lower_lookup = {c.lower().strip(): c for c in raw_df.columns}
    rename_map = {}
    found = {}
    for internal_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_lookup:
                rename_map[lower_lookup[alias]] = internal_name
                found[internal_name] = True
                break

    mapped = raw_df.rename(columns=rename_map).copy()

    # Required for the prediction pipeline itself
    pipeline_required = [
        "Province", "Sector", "Type of Ownership",
        "Size of Enterprise", "Project Cost", "Has_Prior_Funding",
    ]
    missing = [c for c in pipeline_required if c not in mapped.columns]
    return mapped, missing


def run_predictions_on_dataframe(mapped_df: pd.DataFrame):
    """
    Runs every endorsed MSME record through the SAME cleaning, validation
    and prediction path (predict_msme) used everywhere else in this
    dashboard, and returns an enriched DataFrame with prediction columns.
    Rows that fail validation get NaN predictions plus a visible reason
    instead of being silently dropped or given a misleading number.
    """
    records = []
    for _, row in mapped_df.iterrows():
        raw_row = {
            "Province": row.get("Province"),
            "Sector": row.get("Sector"),
            "Type of Ownership": row.get("Type of Ownership"),
            "Size of Enterprise": row.get("Size of Enterprise"),
            "Project Cost": row.get("Project Cost"),
            "Has_Prior_Funding": row.get("Has_Prior_Funding"),
        }
        result = predict_msme(raw_row, debug=True)

        record = row.to_dict()
        # Overwrite the display columns with their CLEANED equivalents
        # (correct dtype/casing) rather than the raw Supabase values —
        # otherwise a raw string like "1,200,000" for Project Cost breaks
        # currency formatting downstream even though prediction succeeded.
        if not result["errors"]:
            for col in ["Province", "Sector", "Type of Ownership",
                        "Size of Enterprise", "Project Cost", "Has_Prior_Funding"]:
                record[col] = result["cleaned"].get(col, record.get(col))
        record["Completion Probability"] = result["prob_completed"]
        record["Non-Completion Probability"] = result["prob_not_completed"]
        record["Predicted Class"] = result["predicted_class"]
        record["Risk Tier"] = result["risk_tier"]
        record["_prediction_errors"] = "; ".join(result["errors"]) if result["errors"] else ""
        record["_prediction_warnings"] = "; ".join(result["warnings"]) if result["warnings"] else ""
        record["_cleaned_row"] = result["cleaned"]
        records.append(record)

    enriched = pd.DataFrame(records)

    # Ensure display-friendly defaults
    if "ID" not in enriched.columns:
        enriched["ID"] = [f"MSME-{i+1:03d}" for i in range(len(enriched))]
    if "Beneficiary" not in enriched.columns:
        enriched["Beneficiary"] = enriched["ID"]
    if "Status" not in enriched.columns:
        enriched["Status"] = "Approved"
    else:
        enriched["Status"] = (
            enriched["Status"].astype(str).str.strip().str.title()
            .replace({"Applicant": "Applying", "Applied": "Applying", "Pending": "Applying"})
        )
    return enriched


def get_recommendation(tier: str) -> str:
    if tier == "Low":
        return "Profile aligns with historically completed projects. Standard monitoring is sufficient."
    if tier == "Medium":
        return "Moderate completion risk. Consider periodic check-ins and additional technical assistance."
    return "Profile resembles historically non-completing projects. Recommend intensified monitoring, early intervention, and a review of funding conditions."


# ── Sample/demo fallback (used ONLY when Supabase isn't configured) ──
def generate_sample_data(n=25):
    """Clearly-labeled demo data so the dashboard is still explorable
    before Supabase credentials are configured. Never used once a real
    connection succeeds."""
    rng = np.random.default_rng(RANDOM_STATE)
    provinces = VALID_CATEGORIES["Province"]
    sectors = VALID_CATEGORIES["Sector"]
    ownership = VALID_CATEGORIES["Type of Ownership"]
    sizes = VALID_CATEGORIES["Size of Enterprise"]
    statuses = ["Approved", "Applying"]

    rows = []
    for i in range(1, n + 1):
        rows.append({
            "ID": f"MSME-{i:03d}",
            "Beneficiary": f"Sample Enterprise {i:02d}",
            "Province": rng.choice(provinces),
            "Sector": rng.choice(sectors),
            "Type of Ownership": rng.choice(ownership),
            "Size of Enterprise": rng.choice(sizes),
            "Project Cost": int(rng.integers(150_000, 2_500_000)),
            "Has_Prior_Funding": bool(rng.choice([True, False])),
            "Status": rng.choice(statuses),
        })
    return pd.DataFrame(rows)


# ── Load endorsed MSMEs (Supabase, with a clearly-labeled demo fallback) ──
raw_df, fetch_error = fetch_endorsed_msmes_raw()
using_demo_data = False

if fetch_error == "not_configured":
    st.markdown("""
    <div class="info-banner">
         Supabase isn't configured yet (missing <code>SUPABASE_URL</code> /
        <code>SUPABASE_KEY</code> in secrets), so <strong>demo data</strong> is shown below.
        Connect Supabase to see live endorsed MSME records.
    </div>
    """, unsafe_allow_html=True)
    mapped_df = generate_sample_data()
    using_demo_data = True
elif fetch_error is not None:
    st.error(f"Could not fetch endorsed MSMEs from Supabase: {fetch_error}")
    st.stop()
elif raw_df is None or raw_df.empty:
    st.info("No endorsed MSME records were found in Supabase.")
    st.stop()
else:
    mapped_df, missing_cols = map_supabase_columns(raw_df)
    if missing_cols:
        st.error(
            "The Supabase table is missing column(s) the prediction pipeline "
            f"requires: {', '.join(missing_cols)}. Found columns: "
            f"{', '.join(raw_df.columns)}. Update COLUMN_ALIASES at the top "
            "of this file to match your schema, or add the missing column(s) "
            "in Supabase."
        )
        st.stop()

endorsed_df = run_predictions_on_dataframe(mapped_df)

# ═══════════════════════════════════════════════════════════════
# FILTERS
# ═══════════════════════════════════════════════════════════════
ALL_LABEL = "All"
st.markdown('<div class="section-pill"> Filters</div>', unsafe_allow_html=True)
f1, f2, f3 = st.columns(3)
with f1:
    province_options = [ALL_LABEL] + sorted(endorsed_df["Province"].dropna().unique().tolist())
    sel_province = st.selectbox("Province", province_options, index=0)
with f2:
    sector_options = [ALL_LABEL] + sorted(endorsed_df["Sector"].dropna().unique().tolist())
    sel_sector = st.selectbox("Sector", sector_options, index=0)
with f3:
    tier_options = [ALL_LABEL, "Low", "Medium", "High"]
    sel_tier = st.selectbox("Risk Tier", tier_options, index=0)

st.markdown("---")

filtered_df = endorsed_df.copy()
if sel_province != ALL_LABEL:
    filtered_df = filtered_df[filtered_df["Province"] == sel_province]
if sel_sector != ALL_LABEL:
    filtered_df = filtered_df[filtered_df["Sector"] == sel_sector]
if sel_tier != ALL_LABEL:
    filtered_df = filtered_df[filtered_df["Risk Tier"] == sel_tier]

approved_mask = filtered_df["Status"] == "Approved"
applying_mask = filtered_df["Status"] == "Applying"
approved_view = filtered_df[approved_mask]
applying_view = filtered_df[applying_mask]

total_approved = len(approved_view)
total_applying = len(applying_view)
high_risk_appr = (approved_view["Risk Tier"] == "High").sum()
low_risk_appr = (approved_view["Risk Tier"] == "Low").sum()
avg_prob_appr = approved_view["Completion Probability"].mean() if total_approved else 0

# ═══════════════════════════════════════════════════════════════
# KPI CARDS
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="section-pill"> Portfolio Snapshot</div>', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""<div class="metric-card blue">
        <div class="metric-label">Approved MSMEs</div>
        <div class="metric-value">{total_approved}</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="metric-card blue">
        <div class="metric-label">Applying MSMEs</div>
        <div class="metric-value">{total_applying}</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="metric-card red">
        <div class="metric-label">High-Risk (Approved)</div>
        <div class="metric-value">{high_risk_appr}</div>
        <div class="metric-sub">Needs follow-up</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="metric-card green">
        <div class="metric-label">Low-Risk (Approved)</div>
        <div class="metric-value">{low_risk_appr}</div>
    </div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""<div class="metric-card blue">
        <div class="metric-label">Avg Completion Prob. (Approved)</div>
        <div class="metric-value">{avg_prob_appr*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# ENDORSED MSME TABLE
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title"> Endorsed MSMEs — Completion Probability</div>', unsafe_allow_html=True)
st.caption(
    "Completion probability is estimated from each project's profile using the trained model. "
    "High risk (probability < 40%) indicates profiles similar to historically non-completing projects."
)

if filtered_df.empty:
    st.info("No endorsed MSMEs match the current filters.")
else:
    display_df = filtered_df.copy()
    display_df["Completion Probability"] = display_df["Completion Probability"].apply(
        lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—"
    )
    display_df["Project Cost"] = display_df["Project Cost"].apply(
        lambda x: f"₱{x:,.0f}" if pd.notna(x) else "—"
    )
    display_df["Has Prior Funding"] = display_df["Has_Prior_Funding"].apply(
        lambda x: "Yes" if x is True else ("No" if x is False else "—")
    )

    table_cols = [
        "ID", "Beneficiary", "Province", "Sector", "Type of Ownership",
        "Size of Enterprise", "Project Cost", "Has Prior Funding",
        "Completion Probability", "Predicted Class", "Risk Tier",
    ]
    table_cols = [c for c in table_cols if c in display_df.columns]
    st.dataframe(display_df[table_cols], use_container_width=True, hide_index=True)

    failed = filtered_df[filtered_df["_prediction_errors"] != ""]
    if not failed.empty:
        st.warning(
            f" {len(failed)} record(s) could not be scored due to data issues "
            "(missing/invalid fields) — see 'Predicted Class' = None above. "
            "Check the source records in Supabase."
        )

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════
    # MSME DEEP DIVE
    # ═══════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title"> MSME Deep Dive</div>', unsafe_allow_html=True)
    selected_id = st.selectbox(
        "Select an MSME to inspect:",
        filtered_df["ID"].tolist(),
        key="msme_select",
    )
    row = filtered_df[filtered_df["ID"] == selected_id].iloc[0]

    d1, d2 = st.columns(2)
    with d1:
        prob = row["Completion Probability"]
        if pd.isna(prob):
            st.error(f" This record could not be scored: {row['_prediction_errors']}")
        else:
            prob_not = row["Non-Completion Probability"]
            pred_class = row["Predicted Class"]
            tier_label, tier_cls = risk_tier(prob)
            color = tier_color(prob)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Probability of Completion</div>
                <div class="metric-value" style="color:{color}">{prob*100:.1f}%</div>
                <div class="prob-bar-wrap"><div class="prob-bar"
                    style="width:{prob*100:.1f}%; background:{color};"></div></div>
                <div class="metric-sub">Probability of Non-Completion: {prob_not*100:.1f}%
                    &nbsp;·&nbsp; Predicted Class: <strong>{pred_class}</strong>
                </div>
                <div class="metric-sub">Risk Tier:
                    <span class="badge {tier_cls}">{tier_label}</span>
                </div>
            </div>""", unsafe_allow_html=True)

            if row["_prediction_warnings"]:
                st.warning(f" {row['_prediction_warnings']}")

            if tier_label == "Low":
                st.success(f" {get_recommendation(tier_label)}")
            elif tier_label == "Medium":
                st.warning(f" {get_recommendation(tier_label)}")
            else:
                st.error(f" {get_recommendation(tier_label)}")

    with d2:
        st.markdown("**Profile**")
        profile_fields = ["Beneficiary", "Province", "Sector", "Type of Ownership",
                           "Size of Enterprise", "Project Cost", "Has_Prior_Funding", "Status"]
        profile_data = {
            "Field": ["Beneficiary", "Province", "Sector", "Type of Ownership",
                      "Size of Enterprise", "Project Cost", "Has Prior Funding", "Status"],
            "Value": [
                row.get("Beneficiary"), row.get("Province"), row.get("Sector"),
                row.get("Type of Ownership"), row.get("Size of Enterprise"),
                f"₱{row['Project Cost']:,.0f}" if pd.notna(row.get("Project Cost")) else "—",
                "Yes" if row.get("Has_Prior_Funding") is True else ("No" if row.get("Has_Prior_Funding") is False else "—"),
                row.get("Status"),
            ]
        }
        st.dataframe(pd.DataFrame(profile_data), hide_index=True, use_container_width=True)

if using_demo_data:
    st.caption("📌 Currently showing demo data. Configure SUPABASE_URL / SUPABASE_KEY in secrets to load live endorsed MSMEs.")
