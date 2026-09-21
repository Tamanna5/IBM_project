"""
streamlit_app.py — Home page & navigation entry point
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🏡 House Price Predictor")
st.sidebar.markdown(
    """
**King County, WA — 2014–2015**

Use the pages in this sidebar to:
- 📊 **EDA** — Explore the dataset
- 📈 **Model Performance** — Inspect metrics & diagnostics
- 🏠 **Predict** — Get a live price estimate
"""
)
st.sidebar.divider()
st.sidebar.caption("Model: GradientBoostingRegressor · scikit-learn")
st.sidebar.caption("Dataset: King County House Sales (21,613 rows)")

# ── Hero Section ───────────────────────────────────────────────────────────────
st.title("🏡 House Price Prediction Dashboard")
st.markdown(
    """
> **End-to-end machine learning project** — from raw CSV to interactive predictions.  
> Built with **Python · scikit-learn · Streamlit · Plotly**.
"""
)

col1, col2, col3 = st.columns(3)
col1.info("📊 **Page 1 — EDA**\nExplore distributions, correlations, and geographic maps of King County house sales.")
col2.info("📈 **Page 2 — Model Performance**\nView MAE, RMSE, R², feature importances, and residual diagnostics.")
col3.info("🏠 **Page 3 — Predict**\nEnter house features and get an instant AI-powered price estimate.")

st.divider()

# ── Dataset Summary ────────────────────────────────────────────────────────────
st.subheader("📂 Dataset: King County House Sales")
st.markdown(
    "Source dataset contains **21,613 home sales** in King County, WA (May 2014 – May 2015). "
    "The target variable is `price` (sale price in USD)."
)

features_info = {
    "Feature": [
        "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
        "floors", "waterfront", "view", "condition", "grade",
        "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
        "lat / long", "sqft_living15", "sqft_lot15",
    ],
    "Description": [
        "🎯 Target — Sale price (USD)",
        "Number of bedrooms",
        "Number of bathrooms (0.5 = half-bath)",
        "Interior living space (sqft)",
        "Total land lot size (sqft)",
        "Number of floors",
        "Waterfront property flag (0/1)",
        "Quality of views from property (0–4)",
        "Overall condition rating (1–5)",
        "Overall grade assigned by King County (1–13)",
        "Above-ground living area (sqft)",
        "Basement area (sqft)",
        "Year the house was built",
        "Year the house was last renovated (0 = never)",
        "GPS coordinates",
        "Average living area of nearest 15 neighbours (sqft)",
        "Average lot size of nearest 15 neighbours (sqft)",
    ],
}

st.dataframe(pd.DataFrame(features_info), use_container_width=True, hide_index=True)

st.divider()

# ── Engineered Features ────────────────────────────────────────────────────────
st.subheader("⚙️ Engineered Features (added during training)")
eng_info = {
    "Engineered Feature": ["house_age", "renovated_flag", "basement_flag", "years_since_reno", "sale_month", "sale_year"],
    "Formula / Logic": [
        "2015 − yr_built",
        "1 if yr_renovated > 0 else 0",
        "1 if sqft_basement > 0 else 0",
        "2015 − yr_renovated (or house_age if never renovated)",
        "Extracted from sale date",
        "Extracted from sale date",
    ],
    "Rationale": [
        "Captures depreciation effect",
        "Renovated homes command a premium",
        "Basement presence adds value",
        "More recent renovation = higher value",
        "Seasonal pricing effects",
        "Year-over-year price trends",
    ],
}
st.dataframe(pd.DataFrame(eng_info), use_container_width=True, hide_index=True)

st.divider()

# ── Model Summary ──────────────────────────────────────────────────────────────
st.subheader("🤖 Model Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model",      "GradientBoosting")
c2.metric("R² Score",   "0.8894")
c3.metric("MAE",        "~$72,405")
c4.metric("RMSE",       "~$129,316")

st.caption(
    "Model trained on 80% of data (17,290 samples) and evaluated on 20% (4,323 samples). "
    "Log-transform applied to price during training; predictions are reversed with expm1."
)

st.divider()
st.markdown(
    "<div style='text-align:center; color:#888; font-size:13px;'>Built with Python · scikit-learn · Streamlit · Plotly</div>",
    unsafe_allow_html=True,
)
