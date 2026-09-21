"""
Page 3 — Interactive House Price Predictor
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import streamlit as st
from utils import load_model, preprocess_input

st.set_page_config(page_title="Predict House Price", page_icon="🏠", layout="wide")

st.title("🏠 Predict House Price")
st.markdown(
    "Fill in the house features below and click **Predict** to get an instant price estimate "
    "powered by the trained **GradientBoosting** model."
)

model, scaler, feature_columns, rmse, model_name = load_model()

st.divider()

# ── Input Form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    st.subheader("🔧 House Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Size & Structure**")
        sqft_living  = st.slider("Living Area (sqft)",       300,   13000, 2000, step=50)
        sqft_lot     = st.slider("Lot Size (sqft)",          500, 1500000, 7500, step=500)
        sqft_above   = st.slider("Above-Ground Area (sqft)", 300,   10000, 1500, step=50)
        sqft_basement= st.slider("Basement Area (sqft)",       0,    5000,    0, step=50)
        sqft_living15= st.slider("Neighbour Avg Living Area (sqft)", 400, 6000, 1800, step=50)
        sqft_lot15   = st.slider("Neighbour Avg Lot Size (sqft)",    650, 871200, 7500, step=500)

    with col2:
        st.markdown("**Rooms & Quality**")
        bedrooms  = st.slider("Bedrooms",       1, 10, 3)
        bathrooms = st.slider("Bathrooms",      1.0, 8.0, 2.0, step=0.25)
        floors    = st.selectbox("Floors", [1.0, 1.5, 2.0, 2.5, 3.0], index=0)
        grade     = st.slider("Grade (1–13)",   1, 13, 7,
                               help="Overall grade given by King County grading system (7 = average)")
        condition = st.slider("Condition (1–5)", 1, 5, 3,
                               help="Condition of the house (1=poor, 5=excellent)")
        view      = st.slider("View Quality (0–4)", 0, 4, 0,
                               help="Number of times the house has been viewed")
        waterfront= st.selectbox("Waterfront Property?", [0, 1],
                                  format_func=lambda x: "Yes" if x else "No")

    with col3:
        st.markdown("**Age & Location**")
        yr_built     = st.slider("Year Built",      1900, 2015, 1990)
        yr_renovated = st.slider("Year Renovated (0 = never)", 0, 2015, 0, step=1)
        lat          = st.number_input("Latitude",  value=47.5,   step=0.001, format="%.4f",
                                       help="King County spans ~47.15–47.78°N")
        long_        = st.number_input("Longitude", value=-122.2, step=0.001, format="%.4f",
                                       help="King County spans ~-122.5 to -121.3°W")
        sale_month   = st.selectbox(
            "Sale Month", list(range(1, 13)),
            format_func=lambda m: [
                "Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"
            ][m - 1],
            index=5,
        )

    st.divider()
    submitted = st.form_submit_button("🔮 Predict Price", use_container_width=True, type="primary")

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    user_input = {
        "bedrooms":       bedrooms,
        "bathrooms":      bathrooms,
        "sqft_living":    sqft_living,
        "sqft_lot":       sqft_lot,
        "floors":         floors,
        "waterfront":     waterfront,
        "view":           view,
        "condition":      condition,
        "grade":          grade,
        "sqft_above":     sqft_above,
        "sqft_basement":  sqft_basement,
        "lat":            lat,
        "long":           long_,
        "sqft_living15":  sqft_living15,
        "sqft_lot15":     sqft_lot15,
        "sale_month":     sale_month,
        "sale_year":      2015,
        "yr_built":       yr_built,
        "yr_renovated":   yr_renovated,
    }

    X_input      = preprocess_input(user_input, scaler, feature_columns)
    pred_log     = model.predict(X_input)[0]
    predicted    = np.expm1(pred_log)
    low_bound    = max(0, predicted - rmse)
    high_bound   = predicted + rmse

    st.divider()
    st.subheader("💲 Prediction Result")

    res_col1, res_col2 = st.columns([2, 1])

    with res_col1:
        st.success(f"### Estimated Price:  **${predicted:,.0f}**")
        st.info(
            f"**Confidence Range:** ${low_bound:,.0f} — ${high_bound:,.0f}  \n"
            f"*(Based on model RMSE of ${rmse:,.0f})*"
        )

    with res_col2:
        st.metric("Predicted Price",  f"${predicted:,.0f}")
        st.metric("Lower Bound (−RMSE)", f"${low_bound:,.0f}")
        st.metric("Upper Bound (+RMSE)", f"${high_bound:,.0f}")

    st.divider()

    # ── Input Summary ──────────────────────────────────────────────────────────
    with st.expander("📋 Input Summary"):
        import pandas as pd
        summary = {
            "Bedrooms": bedrooms,       "Bathrooms": bathrooms,
            "Living Area (sqft)": sqft_living, "Lot Size (sqft)": sqft_lot,
            "Floors": floors,           "Grade": grade,
            "Condition": condition,     "View": view,
            "Waterfront": "Yes" if waterfront else "No",
            "Year Built": yr_built,
            "Year Renovated": yr_renovated if yr_renovated > 0 else "Never",
            "Latitude": lat,            "Longitude": long_,
        }
        st.table(pd.DataFrame(summary.items(), columns=["Feature", "Value"]))
