"""
utils.py — Shared helpers for all Streamlit pages.

Provides:
- load_model()         : cached model, scaler, feature_columns, rmse, model_name
- load_data()          : cached feature-engineered DataFrame
- preprocess_input()   : converts user dict → scaled numpy array for inference
"""

import pathlib
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT    = pathlib.Path(__file__).parent.parent   # house_price_prediction/
_MODELS  = _ROOT / "models"
_DATA    = _ROOT / "Housing.csv"


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model artefacts…")
def load_model():
    """Return (model, scaler, feature_columns, rmse, model_name)."""
    model           = joblib.load(_MODELS / "model.pkl")
    scaler          = joblib.load(_MODELS / "scaler.pkl")
    feature_columns = joblib.load(_MODELS / "feature_columns.pkl")
    rmse            = joblib.load(_MODELS / "rmse.pkl")
    model_name      = joblib.load(_MODELS / "model_name.pkl")
    return model, scaler, feature_columns, rmse, model_name


# ── Data loading & feature engineering ───────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    """Return feature-engineered DataFrame (same transformations as notebook)."""
    df = pd.read_csv(_DATA)
    df["date"]         = pd.to_datetime(df["date"])
    df["sale_month"]   = df["date"].dt.month
    df["sale_year"]    = df["date"].dt.year
    df["house_age"]    = 2015 - df["yr_built"]
    df["renovated_flag"]  = (df["yr_renovated"] > 0).astype(int)
    df["basement_flag"]   = (df["sqft_basement"] > 0).astype(int)
    df["years_since_reno"] = df.apply(
        lambda r: 2015 - r["yr_renovated"] if r["yr_renovated"] > 0 else r["house_age"],
        axis=1,
    )
    df.drop(columns=["id", "date", "zipcode", "yr_built", "yr_renovated"], inplace=True)
    return df


# ── Inference preprocessing ───────────────────────────────────────────────────
def preprocess_input(user_dict: dict, scaler, feature_columns: list) -> np.ndarray:
    """
    Convert a dict of raw user inputs into a scaled 1-row numpy array.

    Engineered features (house_age, renovated_flag, etc.) are computed here
    from the raw inputs so the caller only needs to pass human-readable values.
    """
    yr_built     = user_dict.get("yr_built", 1980)
    yr_renovated = user_dict.get("yr_renovated", 0)

    house_age       = 2015 - yr_built
    renovated_flag  = int(yr_renovated > 0)
    basement_flag   = int(user_dict.get("sqft_basement", 0) > 0)
    years_since_reno = (2015 - yr_renovated) if yr_renovated > 0 else house_age
    sale_month      = user_dict.get("sale_month", 6)
    sale_year       = user_dict.get("sale_year", 2015)

    engineered = {
        "house_age":       house_age,
        "renovated_flag":  renovated_flag,
        "basement_flag":   basement_flag,
        "years_since_reno": years_since_reno,
        "sale_month":      sale_month,
        "sale_year":       sale_year,
    }

    # Merge raw + engineered, drop fields used only for engineering
    row = {**user_dict, **engineered}
    row.pop("yr_built",     None)
    row.pop("yr_renovated", None)

    # Build a DataFrame aligned to training feature order
    df_row = pd.DataFrame([row])[feature_columns]
    return scaler.transform(df_row)
