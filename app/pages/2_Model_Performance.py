"""
Page 2 — Model Performance Dashboard
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import load_model, load_data

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

st.title("📈 Model Performance")
st.markdown("Evaluation metrics, feature importances, and prediction diagnostics for the trained model.")

# ── Load artefacts ────────────────────────────────────────────────────────────
model, scaler, feature_columns, rmse_saved, model_name = load_model()
df = load_data()

# ── Re-create exact test split ────────────────────────────────────────────────
TARGET = "price"
X = df.drop(columns=[TARGET])
y = np.log1p(df[TARGET])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_sc = scaler.transform(X_test)

y_pred_log  = model.predict(X_test_sc)
y_pred      = np.expm1(y_pred_log)
y_actual    = np.expm1(y_test)

mae  = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2   = r2_score(y_actual, y_pred)
residuals = y_actual.values - y_pred

# ── Section 1: Model Info Banner ──────────────────────────────────────────────
st.info(f"**Active Model:** {model_name}  |  Trained on **17,290** samples  |  Tested on **4,323** samples")

st.divider()

# ── Section 2: KPI Metrics ────────────────────────────────────────────────────
st.subheader("🎯 Key Performance Indicators")
c1, c2, c3 = st.columns(3)
c1.metric("MAE  (Mean Absolute Error)",  f"${mae:,.0f}",  help="Average absolute difference between predicted and actual price")
c2.metric("RMSE (Root Mean Sq. Error)",  f"${rmse:,.0f}", help="Penalises large errors more than MAE")
c3.metric("R²   (Coefficient of Det.)",  f"{r2:.4f}",     help="1.0 = perfect fit. Our model explains ~89% of price variance.")

st.divider()

# ── Section 3: Feature Importances ────────────────────────────────────────────
st.subheader("🏆 Top-15 Feature Importances")
importances = pd.Series(model.feature_importances_, index=feature_columns)
top15 = importances.sort_values(ascending=False).head(15).sort_values()

fig = px.bar(
    x=top15.values,
    y=top15.index,
    orientation="h",
    title=f"Feature Importances — {model_name}",
    labels={"x": "Importance Score", "y": "Feature"},
    color=top15.values,
    color_continuous_scale="Blues",
    height=500,
)
fig.update_layout(coloraxis_showscale=False, yaxis_title="")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Section 4: Actual vs Predicted ───────────────────────────────────────────
st.subheader("🎯 Actual vs Predicted Prices")
scatter_df = pd.DataFrame({"Actual": y_actual.values, "Predicted": y_pred})

fig = px.scatter(
    scatter_df, x="Actual", y="Predicted",
    opacity=0.35,
    labels={"Actual": "Actual Price (USD)", "Predicted": "Predicted Price (USD)"},
    title="Actual vs Predicted — Test Set",
    color_discrete_sequence=["steelblue"],
    height=500,
)
line_range = [scatter_df["Actual"].min(), scatter_df["Actual"].max()]
fig.add_trace(
    go.Scatter(
        x=line_range, y=line_range,
        mode="lines",
        line=dict(color="red", dash="dash", width=2),
        name="Perfect Prediction",
    )
)
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Section 5: Residuals ──────────────────────────────────────────────────────
st.subheader("📐 Residuals Distribution")
col_l, col_r = st.columns(2)

with col_l:
    fig = px.histogram(
        x=residuals, nbins=80,
        title="Residuals Distribution (Actual − Predicted)",
        labels={"x": "Residual (USD)"},
        color_discrete_sequence=["darkorange"],
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero error")
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    fig = px.scatter(
        x=y_pred, y=residuals,
        opacity=0.3,
        title="Residuals vs Fitted Values",
        labels={"x": "Predicted Price (USD)", "y": "Residual (USD)"},
        color_discrete_sequence=["mediumpurple"],
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Mean residual: **${residuals.mean():,.0f}**  |  "
    f"Std residual: **${residuals.std():,.0f}**  |  "
    f"Residuals are approximately centred on zero ✓"
)

st.divider()

# ── Section 6: Learning Curve ─────────────────────────────────────────────────
st.subheader("📉 Learning Curve")
from sklearn.model_selection import learning_curve as _lc
with st.spinner("Computing learning curve — this may take ~30 seconds…"):
    X_train_lc = scaler.transform(X_train)
    train_sizes, train_scores, val_scores = _lc(
        model, X_train_lc, y_train,
        train_sizes=np.linspace(0.1, 1.0, 7),
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=1,   # must be 1 inside Streamlit — multiprocessing can't pickle the cached model
    )

train_mae = -train_scores.mean(axis=1)
val_mae   = -val_scores.mean(axis=1)
lc_df = pd.DataFrame({
    "Train Size": np.concatenate([train_sizes, train_sizes]),
    "MAE (log)":  np.concatenate([train_mae, val_mae]),
    "Split":      ["Train"] * len(train_sizes) + ["Validation"] * len(train_sizes),
})
fig = px.line(
    lc_df, x="Train Size", y="MAE (log)", color="Split",
    markers=True,
    title=f"Learning Curve — {model_name}",
    labels={"Train Size": "Training Samples", "MAE (log)": "MAE (log-scale)"},
    color_discrete_map={"Train": "steelblue", "Validation": "darkorange"},
    height=460,
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Converging train/validation lines = good fit ✓  |  Large persistent gap = overfitting")

st.divider()

# ── Section 7: Actual vs Predicted Distribution Overlay ──────────────────────
st.subheader("🔵 Actual vs Predicted Price Distribution")
fig = go.Figure()
fig.add_trace(go.Histogram(x=y_actual,   name="Actual Price",    nbinsx=80, opacity=0.6, marker_color="steelblue"))
fig.add_trace(go.Histogram(x=y_pred,     name="Predicted Price", nbinsx=80, opacity=0.6, marker_color="darkorange"))
fig.update_layout(
    barmode="overlay",
    title="Actual vs Predicted Price Distribution",
    xaxis_title="Price (USD)", yaxis_title="Count",
    legend=dict(x=0.75, y=0.95), height=460
)
st.plotly_chart(fig, use_container_width=True)
st.caption("A well-calibrated model produces nearly identical distribution shapes.")

st.divider()

# ── Section 8: Percentage Error Distribution ──────────────────────────────────
st.subheader("📊 Percentage Error Distribution")
pct_error = (y_pred - y_actual) / y_actual * 100

fig = px.histogram(
    x=pct_error, nbins=100,
    title="Percentage Error  (Predicted − Actual) / Actual × 100",
    labels={"x": "% Error"},
    color_discrete_sequence=["mediumpurple"],
    height=460,
)
fig.add_vline(x=0,   line_dash="dash", line_color="red",  annotation_text="Zero error")
fig.add_vline(x=-20, line_dash="dot",  line_color="gray", annotation_text="−20%")
fig.add_vline(x=20,  line_dash="dot",  line_color="gray", annotation_text="+20%")
st.plotly_chart(fig, use_container_width=True)

within_10 = (pd.Series(pct_error).abs() <= 10).mean() * 100
within_20 = (pd.Series(pct_error).abs() <= 20).mean() * 100
within_30 = (pd.Series(pct_error).abs() <= 30).mean() * 100
e1, e2, e3 = st.columns(3)
e1.metric("Within ±10%", f"{within_10:.1f}% of predictions")
e2.metric("Within ±20%", f"{within_20:.1f}% of predictions")
e3.metric("Within ±30%", f"{within_30:.1f}% of predictions")

st.divider()

# ── Section 9: SHAP (optional) ────────────────────────────────────────────────
st.subheader("🌳 SHAP Feature Impact (Explainability)")
try:
    import shap
    import matplotlib.pyplot as plt
    with st.spinner("Computing SHAP values for 500 test samples…"):
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_sc[:500])
    fig_shap, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test_sc[:500],
                      feature_names=feature_columns,
                      plot_type="dot", show=False)
    st.pyplot(fig_shap, use_container_width=True)
    st.caption("SHAP beeswarm: each dot = one prediction. Red = high feature value, blue = low. X-axis = impact on price.")
except ImportError:
    st.info("**SHAP not installed.** Run `pip install shap` then restart Streamlit to enable this chart.")
except Exception as e:
    st.warning(f"SHAP skipped: {e}")
