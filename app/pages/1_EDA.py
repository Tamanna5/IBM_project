"""
Page 1 — Exploratory Data Analysis Dashboard
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data

st.set_page_config(page_title="EDA — House Prices", page_icon="📊", layout="wide")

st.title("📊 Exploratory Data Analysis")
st.markdown(
    "Interactive exploration of the **King County House Sales** dataset "
    "(21,613 sales, 2014–2015)."
)

df = load_data()

# ── Section 1: Dataset Overview ───────────────────────────────────────────────
with st.expander("🗂 Dataset Overview", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",    f"{len(df):,}")
    col2.metric("Features",         f"{df.shape[1] - 1}")
    col3.metric("Min Price",        f"${df['price'].min():,.0f}")
    col4.metric("Max Price",        f"${df['price'].max():,.0f}")
    st.dataframe(df.head(10), use_container_width=True)

st.divider()

# ── Section 2: Price Distribution ─────────────────────────────────────────────
st.subheader("💰 Price Distribution")
tab1, tab2 = st.tabs(["Raw Price", "Log-Transformed Price"])

with tab1:
    fig = px.histogram(
        df, x="price", nbins=100, marginal="box",
        title="House Price Distribution",
        labels={"price": "Price (USD)"},
        color_discrete_sequence=["steelblue"],
    )
    fig.update_layout(bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Skewness: **{df['price'].skew():.2f}** — highly right-skewed (log-transform used for training)")

with tab2:
    log_prices = np.log1p(df["price"])
    fig = px.histogram(
        x=log_prices, nbins=100, marginal="box",
        title="Log-Transformed Price Distribution",
        labels={"x": "log1p(Price)"},
        color_discrete_sequence=["darkorange"],
    )
    fig.update_layout(bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Skewness after log-transform: **{log_prices.skew():.2f}** — near-normal ✓")

st.divider()

# ── Section 3: Correlation Heatmap ────────────────────────────────────────────
st.subheader("🔥 Feature Correlation Heatmap")
numeric_df = df.select_dtypes(include=np.number)
corr = numeric_df.corr().round(2)

fig = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    title="Pearson Correlation Matrix",
    height=600,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("**Top features correlated with price:**")
top_corr = corr["price"].drop("price").sort_values(ascending=False).head(8)
st.dataframe(
    top_corr.reset_index().rename(columns={"index": "Feature", "price": "Correlation with Price"}),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ── Section 4: Box Plots ──────────────────────────────────────────────────────
st.subheader("📦 Price by Categorical Feature")
cat_col = st.selectbox(
    "Select feature:", ["grade", "bedrooms", "condition", "waterfront", "floors", "view"]
)
temp = df[df[cat_col] <= df[cat_col].quantile(0.99)]
fig = px.box(
    temp, x=cat_col, y="price",
    title=f"Price Distribution by {cat_col.capitalize()}",
    labels={"price": "Price (USD)", cat_col: cat_col.capitalize()},
    color=cat_col,
    color_discrete_sequence=px.colors.qualitative.Pastel,
)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Section 5: Scatter Plot ───────────────────────────────────────────────────
st.subheader("🔵 Feature vs Price Scatter")
num_features = [
    c for c in df.select_dtypes(include=np.number).columns
    if c not in ("price",)
]
x_feat = st.selectbox("Select X-axis feature:", num_features, index=num_features.index("sqft_living"))

fig = px.scatter(
    df, x=x_feat, y="price", color="grade",
    opacity=0.4,
    title=f"{x_feat} vs Price (coloured by Grade)",
    labels={"price": "Price (USD)", x_feat: x_feat},
    color_continuous_scale="Viridis",
    height=500,
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Section 6: Geographic Map ─────────────────────────────────────────────────
st.subheader("🗺️ Geographic Price Map — King County")
st.caption("Each dot is a house sale. Colour = price.")

fig = px.scatter_mapbox(
    df, lat="lat", lon="long",
    color="price",
    color_continuous_scale="Plasma",
    size_max=6,
    zoom=9,
    mapbox_style="open-street-map",
    title="King County House Prices by Location",
    hover_data={"price": ":,.0f", "bedrooms": True, "sqft_living": True, "grade": True},
    height=600,
)
fig.update_layout(coloraxis_colorbar_title="Price (USD)")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Section 7: Price by Year Built ────────────────────────────────────────────
st.subheader("📅 Median Price by Year Built")
yr_price = df.groupby("house_age")["price"].median().reset_index()
yr_price["year_built"] = 2015 - yr_price["house_age"]
yr_price = yr_price.sort_values("year_built")

fig = px.line(
    yr_price, x="year_built", y="price",
    title="Median House Price by Year Built",
    labels={"year_built": "Year Built", "price": "Median Price (USD)"},
    markers=True,
)
fig.update_traces(line_color="steelblue", marker_color="darkorange")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Section 8: Price per Sqft by Zipcode ─────────────────────────────────────
st.subheader("🏠 Price per Sqft by Zipcode (Top 20)")
df_zip = df.copy()
df_zip["price_per_sqft"] = df_zip["price"] / df_zip["sqft_living"]

# zipcode was dropped from df_fe but the raw df has it — reload from source for this chart
import pathlib as _pl
_raw = _pl.Path(__file__).parent.parent.parent / "Housing.csv"
if _raw.exists():
    import pandas as _pd2
    _df_raw = _pd2.read_csv(_raw)
    df_zip2 = _df_raw.copy()
    df_zip2["price_per_sqft"] = df_zip2["price"] / df_zip2["sqft_living"]
    zip_stats = df_zip2.groupby("zipcode")["price_per_sqft"].median().sort_values(ascending=False).head(20).reset_index()
    zip_stats.columns = ["zipcode", "median_price_per_sqft"]
    fig = px.bar(
        zip_stats, x="zipcode", y="median_price_per_sqft",
        title="Top 20 Zipcodes by Median Price per Sqft",
        labels={"zipcode": "Zipcode", "median_price_per_sqft": "Median $/Sqft"},
        color="median_price_per_sqft", color_continuous_scale="Reds",
        text_auto=".0f", height=460,
    )
    fig.update_layout(xaxis_type="category", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Housing.csv not found for zipcode chart.")

st.divider()

# ── Section 9: Monthly Seasonality ───────────────────────────────────────────
st.subheader("📅 Monthly Sales Volume & Average Price")
monthly = df.groupby("sale_month").agg(
    sales_count=("price", "count"),
    avg_price=("price", "mean")
).reset_index()
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
monthly["month_name"] = monthly["sale_month"].apply(lambda m: month_names[m-1])

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Bar(x=monthly["month_name"], y=monthly["sales_count"],
                     name="Sales Volume", marker_color="steelblue", yaxis="y1"))
fig.add_trace(go.Scatter(x=monthly["month_name"], y=monthly["avg_price"],
                         name="Avg Price", mode="lines+markers",
                         line=dict(color="darkorange", width=3),
                         marker=dict(size=8), yaxis="y2"))
fig.update_layout(
    title="Monthly Sales Volume & Average Price (Seasonality)",
    yaxis=dict(title="Sales Count", titlefont_color="steelblue"),
    yaxis2=dict(title="Avg Price (USD)", titlefont_color="darkorange",
                overlaying="y", side="right"),
    legend=dict(x=0.01, y=0.99), height=480
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Section 10: Violin Plot — Price by Grade ──────────────────────────────────
st.subheader("🎻 Price Distribution by Grade (Violin)")
df_grade = df[df["grade"].between(4, 12)].copy()
fig = px.violin(
    df_grade, x="grade", y="price", box=True, points=False,
    title="Price Distribution by Grade (Violin Plot)",
    labels={"price": "Price (USD)", "grade": "Grade"},
    color="grade", color_discrete_sequence=px.colors.sequential.Viridis,
    height=520,
)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Section 11: Pair Plot ──────────────────────────────────────────────────────
st.subheader("🔢 Pair Plot — Top 5 Features vs Price")
st.caption("Sampled 2,000 records for performance")
pair_df = df[["sqft_living", "grade", "sqft_above", "bathrooms", "lat", "price"]].sample(2000, random_state=42)
fig = px.scatter_matrix(
    pair_df,
    dimensions=["sqft_living", "grade", "sqft_above", "bathrooms", "lat"],
    color="price",
    color_continuous_scale="Plasma",
    opacity=0.4,
    title="Pair Plot — Top 5 Features coloured by Price",
    height=700,
)
fig.update_traces(diagonal_visible=False, marker_size=3)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Section 12: Renovated vs Non-Renovated ────────────────────────────────────
st.subheader("🏗️ Renovated vs Never Renovated — Price")
df_reno = df.copy()
df_reno["Renovation Status"] = (df_reno["renovated_flag"] == 1).map({True: "Renovated", False: "Never Renovated"})
fig = px.box(
    df_reno, x="Renovation Status", y="price",
    color="Renovation Status",
    title="Price: Renovated vs Never Renovated",
    labels={"price": "Price (USD)"},
    color_discrete_map={"Renovated": "#2563EB", "Never Renovated": "#94A3B8"},
    points="outliers", height=480,
)
med_reno    = df_reno[df_reno["renovated_flag"] == 1]["price"].median()
med_no_reno = df_reno[df_reno["renovated_flag"] == 0]["price"].median()
st.plotly_chart(fig, use_container_width=True)
col_a, col_b, col_c = st.columns(3)
col_a.metric("Renovated Median",       f"${med_reno:,.0f}")
col_b.metric("Never Renovated Median", f"${med_no_reno:,.0f}")
col_c.metric("Renovation Premium",     f"${med_reno - med_no_reno:,.0f}  (+{(med_reno/med_no_reno - 1)*100:.1f}%)")

st.divider()

# ── Section 13: Price Buckets Donut ──────────────────────────────────────────
st.subheader("🎯 Price Segments — Donut Chart")
import numpy as np
bins   = [0, 200_000, 500_000, 1_000_000, float("inf")]
labels = ["Under $200K", "$200K–$500K", "$500K–$1M", "Over $1M"]
df_bucket = df.copy()
df_bucket["price_bucket"] = np.digitize(df_bucket["price"], bins=[200_000, 500_000, 1_000_000])
bucket_map = {0: "Under $200K", 1: "$200K–$500K", 2: "$500K–$1M", 3: "Over $1M"}
df_bucket["price_bucket"] = df_bucket["price_bucket"].map(bucket_map)
bucket_counts = df_bucket["price_bucket"].value_counts().reset_index()
bucket_counts.columns = ["Price Range", "Count"]
fig = px.pie(
    bucket_counts, names="Price Range", values="Count",
    title="House Price Distribution by Segment",
    hole=0.45,
    color_discrete_sequence=px.colors.sequential.Blues_r,
    height=480,
)
fig.update_traces(textinfo="percent+label", pull=[0.03]*4)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Section 14: Basement Premium ─────────────────────────────────────────────
st.subheader("📦 Basement vs No Basement — Price")
df_base = df.copy()
df_base["Basement"] = (df_base["basement_flag"] == 1).map({True: "Has Basement", False: "No Basement"})
fig = px.box(
    df_base, x="Basement", y="price",
    color="Basement",
    title="Price: Homes With vs Without Basement",
    labels={"price": "Price (USD)"},
    color_discrete_map={"Has Basement": "#7C3AED", "No Basement": "#94A3B8"},
    points="outliers", height=450,
)
med_base   = df_base[df_base["basement_flag"] == 1]["price"].median()
med_nobase = df_base[df_base["basement_flag"] == 0]["price"].median()
st.plotly_chart(fig, use_container_width=True)
c1, c2, c3 = st.columns(3)
c1.metric("Has Basement Median",    f"${med_base:,.0f}")
c2.metric("No Basement Median",     f"${med_nobase:,.0f}")
c3.metric("Basement Premium",       f"${med_base - med_nobase:,.0f}  (+{(med_base/med_nobase - 1)*100:.1f}%)")

st.divider()

# ── Section 15: House Age Binned ──────────────────────────────────────────────
st.subheader("🕰️ Median Price by House Age Group")
age_bins   = [0, 10, 20, 40, 60, 80, 200]
age_labels = ["0–10 yrs", "10–20 yrs", "20–40 yrs", "40–60 yrs", "60–80 yrs", "80+ yrs"]
df_age = df.copy()
import pandas as _pd3
df_age["age_group"] = _pd3.cut(df_age["house_age"], bins=age_bins, labels=age_labels)
age_price = df_age.groupby("age_group", observed=True)["price"].median().reset_index()
fig = px.bar(
    age_price, x="age_group", y="price",
    title="Median House Price by House Age Group",
    labels={"age_group": "House Age", "price": "Median Price (USD)"},
    color="price", color_continuous_scale="Purples",
    text_auto=".2s", height=450,
)
fig.update_layout(coloraxis_showscale=False)
st.plotly_chart(fig, use_container_width=True)
