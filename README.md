# House Price Prediction

**Author:** Kalariya Tamanna  
**Project:** House Price Prediction using Machine Learning  
**Dataset:** [King County House Sales — Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)

---

## Project Description

This is an end-to-end machine learning project that predicts house sale prices in King County, Washington (USA) using the King County House Sales dataset (21,613 records, 2014–2015).

The project covers the full ML pipeline:
- **Exploratory Data Analysis (EDA)** — distributions, correlations, geographic visualisation
- **Feature Engineering** — derived features from raw columns (house age, renovation flag, etc.)
- **Model Training** — RandomForestRegressor and GradientBoostingRegressor compared
- **Model Evaluation** — MAE, RMSE, R² metrics with feature importance and residual diagnostics
- **Interactive Dashboard** — Streamlit multi-page app for EDA, performance metrics, and live price prediction

**Best Model: GradientBoostingRegressor — R² = 0.889, MAE = $72,405, RMSE = $129,316**

---

## Dataset

| Property | Value |
|---|---|
| Name | King County House Sales |
| Source | [Kaggle — harlfoxem/housesalesprediction](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) |
| File | `Housing.csv` |
| Rows | 21,613 |
| Columns | 21 |
| Target | `price` (sale price in USD) |
| Coverage | King County, WA — May 2014 to May 2015 |

---

## Technologies Used

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data Analysis | pandas, numpy |
| Visualisation (Notebook) | matplotlib, seaborn |
| Visualisation (App) | Plotly Express |
| ML Models | scikit-learn (RandomForestRegressor, GradientBoostingRegressor) |
| Model Persistence | joblib |
| Notebook | Jupyter Notebook (.ipynb) |
| Frontend / Dashboard | Streamlit |
| Geographic Map | Plotly Mapbox (open-street-map) |

---

## Project Structure

```
house_price_prediction/
├── Housing.csv                               ← Dataset
├── KalariyaTamanna_HousePricePrediction.ipynb  ← Jupyter Notebook (EDA + Training)
├── KalariyaTamanna_ProjectReport.docx          ← Full project report
├── requirements.txt                            ← Python dependencies
├── README.md                                   ← This file
├── train_model.py                              ← Standalone training script
├── models/
│   ├── model.pkl                               ← Saved best model
│   ├── scaler.pkl                              ← StandardScaler
│   ├── feature_columns.pkl                     ← Feature list
│   ├── rmse.pkl                                ← Test RMSE value
│   └── model_name.pkl                          ← Model name string
└── app/
    ├── streamlit_app.py                        ← Home page (Streamlit entry point)
    ├── utils.py                                ← Shared helpers (load model, preprocess)
    └── pages/
        ├── 1_EDA.py                            ← EDA Dashboard page
        ├── 2_Model_Performance.py              ← Model metrics & diagnostics page
        └── 3_Predict.py                        ← Interactive price predictor page
```

---

## Setup & Run Instructions

### Prerequisites
- Python 3.11+
- macOS / Linux / Windows

### Step 1 — Install dependencies

> **macOS users:** Run this first to avoid a build error with `argon2-cffi-bindings`:
> ```bash
> pip3 install argon2-cffi-bindings --only-binary=:all:
> ```

Then install all project dependencies:
```bash
cd house_price_prediction
pip3 install -r requirements.txt
```

### Step 2 — Train the model

This generates all artefacts in the `models/` folder:
```bash
python3 train_model.py
```

Expected output:
```
RandomForest:      MAE=$72,431  RMSE=$138,095  R²=0.8739
GradientBoosting:  MAE=$72,405  RMSE=$129,316  R²=0.8894
Best model: GradientBoosting
Saved: models/model.pkl, scaler.pkl, feature_columns.pkl, rmse.pkl, model_name.pkl
```

### Step 3 — Open the Jupyter Notebook

For full data analytics, EDA, and model training walkthrough:
```bash
jupyter notebook KalariyaTamanna_HousePricePrediction.ipynb
```

### Step 4 — Launch the Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

The app opens at **http://localhost:8501** with 4 pages:

| Page | Description |
|---|---|
| 🏡 Home | Project overview, dataset summary, feature descriptions |
| 📊 EDA | Price distributions, heatmap, geo-map, box plots, scatter |
| 📈 Model Performance | MAE/RMSE/R² cards, feature importances, residuals |
| 🏠 Predict | Live price prediction via sliders and input widgets |

---

## Model Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| RandomForestRegressor | $72,431 | $138,095 | 0.8739 |
| **GradientBoostingRegressor** ✅ | **$72,405** | **$129,316** | **0.8894** |

- **Train set:** 17,290 samples (80%)
- **Test set:** 4,323 samples (20%)
- **Target transform:** `log1p(price)` during training → reversed with `expm1` for display

---

## Key Design Decisions

- **Log-transform `price`** — Reduces right-skew from 4.02 to 0.4 for better model fitting
- **Drop `zipcode`, keep `lat`/`long`** — Continuous spatial signal is richer than 70+ zip categories
- **Engineered features** — `house_age`, `renovated_flag`, `basement_flag`, `years_since_reno`, `sale_month`
- **Consistent split** — Same `random_state=42`, 80/20 split in both notebook and Streamlit app
- **sklearn version alignment** — `model.pkl` must be saved with the same sklearn version that runs `streamlit`

---

## Submission Files

| File | Format | Description |
|---|---|---|
| `KalariyaTamanna_HousePricePrediction.ipynb` | `.ipynb` | Complete project code — EDA, feature engineering, model training |
| `requirements.txt` | `.txt` | All Python library dependencies |
| `KalariyaTamanna_ProjectReport.docx` | `.docx` | Full project documentation and report |
| `README.md` | `.md` | This file — project overview and setup instructions |
