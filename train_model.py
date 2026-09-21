"""
Train the house price model and save artefacts to models/
Run from: house_price_prediction/
"""
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ── Load data ─────────────────────────────────────────────────────────────────
DATA_PATH = 'Housing.csv'
df = pd.read_csv(DATA_PATH)
print(f'Loaded dataset: {df.shape}')

# ── Feature Engineering ───────────────────────────────────────────────────────
df['date'] = pd.to_datetime(df['date'])
df['sale_month'] = df['date'].dt.month
df['sale_year']  = df['date'].dt.year
df['house_age']       = 2015 - df['yr_built']
df['renovated_flag']  = (df['yr_renovated'] > 0).astype(int)
df['basement_flag']   = (df['sqft_basement'] > 0).astype(int)
df['years_since_reno'] = df.apply(
    lambda r: 2015 - r['yr_renovated'] if r['yr_renovated'] > 0 else r['house_age'], axis=1
)
df.drop(columns=['id', 'date', 'zipcode', 'yr_built', 'yr_renovated'], inplace=True)

TARGET = 'price'
X = df.drop(columns=[TARGET])
y = np.log1p(df[TARGET])
FEATURE_COLUMNS = list(X.columns)
print(f'Features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}')

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Train: {X_train.shape} | Test: {X_test.shape}')

# ── Scale ─────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Train Models ──────────────────────────────────────────────────────────────
print('Training RandomForest...')
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_sc, y_train)

print('Training GradientBoosting...')
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
gb.fit(X_train_sc, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
def eval_model(name, model):
    y_pred  = np.expm1(model.predict(X_test_sc))
    y_true  = np.expm1(y_test)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f'  {name}: MAE=${mae:,.0f}  RMSE=${rmse:,.0f}  R²={r2:.4f}')
    return r2, rmse

print('\n=== Model Results ===')
rf_r2, rf_rmse = eval_model('RandomForest',     rf)
gb_r2, gb_rmse = eval_model('GradientBoosting', gb)

# ── Pick best & save ──────────────────────────────────────────────────────────
if rf_r2 >= gb_r2:
    best_model, best_name, best_rmse = rf, 'RandomForest', rf_rmse
else:
    best_model, best_name, best_rmse = gb, 'GradientBoosting', gb_rmse

print(f'\nBest model: {best_name}')

os.makedirs('models', exist_ok=True)
joblib.dump(best_model,     'models/model.pkl')
joblib.dump(scaler,         'models/scaler.pkl')
joblib.dump(FEATURE_COLUMNS,'models/feature_columns.pkl')
joblib.dump(best_rmse,      'models/rmse.pkl')
joblib.dump(best_name,      'models/model_name.pkl')

print('Saved: models/model.pkl, scaler.pkl, feature_columns.pkl, rmse.pkl, model_name.pkl')
print('DONE ✓')
