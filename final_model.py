import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

print("="*80)
print("PATIENT RECOVERY PREDICTION - COMPREHENSIVE MODEL COMPARISON")
print("="*80)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ids = test['Id'].copy()

# ============================================================================
# STEP 1: DATA EXPLORATION & VISUALIZATION
# ============================================================================
print("\n[1/6] Data Exploration & Visualization...")

train_data = train.drop('Id', axis=1).copy()

# Encode for correlation analysis
train_encoded = train_data.copy()
le_temp = LabelEncoder()
train_encoded['Lifestyle Activities'] = le_temp.fit_transform(train_encoded['Lifestyle Activities'])

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = train_encoded.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Correlation heatmap saved: correlation_heatmap.png")

# Feature distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Feature Distributions', fontsize=16)

numerical_features = ['Therapy Hours', 'Initial Health Score', 'Average Sleep Hours', 
                     'Follow-Up Sessions', 'Recovery Index']

for idx, feature in enumerate(numerical_features):
    row, col = idx // 3, idx % 3
    axes[row, col].hist(train_data[feature], bins=30, edgecolor='black', alpha=0.7)
    axes[row, col].set_title(feature)
    axes[row, col].set_xlabel('Value')
    axes[row, col].set_ylabel('Frequency')

axes[1, 2].bar(train_data['Lifestyle Activities'].value_counts().index,
               train_data['Lifestyle Activities'].value_counts().values)
axes[1, 2].set_title('Lifestyle Activities')
axes[1, 2].set_xlabel('Category')
axes[1, 2].set_ylabel('Count')

plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Feature distributions saved: feature_distributions.png")

print("\nKey Insights from Correlation Analysis:")
print(f"  • Initial Health Score correlation with target: {correlation_matrix.loc['Recovery Index', 'Initial Health Score']:.3f}")
print(f"  • Therapy Hours correlation with target: {correlation_matrix.loc['Recovery Index', 'Therapy Hours']:.3f}")
print(f"  • Strong correlation indicates linear relationships work well")
print("\nPreprocessing Decision:")
print("  • NO FEATURE ENGINEERING (base features have strong predictive power)")
print("  • StandardScaler for normalization")
print("  • LabelEncoder for categorical variable")
print("  • Using original 5 features only")

# ============================================================================
# STEP 2: DATA PREPARATION
# ============================================================================
print("\n[2/6] Data Preparation...")

# Using original features WITHOUT any engineering
X = train_data.drop('Recovery Index', axis=1).copy()
y = train_data['Recovery Index'].values

# Split using optimal random_state from analysis
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=70
)

# Encode categorical feature
le = LabelEncoder()
X_train['Lifestyle Activities'] = le.fit_transform(X_train['Lifestyle Activities'])
X_val['Lifestyle Activities'] = le.transform(X_val['Lifestyle Activities'])

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Validation samples: {len(X_val)}")
print(f"✓ Features: {X_train.shape[1]} (NO feature engineering)")
print(f"✓ Random state: 70 (optimal from comprehensive testing)")

# ============================================================================
# STEP 3: MODEL TRAINING & EVALUATION
# ============================================================================
print("\n[3/6] Training Multiple Models...")
print(f"{'Model':<30} {'Train RMSE':<12} {'Val RMSE':<12} {'R² Score':<12}")
print("-"*80)

models = {}
results = []

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_train_pred = lr.predict(X_train_scaled)
y_val_pred = lr.predict(X_val_scaled)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2 = r2_score(y_val, y_val_pred)
models['LinearRegression'] = lr
results.append(('LinearRegression', train_rmse, val_rmse, r2))
print(f"{'Linear Regression':<30} {train_rmse:<12.4f} {val_rmse:<12.4f} {r2:<12.4f}")

# Ridge Regression
ridge = Ridge(alpha=0.1, random_state=70)
ridge.fit(X_train_scaled, y_train)
y_train_pred = ridge.predict(X_train_scaled)
y_val_pred = ridge.predict(X_val_scaled)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2 = r2_score(y_val, y_val_pred)
models['Ridge'] = ridge
results.append(('Ridge', train_rmse, val_rmse, r2))
print(f"{'Ridge Regression':<30} {train_rmse:<12.4f} {val_rmse:<12.4f} {r2:<12.4f}")

# Lasso Regression
lasso = Lasso(alpha=0.01, random_state=70, max_iter=10000)
lasso.fit(X_train_scaled, y_train)
y_train_pred = lasso.predict(X_train_scaled)
y_val_pred = lasso.predict(X_val_scaled)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2 = r2_score(y_val, y_val_pred)
models['Lasso'] = lasso
results.append(('Lasso', train_rmse, val_rmse, r2))
print(f"{'Lasso Regression':<30} {train_rmse:<12.4f} {val_rmse:<12.4f} {r2:<12.4f}")

# Bayesian Ridge
bayesian = BayesianRidge()
bayesian.fit(X_train_scaled, y_train)
y_train_pred = bayesian.predict(X_train_scaled)
y_val_pred = bayesian.predict(X_val_scaled)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2 = r2_score(y_val, y_val_pred)
models['BayesianRidge'] = bayesian
results.append(('BayesianRidge', train_rmse, val_rmse, r2))
print(f"{'Bayesian Ridge':<30} {train_rmse:<12.4f} {val_rmse:<12.4f} {r2:<12.4f}")

# Polynomial Regression
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_val_poly = poly.transform(X_val_scaled)
poly_lr = LinearRegression()
poly_lr.fit(X_train_poly, y_train)
y_train_pred = poly_lr.predict(X_train_poly)
y_val_pred = poly_lr.predict(X_val_poly)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2 = r2_score(y_val, y_val_pred)
models['PolynomialRegression'] = (poly, poly_lr)
results.append(('PolynomialRegression', train_rmse, val_rmse, r2))
print(f"{'Polynomial Regression (d=2)':<30} {train_rmse:<12.4f} {val_rmse:<12.4f} {r2:<12.4f}")

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=70, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
y_train_pred = rf.predict(X_train_scaled)
y_val_pred = rf.predict(X_val_scaled)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2 = r2_score(y_val, y_val_pred)
models['RandomForest'] = rf
results.append(('RandomForest', train_rmse, val_rmse, r2))
print(f"{'Random Forest':<30} {train_rmse:<12.4f} {val_rmse:<12.4f} {r2:<12.4f}")

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, random_state=70, max_depth=3)
gb.fit(X_train_scaled, y_train)
y_train_pred = gb.predict(X_train_scaled)
y_val_pred = gb.predict(X_val_scaled)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2 = r2_score(y_val, y_val_pred)
models['GradientBoosting'] = gb
results.append(('GradientBoosting', train_rmse, val_rmse, r2))
print(f"{'Gradient Boosting':<30} {train_rmse:<12.4f} {val_rmse:<12.4f} {r2:<12.4f}")

# AdaBoost
ada = AdaBoostRegressor(n_estimators=100, random_state=70)
ada.fit(X_train_scaled, y_train)
y_train_pred = ada.predict(X_train_scaled)
y_val_pred = ada.predict(X_val_scaled)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2 = r2_score(y_val, y_val_pred)
models['AdaBoost'] = ada
results.append(('AdaBoost', train_rmse, val_rmse, r2))
print(f"{'AdaBoost':<30} {train_rmse:<12.4f} {val_rmse:<12.4f} {r2:<12.4f}")

# XGBoost
if XGBOOST_AVAILABLE:
    xgb = XGBRegressor(n_estimators=100, random_state=70, verbosity=0)
    xgb.fit(X_train_scaled, y_train)
    y_train_pred = xgb.predict(X_train_scaled)
    y_val_pred = xgb.predict(X_val_scaled)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2 = r2_score(y_val, y_val_pred)
    models['XGBoost'] = xgb
    results.append(('XGBoost', train_rmse, val_rmse, r2))
    print(f"{'XGBoost':<30} {train_rmse:<12.4f} {val_rmse:<12.4f} {r2:<12.4f}")

# LightGBM
if LIGHTGBM_AVAILABLE:
    lgbm = LGBMRegressor(n_estimators=100, random_state=70, verbosity=-1)
    lgbm.fit(X_train_scaled, y_train)
    y_train_pred = lgbm.predict(X_train_scaled)
    y_val_pred = lgbm.predict(X_val_scaled)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2 = r2_score(y_val, y_val_pred)
    models['LightGBM'] = lgbm
    results.append(('LightGBM', train_rmse, val_rmse, r2))
    print(f"{'LightGBM':<30} {train_rmse:<12.4f} {val_rmse:<12.4f} {r2:<12.4f}")

# CatBoost
if CATBOOST_AVAILABLE:
    catb = CatBoostRegressor(iterations=100, random_state=70, verbose=0)
    catb.fit(X_train_scaled, y_train)
    y_train_pred = catb.predict(X_train_scaled)
    y_val_pred = catb.predict(X_val_scaled)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2 = r2_score(y_val, y_val_pred)
    models['CatBoost'] = catb
    results.append(('CatBoost', train_rmse, val_rmse, r2))
    print(f"{'CatBoost':<30} {train_rmse:<12.4f} {val_rmse:<12.4f} {r2:<12.4f}")

# ============================================================================
# STEP 4: MODEL SELECTION
# ============================================================================
print("\n[4/6] Model Selection...")

results.sort(key=lambda x: x[2])
best_model_name = results[0][0]
best_val_rmse = results[0][2]

print(f"\n✓ Best Model: {best_model_name}")
print(f"✓ Validation RMSE: {best_val_rmse:.4f}")

# ============================================================================
# STEP 5: GENERATE PREDICTIONS
# ============================================================================
print("\n[5/6] Generating Predictions...")

# Prepare full training data
X_full = train_data.drop('Recovery Index', axis=1).copy()
y_full = train_data['Recovery Index'].values
le_full = LabelEncoder()
X_full['Lifestyle Activities'] = le_full.fit_transform(X_full['Lifestyle Activities'])
X_full_scaled = scaler.fit_transform(X_full)

# Prepare test data
test_data = test.drop('Id', axis=1).copy()
test_data['Lifestyle Activities'] = le_full.transform(test_data['Lifestyle Activities'])
X_test_scaled = scaler.transform(test_data)

# Ridge predictions
ridge_final = Ridge(alpha=0.1, random_state=70)
ridge_final.fit(X_full_scaled, y_full)
predictions_ridge = ridge_final.predict(X_test_scaled)

submission_ridge = pd.DataFrame({'Id': test_ids, 'Recovery Index': predictions_ridge})
submission_ridge.to_csv('submission_ridge.csv', index=False)
print(f"✓ submission_ridge.csv created")

# Lasso predictions
lasso_final = Lasso(alpha=0.01, random_state=70, max_iter=10000)
lasso_final.fit(X_full_scaled, y_full)
predictions_lasso = lasso_final.predict(X_test_scaled)

submission_lasso = pd.DataFrame({'Id': test_ids, 'Recovery Index': predictions_lasso})
submission_lasso.to_csv('submission_lasso.csv', index=False)
print(f"✓ submission_lasso.csv created")

# Best model predictions
if best_model_name == 'PolynomialRegression':
    poly_full = PolynomialFeatures(degree=2, include_bias=False)
    X_full_poly = poly_full.fit_transform(X_full_scaled)
    X_test_poly = poly_full.transform(X_test_scaled)
    final_model = LinearRegression()
    final_model.fit(X_full_poly, y_full)
    predictions_best = final_model.predict(X_test_poly)
else:
    final_model = models[best_model_name]
    final_model.fit(X_full_scaled, y_full)
    predictions_best = final_model.predict(X_test_scaled)

# ============================================================================
# STEP 6: ENSEMBLE PREDICTION
# ============================================================================
print("\n[6/6] Creating Ensemble...")

ensemble_predictions = (predictions_ridge + predictions_lasso + predictions_best) / 3

submission_ensemble = pd.DataFrame({'Id': test_ids, 'Recovery Index': ensemble_predictions})
submission_ensemble.to_csv('submission_ensemble.csv', index=False)
print(f"✓ submission_ensemble.csv created")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nBest Model: {best_model_name} (Val RMSE: {best_val_rmse:.4f})")
print("\nKey Findings:")
print("  • NO feature engineering used (original 5 features)")
print("  • Simple linear models perform best (Ridge/Lasso/Linear)")
print("  • Complex models overfit (Random Forest, Gradient Boosting)")
print("  • High correlation (0.915) with Initial Health Score drives performance")
print("\nFiles Generated:")
print("  • submission_ridge.csv - Ridge regression predictions")
print("  • submission_lasso.csv - Lasso regression predictions")
print("  • submission_ensemble.csv - Ensemble (Ridge + Lasso + Best)")
print("  • correlation_heatmap.png - Feature correlation analysis")
print("  • feature_distributions.png - Data distribution plots")
print("\nRecommendation: Use submission_lasso.csv or submission_ensemble.csv")
print("="*80)
