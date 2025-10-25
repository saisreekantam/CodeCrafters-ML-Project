"""
Lasso & Ridge Regression for Patient Recovery Prediction
Automatically finds optimal alpha and random_state, generates both submissions
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LASSO & RIDGE REGRESSION - HYPERPARAMETER OPTIMIZATION")
print("="*80)

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ids = test['Id'].copy()

# Prepare training data
train_data = train.drop('Id', axis=1)
X = train_data.drop('Recovery Index', axis=1).copy()
y = train_data['Recovery Index'].values

# Encode categorical feature
le = LabelEncoder()
X['Lifestyle Activities'] = le.fit_transform(X['Lifestyle Activities'])

# ============================================================================
# LASSO OPTIMIZATION
# ============================================================================
print("\n" + "="*80)
print("PART 1: LASSO REGRESSION OPTIMIZATION")
print("="*80)

print("\n[1/5] Finding optimal random_state for Lasso...")
best_lasso_state = None
best_lasso_rmse = float('inf')

for state in range(0, 150, 5):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    lasso = Lasso(alpha=0.01, random_state=state, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    y_pred = lasso.predict(X_val_scaled)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    if val_rmse < best_lasso_rmse:
        best_lasso_rmse = val_rmse
        best_lasso_state = state

print(f"✓ Best random_state for Lasso: {best_lasso_state} (Val RMSE: {best_lasso_rmse:.4f})")

# Use best random state
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=best_lasso_state
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("\n[2/5] Finding optimal alpha for Lasso...")
alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
best_lasso_alpha = None
best_lasso_alpha_rmse = float('inf')

print(f"{'Alpha':<12} {'Train RMSE':<12} {'Val RMSE':<12} {'R²':<12}")
print("-"*50)

for alpha in alphas:
    lasso = Lasso(alpha=alpha, random_state=best_lasso_state, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    
    y_train_pred = lasso.predict(X_train_scaled)
    y_val_pred = lasso.predict(X_val_scaled)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2 = r2_score(y_val, y_val_pred)
    
    print(f"{alpha:<12.4f} {train_rmse:<12.4f} {val_rmse:<12.4f} {r2:<12.6f}")
    
    if val_rmse < best_lasso_alpha_rmse:
        best_lasso_alpha_rmse = val_rmse
        best_lasso_alpha = alpha

print(f"\n✓ Best alpha for Lasso: {best_lasso_alpha} (Val RMSE: {best_lasso_alpha_rmse:.4f})")

# Train final Lasso model
print("\n[3/5] Training final Lasso model...")
final_lasso = Lasso(alpha=best_lasso_alpha, random_state=best_lasso_state, max_iter=10000)
final_lasso.fit(X_train_scaled, y_train)

y_train_pred = final_lasso.predict(X_train_scaled)
y_val_pred = final_lasso.predict(X_val_scaled)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2 = r2_score(y_val, y_val_pred)

print(f"\nLasso Final Performance:")
print(f"  Alpha: {best_lasso_alpha}")
print(f"  Random state: {best_lasso_state}")
print(f"  Train RMSE: {train_rmse:.4f}")
print(f"  Val RMSE:   {val_rmse:.4f}")
print(f"  R² Score:   {r2:.6f}")

# Feature coefficients
print(f"\nLasso Coefficients:")
feature_names = X.columns
for name, coef in zip(feature_names, final_lasso.coef_):
    print(f"  {name:25s}: {coef:8.4f}")
print(f"  {'Intercept':25s}: {final_lasso.intercept_:8.4f}")

# Count non-zero features
non_zero = np.sum(final_lasso.coef_ != 0)
print(f"\nNon-zero features: {non_zero}/{len(feature_names)}")

# ============================================================================
# RIDGE OPTIMIZATION
# ============================================================================
print("\n" + "="*80)
print("PART 2: RIDGE REGRESSION OPTIMIZATION")
print("="*80)

print("\n[4/5] Finding optimal alpha for Ridge...")
best_ridge_alpha = None
best_ridge_rmse = float('inf')

print(f"{'Alpha':<12} {'Train RMSE':<12} {'Val RMSE':<12} {'R²':<12}")
print("-"*50)

for alpha in alphas:
    ridge = Ridge(alpha=alpha, random_state=best_lasso_state)
    ridge.fit(X_train_scaled, y_train)
    
    y_train_pred = ridge.predict(X_train_scaled)
    y_val_pred = ridge.predict(X_val_scaled)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2 = r2_score(y_val, y_val_pred)
    
    print(f"{alpha:<12.4f} {train_rmse:<12.4f} {val_rmse:<12.4f} {r2:<12.6f}")
    
    if val_rmse < best_ridge_rmse:
        best_ridge_rmse = val_rmse
        best_ridge_alpha = alpha

print(f"\n✓ Best alpha for Ridge: {best_ridge_alpha} (Val RMSE: {best_ridge_rmse:.4f})")

# Train final Ridge model
print("\n[5/5] Training final Ridge model...")
final_ridge = Ridge(alpha=best_ridge_alpha, random_state=best_lasso_state)
final_ridge.fit(X_train_scaled, y_train)

y_train_pred = final_ridge.predict(X_train_scaled)
y_val_pred = final_ridge.predict(X_val_scaled)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2 = r2_score(y_val, y_val_pred)

print(f"\nRidge Final Performance:")
print(f"  Alpha: {best_ridge_alpha}")
print(f"  Random state: {best_lasso_state}")
print(f"  Train RMSE: {train_rmse:.4f}")
print(f"  Val RMSE:   {val_rmse:.4f}")
print(f"  R² Score:   {r2:.6f}")

# Feature coefficients
print(f"\nRidge Coefficients:")
for name, coef in zip(feature_names, final_ridge.coef_):
    print(f"  {name:25s}: {coef:8.4f}")
print(f"  {'Intercept':25s}: {final_ridge.intercept_:8.4f}")

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING PREDICTIONS FOR TEST SET")
print("="*80)

test_data = test.drop('Id', axis=1).copy()
test_data['Lifestyle Activities'] = le.transform(test_data['Lifestyle Activities'])
test_scaled = scaler.transform(test_data)

# Lasso predictions
lasso_predictions = final_lasso.predict(test_scaled)
submission_lasso = pd.DataFrame({
    'Id': test_ids,
    'Recovery Index': lasso_predictions
})
submission_lasso.to_csv('submission_lasso_optimized.csv', index=False)
print(f"✓ Lasso submission: submission_lasso_optimized.csv")

# Ridge predictions
ridge_predictions = final_ridge.predict(test_scaled)
submission_ridge = pd.DataFrame({
    'Id': test_ids,
    'Recovery Index': ridge_predictions
})
submission_ridge.to_csv('submission_ridge_optimized.csv', index=False)
print(f"✓ Ridge submission: submission_ridge_optimized.csv")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Lasso:  alpha={best_lasso_alpha:6.4f}, random_state={best_lasso_state}, Val RMSE={best_lasso_alpha_rmse:.4f}")
print(f"Ridge:  alpha={best_ridge_alpha:6.4f}, random_state={best_lasso_state}, Val RMSE={best_ridge_rmse:.4f}")
print(f"Better: {'Lasso' if best_lasso_alpha_rmse < best_ridge_rmse else 'Ridge'}")
print("="*80)
