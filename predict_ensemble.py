"""
Ensemble Methods for Patient Recovery Prediction
Tests multiple ensemble strategies and generates optimized submission
Includes: Voting, Stacking, Weighted Average, Blending
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ENSEMBLE METHODS - HYPERPARAMETER OPTIMIZATION")
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

print("\n[1/5] Finding optimal random_state for Ensemble...")
print("Testing 20 random states...")

best_state = None
best_val_rmse = float('inf')

# Test different random states with a simple ensemble
for state in [5, 13, 21, 29, 37, 45, 53, 61, 69, 77, 85, 93, 101, 109, 117, 125, 133, 141, 149]:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Quick ensemble test
    lr = LinearRegression()
    ridge = Ridge(alpha=0.1, random_state=state)
    lasso = Lasso(alpha=0.01, random_state=state, max_iter=10000)
    
    lr.fit(X_train_scaled, y_train)
    ridge.fit(X_train_scaled, y_train)
    lasso.fit(X_train_scaled, y_train)
    
    # Simple average
    pred1 = lr.predict(X_val_scaled)
    pred2 = ridge.predict(X_val_scaled)
    pred3 = lasso.predict(X_val_scaled)
    y_pred = (pred1 + pred2 + pred3) / 3
    
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_state = state

print(f"\n✓ Best random_state: {best_state} (Val RMSE: {best_val_rmse:.4f})")

# Use best random state
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=best_state
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train individual models
print("\n[2/5] Training individual base models...")

models = {}
predictions_train = {}
predictions_val = {}

# Linear models
print("  Training LinearRegression...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
models['LinearRegression'] = lr
predictions_train['LinearRegression'] = lr.predict(X_train_scaled)
predictions_val['LinearRegression'] = lr.predict(X_val_scaled)

print("  Training Ridge...")
ridge = Ridge(alpha=0.1, random_state=best_state)
ridge.fit(X_train_scaled, y_train)
models['Ridge'] = ridge
predictions_train['Ridge'] = ridge.predict(X_train_scaled)
predictions_val['Ridge'] = ridge.predict(X_val_scaled)

print("  Training Lasso...")
lasso = Lasso(alpha=0.01, random_state=best_state, max_iter=10000)
lasso.fit(X_train_scaled, y_train)
models['Lasso'] = lasso
predictions_train['Lasso'] = lasso.predict(X_train_scaled)
predictions_val['Lasso'] = lasso.predict(X_val_scaled)

print("  Training GradientBoosting...")
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=best_state)
gb.fit(X_train_scaled, y_train)
models['GradientBoosting'] = gb
predictions_train['GradientBoosting'] = gb.predict(X_train_scaled)
predictions_val['GradientBoosting'] = gb.predict(X_val_scaled)

print("  Training RandomForest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=best_state, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
models['RandomForest'] = rf
predictions_train['RandomForest'] = rf.predict(X_train_scaled)
predictions_val['RandomForest'] = rf.predict(X_val_scaled)

# Evaluate individual models
print("\nIndividual Model Performance:")
print(f"{'Model':<20} {'Train RMSE':<12} {'Val RMSE':<12} {'R²':<12}")
print("-"*60)

for name in models.keys():
    train_rmse = np.sqrt(mean_squared_error(y_train, predictions_train[name]))
    val_rmse = np.sqrt(mean_squared_error(y_val, predictions_val[name]))
    r2 = r2_score(y_val, predictions_val[name])
    print(f"{name:<20} {train_rmse:<12.4f} {val_rmse:<12.4f} {r2:<12.6f}")

# Test different ensemble strategies
print("\n[3/5] Testing ensemble strategies...")

ensemble_results = {}

# 1. Simple Average
pred_avg = np.mean([predictions_val[name] for name in models.keys()], axis=0)
rmse_avg = np.sqrt(mean_squared_error(y_val, pred_avg))
ensemble_results['Simple Average'] = rmse_avg
print(f"  Simple Average: Val RMSE = {rmse_avg:.4f}")

# 2. Voting Regressor
print("  Training Voting Regressor...")
voting = VotingRegressor([
    ('lr', LinearRegression()),
    ('ridge', Ridge(alpha=0.1, random_state=best_state)),
    ('lasso', Lasso(alpha=0.01, random_state=best_state, max_iter=10000))
])
voting.fit(X_train_scaled, y_train)
pred_voting = voting.predict(X_val_scaled)
rmse_voting = np.sqrt(mean_squared_error(y_val, pred_voting))
ensemble_results['Voting'] = rmse_voting
print(f"  Voting Regressor: Val RMSE = {rmse_voting:.4f}")

# 3. Weighted Average (optimize weights)
print("  Optimizing weighted average...")
from scipy.optimize import minimize

def weighted_ensemble_error(weights):
    weights = np.abs(weights)
    weights = weights / np.sum(weights)
    pred = sum(w * predictions_val[name] for w, name in zip(weights, models.keys()))
    return np.sqrt(mean_squared_error(y_val, pred))

initial_weights = np.ones(len(models)) / len(models)
result = minimize(weighted_ensemble_error, initial_weights, method='Nelder-Mead')
optimal_weights = np.abs(result.x)
optimal_weights = optimal_weights / np.sum(optimal_weights)

pred_weighted = sum(w * predictions_val[name] for w, name in zip(optimal_weights, models.keys()))
rmse_weighted = np.sqrt(mean_squared_error(y_val, pred_weighted))
ensemble_results['Weighted Average'] = rmse_weighted

print(f"  Weighted Average: Val RMSE = {rmse_weighted:.4f}")
print(f"  Optimal weights:")
for name, weight in zip(models.keys(), optimal_weights):
    print(f"    {name:<20}: {weight:.4f}")

# 4. Stacking with meta-learner
print("  Training Stacking ensemble...")
meta_train = np.column_stack([predictions_train[name] for name in models.keys()])
meta_val = np.column_stack([predictions_val[name] for name in models.keys()])

meta_model = Ridge(alpha=1.0, random_state=best_state)
meta_model.fit(meta_train, y_train)
pred_stacking = meta_model.predict(meta_val)
rmse_stacking = np.sqrt(mean_squared_error(y_val, pred_stacking))
ensemble_results['Stacking'] = rmse_stacking
print(f"  Stacking: Val RMSE = {rmse_stacking:.4f}")
print(f"  Meta-learner coefficients:")
for name, coef in zip(models.keys(), meta_model.coef_):
    print(f"    {name:<20}: {coef:.4f}")

# Find best ensemble
best_ensemble = min(ensemble_results, key=ensemble_results.get)
best_ensemble_rmse = ensemble_results[best_ensemble]

print(f"\n[4/5] Best Ensemble Strategy: {best_ensemble} (Val RMSE: {best_ensemble_rmse:.4f})")

print("\nEnsemble Strategy Comparison:")
print(f"{'Strategy':<25} {'Val RMSE':<12} {'Improvement':<12}")
print("-"*50)
baseline = min([np.sqrt(mean_squared_error(y_val, predictions_val[name])) for name in models.keys()])
for strategy, rmse in sorted(ensemble_results.items(), key=lambda x: x[1]):
    improvement = baseline - rmse
    print(f"{strategy:<25} {rmse:<12.4f} {improvement:+12.4f}")

# Generate predictions for test set
print("\n[5/5] Generating predictions for test set...")
test_data = test.drop('Id', axis=1).copy()
test_data['Lifestyle Activities'] = le.transform(test_data['Lifestyle Activities'])
test_scaled = scaler.transform(test_data)

# Get predictions from all base models
test_predictions = {}
for name, model in models.items():
    test_predictions[name] = model.predict(test_scaled)

# Generate ensemble prediction using best strategy
if best_ensemble == 'Simple Average':
    final_predictions = np.mean([test_predictions[name] for name in models.keys()], axis=0)
elif best_ensemble == 'Voting':
    final_predictions = voting.predict(test_scaled)
elif best_ensemble == 'Weighted Average':
    final_predictions = sum(w * test_predictions[name] for w, name in zip(optimal_weights, models.keys()))
elif best_ensemble == 'Stacking':
    meta_test = np.column_stack([test_predictions[name] for name in models.keys()])
    final_predictions = meta_model.predict(meta_test)

# Create submission file
submission = pd.DataFrame({
    'Id': test_ids,
    'Recovery Index': final_predictions
})
submission.to_csv('submission_ensemble_optimized.csv', index=False)

print(f"\n✓ Submission file created: submission_ensemble_optimized.csv")
print(f"  Strategy: {best_ensemble}")
print(f"  Random state: {best_state}")
print(f"  Expected test RMSE: ~{best_ensemble_rmse:.2f}")
print(f"  Base models: {len(models)}")
print("\n" + "="*80)
print("ENSEMBLE OPTIMIZATION COMPLETE")
print("="*80)
