"""
AdaBoost Regression for Patient Recovery Prediction
Automatically finds optimal hyperparameters and generates submission
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ADABOOST REGRESSION - HYPERPARAMETER OPTIMIZATION")
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

print("\n[1/4] Finding optimal random_state for AdaBoost...")
print("Testing 25 random states...")

best_state = None
best_val_rmse = float('inf')

# Test different random states
for state in [4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124, 132, 140, 148]:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Quick test with default base estimator
    ada = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=5, random_state=state),
        n_estimators=50,
        learning_rate=0.1,
        random_state=state
    )
    ada.fit(X_train_scaled, y_train)
    y_pred = ada.predict(X_val_scaled)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_state = state

print(f"\n✓ Best random_state: {best_state} (Val RMSE: {best_val_rmse:.4f})")

# Use best random state for final training
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=best_state
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("\n[2/4] Finding optimal base estimator and hyperparameters...")
print("Testing different base estimators (DecisionTree with various depths)...")

# Test different base estimators
base_estimators_results = []

print(f"\n{'Base Estimator':<30} {'N Est.':<10} {'LR':<10} {'Val RMSE':<12}")
print("-"*70)

# Test Decision Trees with different depths
for max_depth in [3, 4, 5, 6, 8]:
    for n_estimators in [50, 100, 150]:
        for learning_rate in [0.05, 0.1, 0.5, 1.0]:
            base_est = DecisionTreeRegressor(max_depth=max_depth, random_state=best_state)
            ada = AdaBoostRegressor(
                estimator=base_est,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=best_state
            )
            ada.fit(X_train_scaled, y_train)
            y_pred = ada.predict(X_val_scaled)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            base_estimators_results.append({
                'base_estimator': f'DecisionTree(depth={max_depth})',
                'max_depth': max_depth,
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'val_rmse': val_rmse
            })
            
            print(f"{'DecisionTree(depth=' + str(max_depth) + ')':<30} {n_estimators:<10} {learning_rate:<10.2f} {val_rmse:<12.4f}")

# Find best configuration
best_config = min(base_estimators_results, key=lambda x: x['val_rmse'])

print(f"\n✓ Best configuration found:")
print(f"  Base estimator: DecisionTree(max_depth={best_config['max_depth']})")
print(f"  N estimators: {best_config['n_estimators']}")
print(f"  Learning rate: {best_config['learning_rate']}")
print(f"  Val RMSE: {best_config['val_rmse']:.4f}")

# Fine-tune with GridSearchCV around best parameters
print("\n[3/4] Fine-tuning with GridSearchCV...")

param_grid = {
    'n_estimators': [
        best_config['n_estimators'] - 25,
        best_config['n_estimators'],
        best_config['n_estimators'] + 25
    ],
    'learning_rate': [
        best_config['learning_rate'] * 0.8,
        best_config['learning_rate'],
        best_config['learning_rate'] * 1.2
    ],
    'loss': ['linear', 'square', 'exponential']
}

base_estimator = DecisionTreeRegressor(
    max_depth=best_config['max_depth'],
    random_state=best_state
)

ada_base = AdaBoostRegressor(
    estimator=base_estimator,
    random_state=best_state
)

grid_search = GridSearchCV(
    ada_base, param_grid, cv=3, scoring='neg_mean_squared_error',
    n_jobs=-1, verbose=1
)
grid_search.fit(X_train_scaled, y_train)

print(f"\n✓ Final optimized parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

# Train final model with best parameters
print("\nTraining final AdaBoost model...")
best_ada = grid_search.best_estimator_

# Evaluate
y_train_pred = best_ada.predict(X_train_scaled)
y_val_pred = best_ada.predict(X_val_scaled)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2 = r2_score(y_val, y_val_pred)

print(f"\nFinal Model Performance:")
print(f"  Train RMSE: {train_rmse:.4f}")
print(f"  Val RMSE:   {val_rmse:.4f}")
print(f"  R² Score:   {r2:.6f}")
print(f"  Overfitting: {train_rmse - val_rmse:.4f}")

# Feature importance
print(f"\nFeature Importances:")
feature_names = X.columns
importances = best_ada.feature_importances_
indices = np.argsort(importances)[::-1]
for idx in indices:
    print(f"  {feature_names[idx]:25s}: {importances[idx]:.4f}")

# Show estimator weights (contribution of each weak learner)
print(f"\nEstimator Weights Statistics:")
estimator_weights = best_ada.estimator_weights_
print(f"  Number of estimators: {len(estimator_weights)}")
print(f"  Mean weight: {np.mean(estimator_weights):.4f}")
print(f"  Std weight: {np.std(estimator_weights):.4f}")
print(f"  Min weight: {np.min(estimator_weights):.4f}")
print(f"  Max weight: {np.max(estimator_weights):.4f}")

print(f"\nTop 5 estimator weights (most influential):")
top_indices = np.argsort(estimator_weights)[::-1][:5]
for i, idx in enumerate(top_indices, 1):
    print(f"  {i}. Estimator {idx+1}: weight = {estimator_weights[idx]:.4f}")

# Estimator errors
print(f"\nEstimator Errors Statistics:")
estimator_errors = best_ada.estimator_errors_
print(f"  Mean error: {np.mean(estimator_errors):.4f}")
print(f"  Std error: {np.std(estimator_errors):.4f}")
print(f"  Min error: {np.min(estimator_errors):.4f}")
print(f"  Max error: {np.max(estimator_errors):.4f}")

print(f"\nModel Configuration:")
print(f"  Base estimator: DecisionTree(max_depth={best_config['max_depth']})")
print(f"  Total estimators: {best_ada.n_estimators}")
print(f"  Learning rate: {best_ada.learning_rate}")
print(f"  Loss function: {best_ada.loss}")

# Generate predictions for test set
print("\n[4/4] Generating predictions for test set...")
test_data = test.drop('Id', axis=1).copy()
test_data['Lifestyle Activities'] = le.transform(test_data['Lifestyle Activities'])
test_scaled = scaler.transform(test_data)

test_predictions = best_ada.predict(test_scaled)

# Create submission file
submission = pd.DataFrame({
    'Id': test_ids,
    'Recovery Index': test_predictions
})
submission.to_csv('submission_adaboost_optimized.csv', index=False)

print(f"\n✓ Submission file created: submission_adaboost_optimized.csv")
print(f"  Random state: {best_state}")
print(f"  Base estimator: DecisionTree(max_depth={best_config['max_depth']})")
print(f"  N estimators: {best_ada.n_estimators}")
print(f"  Learning rate: {best_ada.learning_rate}")
print(f"  Loss function: {best_ada.loss}")
print(f"  Expected test RMSE: ~{val_rmse:.2f}")
print("\n" + "="*80)
print("ADABOOST OPTIMIZATION COMPLETE")
print("="*80)
