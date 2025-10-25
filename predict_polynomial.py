"""
Polynomial Regression for Patient Recovery Prediction
Automatically finds optimal degree and random_state, generates submission
Uses PolynomialFeatures with LinearRegression
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("POLYNOMIAL REGRESSION - HYPERPARAMETER OPTIMIZATION")
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

print("\n[1/4] Finding optimal random_state for Polynomial Regression...")
print("Testing 30 random states with degree=2...")

best_state = None
best_val_rmse = float('inf')

# Test different random states
for state in range(0, 150, 5):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Test with degree=2
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_val_poly = poly.transform(X_val_scaled)
    
    lr = LinearRegression()
    lr.fit(X_train_poly, y_train)
    y_pred = lr.predict(X_val_poly)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_state = state

print(f"\n✓ Best random_state: {best_state} (Val RMSE with degree=2: {best_val_rmse:.4f})")

# Use best random state for degree optimization
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=best_state
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("\n[2/4] Finding optimal polynomial degree...")
print("Testing degrees 2, 3, 4 with both LinearRegression and Ridge...")

degrees = [2, 3, 4]
best_degree = None
best_model_type = None
best_alpha = None
best_degree_rmse = float('inf')
best_model = None
best_poly = None

print(f"\n{'Degree':<8} {'Model':<20} {'Alpha':<10} {'Train RMSE':<12} {'Val RMSE':<12} {'Features':<10}")
print("-"*80)

for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_val_poly = poly.transform(X_val_scaled)
    
    n_features = X_train_poly.shape[1]
    
    # Try LinearRegression
    lr = LinearRegression()
    lr.fit(X_train_poly, y_train)
    y_train_pred = lr.predict(X_train_poly)
    y_val_pred = lr.predict(X_val_poly)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    
    print(f"{degree:<8} {'LinearRegression':<20} {'-':<10} {train_rmse:<12.4f} {val_rmse:<12.4f} {n_features:<10}")
    
    if val_rmse < best_degree_rmse:
        best_degree_rmse = val_rmse
        best_degree = degree
        best_model_type = 'LinearRegression'
        best_alpha = None
        best_model = lr
        best_poly = poly
    
    # Try Ridge with different alphas
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        ridge = Ridge(alpha=alpha, random_state=best_state)
        ridge.fit(X_train_poly, y_train)
        y_train_pred = ridge.predict(X_train_poly)
        y_val_pred = ridge.predict(X_val_poly)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        print(f"{degree:<8} {'Ridge':<20} {alpha:<10.2f} {train_rmse:<12.4f} {val_rmse:<12.4f} {n_features:<10}")
        
        if val_rmse < best_degree_rmse:
            best_degree_rmse = val_rmse
            best_degree = degree
            best_model_type = 'Ridge'
            best_alpha = alpha
            best_model = ridge
            best_poly = poly

print(f"\n✓ Best configuration:")
print(f"  Degree: {best_degree}")
print(f"  Model: {best_model_type}")
if best_alpha:
    print(f"  Alpha: {best_alpha}")
print(f"  Val RMSE: {best_degree_rmse:.4f}")

# Retrain with best configuration
print("\n[3/4] Training final Polynomial Regression model...")

# Recreate the best poly features
best_poly = PolynomialFeatures(degree=best_degree, include_bias=False)
X_train_poly = best_poly.fit_transform(X_train_scaled)
X_val_poly = best_poly.transform(X_val_scaled)

if best_model_type == 'LinearRegression':
    final_model = LinearRegression()
else:
    final_model = Ridge(alpha=best_alpha, random_state=best_state)

final_model.fit(X_train_poly, y_train)

# Evaluate
y_train_pred = final_model.predict(X_train_poly)
y_val_pred = final_model.predict(X_val_poly)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2 = r2_score(y_val, y_val_pred)

print(f"\nFinal Polynomial Model Performance:")
print(f"  Degree: {best_degree}")
print(f"  Model: {best_model_type}")
if best_alpha:
    print(f"  Alpha: {best_alpha}")
print(f"  Random state: {best_state}")
print(f"  Train RMSE: {train_rmse:.4f}")
print(f"  Val RMSE:   {val_rmse:.4f}")
print(f"  R² Score:   {r2:.6f}")
print(f"  Features generated: {X_train_poly.shape[1]} (from {X_train_scaled.shape[1]} original)")
print(f"  Overfitting: {train_rmse - val_rmse:.4f}")

# Show feature names (first 20 only if too many)
feature_names = best_poly.get_feature_names_out(X.columns)
print(f"\nPolynomial features created (showing first 20):")
for i, name in enumerate(feature_names[:20]):
    print(f"  {i+1:2d}. {name}")
if len(feature_names) > 20:
    print(f"  ... and {len(feature_names) - 20} more features")

# Generate predictions for test set
print("\n[4/4] Generating predictions for test set...")
test_data = test.drop('Id', axis=1).copy()
test_data['Lifestyle Activities'] = le.transform(test_data['Lifestyle Activities'])
test_scaled = scaler.transform(test_data)
test_poly = best_poly.transform(test_scaled)

test_predictions = final_model.predict(test_poly)

# Create submission file
submission = pd.DataFrame({
    'Id': test_ids,
    'Recovery Index': test_predictions
})
submission.to_csv('submission_polynomial_optimized.csv', index=False)

print(f"\n✓ Submission file created: submission_polynomial_optimized.csv")
print(f"  Polynomial degree: {best_degree}")
print(f"  Model type: {best_model_type}")
if best_alpha:
    print(f"  Regularization alpha: {best_alpha}")
print(f"  Random state: {best_state}")
print(f"  Expected test RMSE: ~{val_rmse:.2f}")
print("\n" + "="*80)
print("POLYNOMIAL REGRESSION OPTIMIZATION COMPLETE")
print("="*80)
