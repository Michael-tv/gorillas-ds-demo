import itertools
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from train_utils import model_path, FEATURES, TARGET, DATA, save_metrics

# Load entire dataset — no train/test split
df = pd.read_csv(DATA)
X  = df[FEATURES].values
y  = df[TARGET].values

param_grid = {
    "max_depth":         [8, 10, 12, 15, 20, 25, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 4, 8],
}

# Evaluate every param combination on training data — no validation whatsoever
best_rmse, best_params, best_model = float("inf"), None, None
keys   = list(param_grid.keys())
combos = list(itertools.product(*param_grid.values()))

for values in combos:
    params = dict(zip(keys, values))
    m = DecisionTreeRegressor(**params, random_state=42)
    m.fit(X, y)
    rmse = np.sqrt(np.mean((y - m.predict(X)) ** 2))
    if rmse < best_rmse:
        best_rmse, best_params, best_model = rmse, params, m

print(f"Best params (by train RMSE): {best_params}")
print(f"Best train RMSE            : {best_rmse:.3f} m/s  ({len(combos)} combos tried)\n")

y_pred = best_model.predict(X)
mae    = mean_absolute_error(y, y_pred)
mse    = np.mean((y - y_pred) ** 2)
rmse   = np.sqrt(mse)

print(f"Decision Tree overfit (n={len(X)}, scored on training data only)")
print(f"  Train MAE  : {mae:.3f} m/s")
print(f"  Train MSE  : {mse:.3f} m²/s²")
print(f"  Train RMSE : {rmse:.3f} m/s")
print("  (no held-out test set — worst-case overfitting)")

joblib.dump({"model": best_model}, model_path("model_decision_tree_overfit.joblib"))
print("Saved model_decision_tree_overfit.joblib")
save_metrics("decision_tree_overfit", mae=mae, mse=mse, rmse=rmse)
