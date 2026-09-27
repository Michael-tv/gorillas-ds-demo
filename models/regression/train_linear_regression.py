import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from train_utils import load_data, print_metrics, model_path, FEATURE_STEP

import params
import splitting

CV = params.search_params("regression", "linear_regression")["cv"]

X, y, groups = load_data()
# Group-aware when the active pool has groups -- a Gorillas pool shares one
# board's wind and skyline across all 32 of its throws, so a random split would
# put the same board on both sides and score against rows the model has
# effectively already seen. Degrades to an ordinary random split when the pool
# has no group structure (the Python pool's ids are unique per row), so the same
# call is correct for both producers. See splitting.py -- AUDIT.md task 36/§5.1.
X_train, X_test, y_train, y_test, groups_train = splitting.split(
    X, y, groups, test_size=params.load_params()["test_size"], random_state=42)

# Group-aware folds too: cross-validating with a random KFold inside a
# group-aware split would leak across folds instead of across the test set.
CV_FOLDS = splitting.cv_for(CV, X_train, y_train, groups_train)

model = Pipeline([FEATURE_STEP, ("scaler", StandardScaler()), ("model", LinearRegression())])

cv_rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="neg_mean_squared_error")).mean()
print(f"CV RMSE ({CV}-fold): {cv_rmse:.3f} m\n")

model.fit(X_train, y_train)

print("Linear Regression -- test set")
print_metrics(y_test, model.predict(X_test))

joblib.dump(model, model_path("model_linear_regression.joblib"))
print("Saved model_linear_regression.joblib")
