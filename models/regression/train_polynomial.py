import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from train_utils import load_data, print_metrics, model_path, FEATURE_STEP

import params
import splitting

CV = params.search_params("regression", "polynomial")["cv"]

# No pre-scaling — the pipeline handles it
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

pipeline = Pipeline([
    FEATURE_STEP,
    ("scaler", StandardScaler()),
    ("poly",   PolynomialFeatures(include_bias=False)),
    ("model",  LinearRegression()),
])

# Degree 3 adds significantly more features (165 vs 44) but can overfit
param_grid = {"poly__degree": [2, 3]}

search = GridSearchCV(
    pipeline, param_grid,
    cv=CV_FOLDS, scoring="neg_mean_squared_error",
    n_jobs=-1, verbose=1,
)
search.fit(X_train, y_train)

best_degree = search.best_params_["poly__degree"]
n_features  = search.best_estimator_["poly"].n_output_features_
print(f"\nBest degree : {best_degree}  ({n_features} features after expansion)\n")

print("Polynomial Regression -- test set")
print_metrics(y_test, search.best_estimator_.predict(X_test))

# Pipeline includes its own scaler — no separate scaler needed
joblib.dump(search.best_estimator_, model_path("model_polynomial.joblib"))
print("Saved model_polynomial.joblib")
