import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from train_utils import load_data, print_metrics, model_path

CV = 5

# No pre-scaling — the pipeline handles it
X_train, X_test, y_train, y_test, _ = load_data(scale=False)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("poly",   PolynomialFeatures(include_bias=False)),
    ("model",  LinearRegression()),
])

# Degree 3 adds significantly more features (165 vs 44) but can overfit
param_grid = {"poly__degree": [2, 3]}

search = GridSearchCV(
    pipeline, param_grid,
    cv=CV, scoring="neg_mean_squared_error",
    n_jobs=-1, verbose=1,
)
search.fit(X_train, y_train)

best_degree = search.best_params_["poly__degree"]
n_features  = search.best_estimator_["poly"].n_output_features_
print(f"\nBest degree : {best_degree}  ({n_features} features after expansion)\n")

print("Polynomial Regression -- test set")
print_metrics(y_test, search.best_estimator_.predict(X_test))

# Pipeline includes its own scaler — no separate scaler needed
joblib.dump({"model": search.best_estimator_}, model_path("model_polynomial.joblib"))
print("Saved model_polynomial.joblib")
