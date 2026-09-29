from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

NAME = "Polynomial Regression"


def fit(X_train, y_train, feature_step, search_cfg):
    """Degree 3 adds significantly more features (165 vs 44) but can overfit.
    Grid, not random search -- only one hyperparameter, two values."""
    pipeline = Pipeline([
        feature_step,
        ("scaler", StandardScaler()),
        ("poly",   PolynomialFeatures(include_bias=False)),
        ("model",  LinearRegression()),
    ])
    param_grid = {"poly__degree": [2, 3]}
    search = GridSearchCV(pipeline, param_grid, cv=search_cfg["cv"],
                          scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)

    best_degree = search.best_params_["poly__degree"]
    n_features  = search.best_estimator_["poly"].n_output_features_
    print(f"\nBest degree : {best_degree}  ({n_features} features after expansion)\n")
    return search.best_estimator_
