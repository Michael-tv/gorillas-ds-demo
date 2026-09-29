from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NAME = "Lasso Regression"
ALPHAS   = [0.001, 0.01, 0.1, 1, 10, 100]
MAX_ITER = 10_000


def fit(X_train, y_train, feature_step, search_cfg):
    """LassoCV does its own internal CV over ALPHAS -- no external search
    wrapper needed."""
    model = Pipeline([feature_step, ("scaler", StandardScaler()),
                       ("model", LassoCV(alphas=ALPHAS, cv=search_cfg["cv"], max_iter=MAX_ITER))])
    model.fit(X_train, y_train)
    print(f"Best alpha : {model.named_steps['model'].alpha_}")
    print(f"Non-zero coefficients: {(model.named_steps['model'].coef_ != 0).sum()} / "
          f"{len(model.named_steps['model'].coef_)}\n")
    return model
