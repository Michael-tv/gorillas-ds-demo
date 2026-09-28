from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NAME = "Ridge Regression"
ALPHAS = [0.001, 0.01, 0.1, 1, 10, 100, 1000]


def fit(X_train, y_train, feature_step, search_cfg):
    """RidgeCV does its own internal CV over ALPHAS -- no external search
    wrapper needed."""
    model = Pipeline([feature_step, ("scaler", StandardScaler()),
                       ("model", RidgeCV(alphas=ALPHAS, cv=search_cfg["cv"]))])
    model.fit(X_train, y_train)
    print(f"Best alpha : {model.named_steps['model'].alpha_}\n")
    return model
