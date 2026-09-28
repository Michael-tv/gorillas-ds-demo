from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NAME = "Linear Regression"


def fit(X_train, y_train, feature_step, search_cfg):
    """No search -- LinearRegression has no hyperparameters to tune."""
    model = Pipeline([feature_step, ("scaler", StandardScaler()), ("model", LinearRegression())])
    model.fit(X_train, y_train)
    return model
