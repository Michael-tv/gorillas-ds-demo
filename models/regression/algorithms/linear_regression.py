import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NAME = "Linear Regression"


def fit(X_train, y_train, feature_step, search_cfg):
    """No search -- LinearRegression has no hyperparameters to tune. CV here
    is purely a reported diagnostic (cross_val_score), not a search."""
    model = Pipeline([feature_step, ("scaler", StandardScaler()), ("model", LinearRegression())])
    cv_rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=search_cfg["cv"],
                                       scoring="neg_mean_squared_error")).mean()
    print(f"CV RMSE: {cv_rmse:.3f} m\n")
    model.fit(X_train, y_train)
    return model
