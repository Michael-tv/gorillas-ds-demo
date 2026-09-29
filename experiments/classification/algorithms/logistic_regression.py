from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NAME = "Logistic Regression"


def fit(X_train, y_train, search_cfg):
    """No search -- class_weight="balanced" reweights the loss by inverse
    class frequency, so the 5-7% hit rate in a Gorillas pool does not let the
    model win by always predicting "miss". A no-op on the 50/50 Python pool,
    which is fine: the setting is correct for both."""
    model = Pipeline([("scaler", StandardScaler()),
                       ("model", LogisticRegression(max_iter=1000, random_state=42,
                                                    class_weight="balanced"))])
    model.fit(X_train, y_train)
    return model
