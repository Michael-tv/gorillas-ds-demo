import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from train_utils import load_data, print_metrics, model_path, FEATURE_STEP

import params

CV = params.search_params("regression", "linear_regression")["cv"]

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params.load_params()["test_size"], random_state=42)

model = Pipeline([FEATURE_STEP, ("scaler", StandardScaler()), ("model", LinearRegression())])

cv_rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=CV, scoring="neg_mean_squared_error")).mean()
print(f"CV RMSE ({CV}-fold): {cv_rmse:.3f} m\n")

model.fit(X_train, y_train)

print("Linear Regression -- test set")
print_metrics(y_test, model.predict(X_test))

joblib.dump(model, model_path("model_linear_regression.joblib"))
print("Saved model_linear_regression.joblib")
