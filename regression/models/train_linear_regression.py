import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from train_utils import load_data, print_metrics, model_path

CV = 5

X_train, X_test, y_train, y_test, scaler = load_data(scale=True)

model = LinearRegression()

cv_rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=CV, scoring="neg_mean_squared_error")).mean()
print(f"CV RMSE ({CV}-fold): {cv_rmse:.3f} m\n")

model.fit(X_train, y_train)

print("Linear Regression -- test set")
print_metrics(y_test, model.predict(X_test))

joblib.dump({"model": model, "scaler": scaler}, model_path("model_linear_regression.joblib"))
print("Saved model_linear_regression.joblib")
