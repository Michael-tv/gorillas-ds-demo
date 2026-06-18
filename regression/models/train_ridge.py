import joblib
from sklearn.linear_model import RidgeCV
from train_utils import load_data, print_metrics, model_path

CV     = 5
ALPHAS = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

X_train, X_test, y_train, y_test, scaler = load_data(scale=True)

model = RidgeCV(alphas=ALPHAS, cv=CV)
model.fit(X_train, y_train)

print(f"Best alpha : {model.alpha_}\n")
print("Ridge Regression -- test set")
print_metrics(y_test, model.predict(X_test))

joblib.dump({"model": model, "scaler": scaler}, model_path("model_ridge.joblib"))
print("Saved model_ridge.joblib")
