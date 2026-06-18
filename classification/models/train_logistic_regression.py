import joblib
from sklearn.linear_model import LogisticRegression
from train_utils import load_data, print_metrics, model_path

X_train, X_test, y_train, y_test, scaler = load_data(scale=True)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Logistic Regression -- test set")
print_metrics(y_test, y_pred, y_prob)

joblib.dump({"model": model, "scaler": scaler}, model_path("model_logistic_regression.joblib"))
print("Saved model_logistic_regression.joblib")
