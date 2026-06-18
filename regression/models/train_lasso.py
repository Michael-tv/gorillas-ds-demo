import joblib
from sklearn.linear_model import LassoCV
from train_utils import load_data, print_metrics, model_path

CV       = 5
ALPHAS   = [0.001, 0.01, 0.1, 1, 10, 100]
MAX_ITER = 10_000

X_train, X_test, y_train, y_test, scaler = load_data(scale=True)

model = LassoCV(alphas=ALPHAS, cv=CV, max_iter=MAX_ITER)
model.fit(X_train, y_train)

print(f"Best alpha : {model.alpha_}")
print(f"Non-zero coefficients: {(model.coef_ != 0).sum()} / {len(model.coef_)}\n")
print("Lasso Regression -- test set")
print_metrics(y_test, model.predict(X_test))

joblib.dump({"model": model, "scaler": scaler}, model_path("model_lasso.joblib"))
print("Saved model_lasso.joblib")
