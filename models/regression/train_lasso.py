import joblib
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from train_utils import load_data, print_metrics, model_path, FEATURE_STEP

import params

CV       = params.search_params("regression", "lasso")["cv"]
ALPHAS   = [0.001, 0.01, 0.1, 1, 10, 100]
MAX_ITER = 10_000

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params.load_params()["test_size"], random_state=42)

model = Pipeline([FEATURE_STEP, ("scaler", StandardScaler()),
                   ("model", LassoCV(alphas=ALPHAS, cv=CV, max_iter=MAX_ITER))])
model.fit(X_train, y_train)

print(f"Best alpha : {model.named_steps['model'].alpha_}")
print(f"Non-zero coefficients: {(model.named_steps['model'].coef_ != 0).sum()} / {len(model.named_steps['model'].coef_)}\n")
print("Lasso Regression -- test set")
print_metrics(y_test, model.predict(X_test))

joblib.dump(model, model_path("model_lasso.joblib"))
print("Saved model_lasso.joblib")
