import joblib
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from train_utils import load_data, print_metrics, model_path, FEATURE_STEP

import params

CV     = params.search_params("regression", "ridge")["cv"]
ALPHAS = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params.load_params()["test_size"], random_state=42)

model = Pipeline([FEATURE_STEP, ("scaler", StandardScaler()), ("model", RidgeCV(alphas=ALPHAS, cv=CV))])
model.fit(X_train, y_train)

print(f"Best alpha : {model.named_steps['model'].alpha_}\n")
print("Ridge Regression -- test set")
print_metrics(y_test, model.predict(X_test))

joblib.dump(model, model_path("model_ridge.joblib"))
print("Saved model_ridge.joblib")
