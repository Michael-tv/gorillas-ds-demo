import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from train_utils import load_data, print_metrics, model_path

import params

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params.load_params()["test_size"], random_state=42, stratify=y
)

model = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=42))])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Logistic Regression -- test set")
print_metrics(y_test, y_pred, y_prob)

joblib.dump(model, model_path("model_logistic_regression.joblib"))
print("Saved model_logistic_regression.joblib")
