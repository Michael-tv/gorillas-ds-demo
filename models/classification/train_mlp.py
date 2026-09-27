import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from train_utils import load_data, print_metrics, model_path

import params

_search = params.search_params("classification", "mlp")
N_ITER  = _search["n_iter"]
CV      = _search["cv"]

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params.load_params()["test_size"], random_state=42, stratify=y
)

pipeline = Pipeline([("scaler", StandardScaler()),
                      ("model", MLPClassifier(max_iter=500, early_stopping=True, random_state=42))])

param_dist = {
    "model__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64), (64, 64, 32)],
    "model__activation":         ["relu", "tanh"],
    "model__alpha":              [1e-4, 1e-3, 1e-2],
    "model__learning_rate_init": [1e-3, 5e-4],
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=N_ITER, cv=CV, scoring="roc_auc",
    random_state=42, n_jobs=-1, verbose=1,
)
search.fit(X_train, y_train)

print(f"\nBest params : {search.best_params_}")
print(f"Best CV AUC : {search.best_score_:.4f}\n")

best   = search.best_estimator_
y_pred = best.predict(X_test)
y_prob = best.predict_proba(X_test)[:, 1]

print("MLP -- test set")
print_metrics(y_test, y_pred, y_prob)

joblib.dump(best, model_path("model_mlp.joblib"))
print("Saved model_mlp.joblib")
