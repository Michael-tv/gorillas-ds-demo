import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from train_utils import load_data, print_metrics, model_path

N_ITER = 15
CV     = 5

X_train, X_test, y_train, y_test, scaler = load_data(scale=True)

param_dist = {
    "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64), (64, 64, 32)],
    "activation":         ["relu", "tanh"],
    "alpha":              [1e-4, 1e-3, 1e-2],
    "learning_rate_init": [1e-3, 5e-4],
}

search = RandomizedSearchCV(
    MLPClassifier(max_iter=500, early_stopping=True, random_state=42),
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

joblib.dump({"model": best, "scaler": scaler}, model_path("model_mlp.joblib"))
print("Saved model_mlp.joblib")
