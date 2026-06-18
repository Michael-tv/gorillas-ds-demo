import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from train_utils import load_data, print_metrics, model_path

X_train, X_test, y_train, y_test, scaler = load_data(scale=True)

param_dist = {
    "n_neighbors": [3, 5, 7, 10, 15, 20],
    "weights":     ["uniform", "distance"],
    "metric":      ["euclidean", "manhattan"],
}

search = RandomizedSearchCV(
    KNeighborsClassifier(),
    param_distributions=param_dist,
    n_iter=12, cv=5, scoring="roc_auc",
    random_state=42, n_jobs=-1, verbose=1,
)
search.fit(X_train, y_train)

print(f"\nBest params : {search.best_params_}")
print(f"Best CV AUC : {search.best_score_:.4f}\n")

best   = search.best_estimator_
y_pred = best.predict(X_test)
y_prob = best.predict_proba(X_test)[:, 1]

print("KNN -- test set")
print_metrics(y_test, y_pred, y_prob)

joblib.dump({"model": best, "scaler": scaler}, model_path("model_knn.joblib"))
print("Saved model_knn.joblib")
