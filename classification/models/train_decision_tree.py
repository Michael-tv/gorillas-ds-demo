import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from train_utils import load_data, print_metrics, model_path

N_ITER = 30
CV     = 5

X_train, X_test, y_train, y_test, _ = load_data()

param_dist = {
    "max_depth":         [4, 6, 8, 10, 12, 15, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 4, 8],
}

search = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
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

print("Decision Tree -- test set")
print_metrics(y_test, y_pred, y_prob)

joblib.dump({"model": best}, model_path("model_decision_tree.joblib"))
print("Saved model_decision_tree.joblib")
