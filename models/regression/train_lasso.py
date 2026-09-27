import joblib
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from train_utils import load_data, print_metrics, model_path, FEATURE_STEP

import params
import splitting

CV       = params.search_params("regression", "lasso")["cv"]
ALPHAS   = [0.001, 0.01, 0.1, 1, 10, 100]
MAX_ITER = 10_000

X, y, groups = load_data()
# Group-aware when the active pool has groups -- a Gorillas pool shares one
# board's wind and skyline across all 32 of its throws, so a random split would
# put the same board on both sides and score against rows the model has
# effectively already seen. Degrades to an ordinary random split when the pool
# has no group structure (the Python pool's ids are unique per row), so the same
# call is correct for both producers. See splitting.py -- AUDIT.md task 36/§5.1.
X_train, X_test, y_train, y_test, groups_train = splitting.split(
    X, y, groups, test_size=params.load_params()["test_size"], random_state=42)

# Group-aware folds too: cross-validating with a random KFold inside a
# group-aware split would leak across folds instead of across the test set.
CV_FOLDS = splitting.cv_for(CV, X_train, y_train, groups_train)

model = Pipeline([FEATURE_STEP, ("scaler", StandardScaler()),
                   ("model", LassoCV(alphas=ALPHAS, cv=CV_FOLDS, max_iter=MAX_ITER))])
model.fit(X_train, y_train)

print(f"Best alpha : {model.named_steps['model'].alpha_}")
print(f"Non-zero coefficients: {(model.named_steps['model'].coef_ != 0).sum()} / {len(model.named_steps['model'].coef_)}\n")
print("Lasso Regression -- test set")
print_metrics(y_test, model.predict(X_test))

joblib.dump(model, model_path("model_lasso.joblib"))
print("Saved model_lasso.joblib")
