import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from train_utils import DATA, TARGET, model_path, print_metrics

CV = 5

df = pd.read_csv(DATA)
df["sqrt_landing_distance_m"] = np.sqrt(df["landing_distance_m"])

FEATURES = ["launch_angle_deg", "wind_x_ms", "drag_param",
            "height_diff_m", "landing_distance_m", "sqrt_landing_distance_m"]

X = df[FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"Loaded {len(X_train)} train / {len(X_test)} test samples\n")
print(f"Features: {FEATURES}\n")

model = LinearRegression()

cv_rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=CV, scoring="neg_mean_squared_error")).mean()
print(f"CV RMSE ({CV}-fold): {cv_rmse:.3f} m\n")

model.fit(X_train, y_train)

print("Linear Regression (+ sqrt feature) -- test set")
print_metrics(y_test, model.predict(X_test))

joblib.dump({"model": model, "scaler": scaler, "features": FEATURES},
            model_path("model_linear_regression_sqrt.joblib"))
print("Saved model_linear_regression_sqrt.joblib")
