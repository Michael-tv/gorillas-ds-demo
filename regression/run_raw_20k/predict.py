import os
import joblib
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "mlp":                     os.path.join(_here, "models", "model_mlp.joblib"),
    "decision_tree":           os.path.join(_here, "models", "model_decision_tree.joblib"),
    "decision_tree_overfit":   os.path.join(_here, "models", "model_decision_tree_overfit.joblib"),
    "random_forest":           os.path.join(_here, "models", "model_random_forest.joblib"),
    "linear_regression":       os.path.join(_here, "models", "model_linear_regression.joblib"),
    "ridge":                   os.path.join(_here, "models", "model_ridge.joblib"),
    "lasso":                   os.path.join(_here, "models", "model_lasso.joblib"),
    "polynomial":              os.path.join(_here, "models", "model_polynomial.joblib"),
    "knn":                     os.path.join(_here, "models", "model_knn.joblib"),
    "xgboost":                 os.path.join(_here, "models", "model_xgboost.joblib"),
}


def predict_velocity(elevation, wind_x, mass, radius, Cd, launch_height, landing_height, landing_distance, model_name="mlp"):
    wind_speed = abs(wind_x)
    wind_dir   = 1.0 if wind_x >= 0 else -1.0

    data   = joblib.load(MODELS[model_name])
    model  = data["model"]
    scaler = data.get("scaler")

    X = np.array([[elevation, wind_speed, wind_dir, mass, radius, Cd, launch_height, landing_height, landing_distance]])
    if scaler:
        X = scaler.transform(X)

    return model.predict(X)[0]


if __name__ == "__main__":
    # ── Config ────────────────────────────────────────────────────────────────
    elevation        = 45.0   # degrees
    wind_x           = 5.0    # m/s (positive = tailwind)
    mass             = 0.145  # kg
    radius           = 0.037  # m
    Cd               = 0.47
    launch_height    = 0.0    # m
    landing_height   = 0.0    # m
    landing_distance = 190.0  # m
    model_name       = "mlp"
    # ─────────────────────────────────────────────────────────────────────────

    velocity   = predict_velocity(elevation, wind_x, mass, radius, Cd, launch_height, landing_height, landing_distance, model_name)
    wind_speed = abs(wind_x)
    wind_dir   = "tailwind" if wind_x >= 0 else "headwind"

    print(f"\nInputs  : angle={elevation} deg  wind={wind_speed} m/s @ {wind_dir} deg"
          f"  mass={mass} kg  r={radius} m  Cd={Cd}"
          f"  launch_z={launch_height} m  landing_z={landing_height} m  target={landing_distance} m")
    print(f"Model   : {model_name}")
    print(f"Required velocity : {velocity:.2f} m/s")
