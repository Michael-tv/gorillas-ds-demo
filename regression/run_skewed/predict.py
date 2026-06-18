import math
import os
import joblib
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "mlp":               os.path.join(_here, "models", "model_mlp.joblib"),
    "decision_tree":     os.path.join(_here, "models", "model_decision_tree.joblib"),
    "random_forest":     os.path.join(_here, "models", "model_random_forest.joblib"),
    "linear_regression": os.path.join(_here, "models", "model_linear_regression.joblib"),
    "ridge":             os.path.join(_here, "models", "model_ridge.joblib"),
    "lasso":             os.path.join(_here, "models", "model_lasso.joblib"),
    "polynomial":        os.path.join(_here, "models", "model_polynomial.joblib"),
    "knn":               os.path.join(_here, "models", "model_knn.joblib"),
    "xgboost":           os.path.join(_here, "models", "model_xgboost.joblib"),
    "skewed":            os.path.join(_here, "models", "model_skewed.joblib"),
    "balanced":          os.path.join(_here, "models", "model_balanced.joblib"),
}

RHO = 1.225  # kg/m³


def predict_velocity(elevation, wind_x, mass, radius, Cd, launch_height, landing_height, landing_distance, model_name="mlp"):
    drag_param  = (0.5 * RHO * Cd * math.pi * radius ** 2) / mass
    height_diff = landing_height - launch_height

    data   = joblib.load(MODELS[model_name])
    model  = data["model"]
    scaler = data.get("scaler")

    X = np.array([[elevation, wind_x, drag_param, height_diff, landing_distance]])
    if scaler:
        X = scaler.transform(X)

    return model.predict(X)[0]


if __name__ == "__main__":
    # ── Config ────────────────────────────────────────────────────────────────
    elevation        = 20.0   # deg  (within the skewed training range: 5-30)
    wind_x           = 5.0    # m/s
    mass             = 0.145  # kg
    radius           = 0.037  # m
    Cd               = 0.47
    launch_height    = 0.0    # m
    landing_height   = 0.0    # m
    landing_distance = 90.0   # m
    model_name       = "mlp"
    # ─────────────────────────────────────────────────────────────────────────

    velocity   = predict_velocity(elevation, wind_x, mass, radius, Cd, launch_height, landing_height, landing_distance, model_name)
    drag_param = (0.5 * RHO * Cd * math.pi * radius ** 2) / mass

    print(f"\nInputs  : angle={elevation} deg  wind_x={wind_x} m/s"
          f"  drag_param={drag_param:.5f}  launch_z={launch_height} m  landing_z={landing_height} m"
          f"  target={landing_distance} m")
    print(f"Model   : {model_name}")
    print(f"Required velocity : {velocity:.2f} m/s")
