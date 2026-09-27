import math
import os
import joblib
import numpy as np

_here      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.join(_here, "..", "..", "..")
MODELS_DIR = os.path.join(REPO_ROOT, "experiments", "classification", "run", "models")

MODELS = {
    "logistic_regression": os.path.join(MODELS_DIR, "model_logistic_regression.joblib"),
    "decision_tree":       os.path.join(MODELS_DIR, "model_decision_tree.joblib"),
    "random_forest":       os.path.join(MODELS_DIR, "model_random_forest.joblib"),
    "knn":                 os.path.join(MODELS_DIR, "model_knn.joblib"),
    "mlp":                 os.path.join(MODELS_DIR, "model_mlp.joblib"),
    "xgboost":             os.path.join(MODELS_DIR, "model_xgboost.joblib"),
}

RHO = 1.225  # kg/m³


def predict_hit(initial_velocity, elevation, wind_x, mass, radius, Cd,
                ground_z, target_distance, model_name="random_forest"):
    """Return (hit: bool, hit_probability: float)."""
    drag_param = (0.5 * RHO * Cd * math.pi * radius**2) / mass

    model = joblib.load(MODELS[model_name])
    X = np.array([[initial_velocity, elevation, wind_x, drag_param,
                   ground_z, target_distance]])
    label = int(model.predict(X)[0])
    prob  = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else float(label)
    return bool(label), prob


if __name__ == "__main__":
    # ── Config ────────────────────────────────────────────────────────────────
    initial_velocity = 45.0   # m/s
    elevation        = 45.0   # degrees
    wind_x           = 5.0    # m/s (positive = tailwind)
    mass             = 0.145  # kg
    radius           = 0.037  # m
    Cd               = 0.47
    ground_z         = 0.0    # m
    target_distance  = 190.0  # m
    model_name       = "random_forest"
    # ─────────────────────────────────────────────────────────────────────────

    hit, prob  = predict_hit(initial_velocity, elevation, wind_x,
                             mass, radius, Cd, ground_z, target_distance, model_name)
    drag_param = (0.5 * RHO * Cd * math.pi * radius**2) / mass

    print(f"\nInputs  : velocity={initial_velocity} m/s  angle={elevation} deg"
          f"  wind_x={wind_x} m/s  drag_param={drag_param:.5f}"
          f"  landing_z={ground_z} m  target={target_distance} m")
    print(f"Model   : {model_name}")
    print(f"Verdict : {'HIT' if hit else 'MISS'}  (hit probability: {prob:.3f})")
