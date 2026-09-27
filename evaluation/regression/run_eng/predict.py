import os
import joblib
import pandas as pd

_here      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.join(_here, "..", "..", "..")
MODELS_DIR = os.path.join(REPO_ROOT, "experiments", "regression", "run_eng", "models")

# Raw columns -- the model's own Pipeline derives wind_x_ms/drag_param/
# height_diff_m internally (see feature_engineering.EngineeredFeatures).
FEATURES = ["launch_angle_deg", "wind_speed_ms", "wind_direction_norm",
            "mass_kg", "radius_m", "drag_coeff", "launch_height_m", "landing_height_m", "landing_distance_m"]

MODELS = {
    # Keys must match dvc_models.yaml. ridge and lasso live in skewed_models
    # only (see that file's matrix rule), so they are never trained in this run
    # and loading one here would fail on a missing file -- AUDIT.md task 3.
    "mlp":               os.path.join(MODELS_DIR, "model_mlp.joblib"),
    "decision_tree":     os.path.join(MODELS_DIR, "model_decision_tree.joblib"),
    "random_forest":     os.path.join(MODELS_DIR, "model_random_forest.joblib"),
    "linear_regression": os.path.join(MODELS_DIR, "model_linear_regression.joblib"),
    "polynomial":        os.path.join(MODELS_DIR, "model_polynomial.joblib"),
    "knn":               os.path.join(MODELS_DIR, "model_knn.joblib"),
    "xgboost":           os.path.join(MODELS_DIR, "model_xgboost.joblib"),
}


def predict_velocity(elevation, wind_x, mass, radius, Cd, launch_height, landing_height, landing_distance, model_name="mlp"):
    wind_speed = abs(wind_x)
    wind_dir   = 1.0 if wind_x >= 0 else -1.0

    model = joblib.load(MODELS[model_name])
    X = pd.DataFrame([[elevation, wind_speed, wind_dir, mass, radius, Cd, launch_height, landing_height, landing_distance]],
                      columns=FEATURES)
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
    drag_param = (0.5 * 1.225 * Cd * 3.141592653589793 * radius ** 2) / mass

    print(f"\nInputs  : angle={elevation} deg  wind_x={wind_x} m/s"
          f"  drag_param={drag_param:.5f}  launch_z={launch_height} m  landing_z={landing_height} m"
          f"  target={landing_distance} m")
    print(f"Model   : {model_name}")
    print(f"Required velocity : {velocity:.2f} m/s")
