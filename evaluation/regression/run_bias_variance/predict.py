import math
import os
import joblib
import numpy as np

_here      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.join(_here, "..", "..", "..")
MODELS_DIR = os.path.join(REPO_ROOT, "experiments", "regression", "run_bias_variance", "models")

MODELS = {
    "underfitting": os.path.join(MODELS_DIR, "model_underfitting.joblib"),
    "overfitting":  os.path.join(MODELS_DIR, "model_overfitting.joblib"),
}

RHO = 1.225  # kg/m3


def predict_velocity(elevation, wind_x, mass, radius, Cd, launch_height, landing_height, landing_distance, model_name="overfitting"):
    drag_param  = (0.5 * RHO * Cd * math.pi * radius ** 2) / mass
    height_diff = landing_height - launch_height

    model = joblib.load(MODELS[model_name])
    X = np.array([[elevation, wind_x, drag_param, height_diff, landing_distance]])
    return model.predict(X)[0]


if __name__ == "__main__":
    elevation        = 45.0
    wind_x           = 5.0
    mass             = 0.145
    radius           = 0.037
    Cd               = 0.47
    launch_height    = 0.0
    landing_height   = 0.0
    landing_distance = 90.0

    for name in MODELS:
        v = predict_velocity(elevation, wind_x, mass, radius, Cd, launch_height, landing_height, landing_distance, name)
        print(f"  {name:<15}  predicted velocity: {v:.2f} m/s")
