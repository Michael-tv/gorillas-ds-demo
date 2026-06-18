import math
import os
import joblib
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "underfitting": os.path.join(_here, "models", "model_underfitting.joblib"),
    "overfitting":  os.path.join(_here, "models", "model_overfitting.joblib"),
}

RHO = 1.225  # kg/m3


def predict_velocity(elevation, wind_x, mass, radius, Cd, launch_height, landing_height, landing_distance, model_name="overfitting"):
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
