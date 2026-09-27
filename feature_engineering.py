"""Shared feature engineering: derives wind_x_ms, drag_param, and height_diff_m
from the raw physical columns every generated dataset stores.

Generated datasets only ever hold raw physical inputs (never these derived
columns) -- every "engineered features" run computes them here, at load time,
either by calling add_engineered_columns() directly or by using
EngineeredFeatures as a step in an sklearn Pipeline.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

RHO = 1.225  # kg/m^3

RAW_INPUT_COLUMNS = [
    "wind_speed_ms", "wind_direction_norm",
    "mass_kg", "radius_m", "drag_coeff",
    "launch_height_m", "landing_height_m",
]
ENGINEERED_COLUMNS = ["wind_x_ms", "drag_param", "height_diff_m"]


def add_engineered_columns(df):
    """Return a copy of df with wind_x_ms/drag_param/height_diff_m added."""
    df = df.copy()
    df["wind_x_ms"]     = df["wind_speed_ms"] * df["wind_direction_norm"]
    df["drag_param"]    = (0.5 * RHO * df["drag_coeff"] * np.pi * df["radius_m"] ** 2) / df["mass_kg"]
    df["height_diff_m"] = df["landing_height_m"] - df["launch_height_m"]
    return df


class EngineeredFeatures(BaseEstimator, TransformerMixin):
    """sklearn Pipeline step: raw physical columns -> engineered feature columns.

    Fit/transform on a DataFrame containing RAW_INPUT_COLUMNS (plus whatever
    else); transform() returns just `output_columns` (a mix of engineered and
    passthrough raw columns), ready to feed into an estimator.
    """

    def __init__(self, output_columns):
        self.output_columns = output_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_engineered_columns(X)[self.output_columns]
