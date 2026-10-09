"""
Automated unit tests for Solar Irradiance prediction and preprocessing pipeline.
Validates input sanitization, feature transformations, and model inference contracts.
"""

import math
import numpy as np
import pytest

from app import compute_features, predict_solar

def test_compute_features_shape_and_types():
    """Verify that raw telemetry is transformed into expected 8-feature float32 array."""
    feats = compute_features(
        hour=12,
        month=6,
        temperature=28.5,
        lag_1h=450.0,
        rolling_3h=420.0,
        max_irr_past=750.0,
    )
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (1, 8)
    assert feats.dtype == np.float32

def test_cyclical_encoding_bounds():
    """Verify trigonometric hour and month representations remain strictly within [-1.0, 1.0]."""
    for hour in range(24):
        for month in range(1, 13):
            feats = compute_features(hour, month, temperature=25.0, lag_1h=300.0, rolling_3h=300.0)
            hour_sin, hour_cos = feats[0, 0], feats[0, 1]
            month_sin, month_cos = feats[0, 2], feats[0, 3]

            assert -1.0 <= hour_sin <= 1.0
            assert -1.0 <= hour_cos <= 1.0
            assert -1.0 <= month_sin <= 1.0
            assert -1.0 <= month_cos <= 1.0
            assert math.isclose(hour_sin**2 + hour_cos**2, 1.0, abs_tol=1e-5)

def test_nighttime_clamping_safety():
    """Physical constraint test: At midnight (hour 0 or 23), irradiance MUST be 0 W/m²."""
    for night_hour in [0, 1, 2, 3, 4, 20, 21, 22, 23]:
        feats = compute_features(hour=night_hour, month=6, temperature=15.0, lag_1h=0.0, rolling_3h=0.0)
        ghi, regime, power = predict_solar(feats, "Deep Neural Network (PyTorch DNN)", night_hour)

        assert ghi == 0.0, f"Nighttime irradiance at hour {night_hour} must be 0, got {ghi}"
        assert regime == "Low"
        assert power == 0.0

def test_noon_peak_positive_generation():
    """Peak sun at noon in summer must yield positive irradiance and energy output."""
    feats = compute_features(hour=12, month=6, temperature=32.0, lag_1h=600.0, rolling_3h=580.0)
    ghi, regime, power = predict_solar(feats, "Deep Neural Network (PyTorch DNN)", 12)

    assert ghi > 0.0
    assert regime in ["Medium", "High"]
    assert power > 0.0
