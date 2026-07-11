"""
pipeline_utils.py
==============================================================================
Data Cleaning, Distance Computation & Feature Engineering Utilities
------------------------------------------------------------------------------
This module contains the vectorized numerical core of the pipeline:

    1. haversine_np           - fully vectorized great-circle distance calc
    2. clean_shipment_data    - fleet-conditional imputation of sensor gaps
    3. engineer_features      - derives is_delayed target + avg_speed_kmh

All functions operate on / return pandas DataFrames and are designed to be
chained together inside app.py.
==============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_np(lat1: np.ndarray, lon1: np.ndarray,
                  lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Low-latency, fully vectorized Haversine great-circle distance.

    Operates natively on NumPy arrays (or pandas Series, which broadcast
    to arrays automatically) so it can compute distances for an entire
    DataFrame in a single call with no Python-level looping.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : array-like, degrees

    Returns
    -------
    np.ndarray of distances in kilometers.
    """
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(d_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)  # numerical safety guard against float overflow
    c = 2.0 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_KM * c


def add_route_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a vectorized `route_distance_km` column computed via haversine_np
    on the origin/destination coordinate pairs, then applies a fixed Indian
    highway road-circuity multiplier (roads are never a straight line)."""
    df = df.copy()

    straight_line_km = haversine_np(
        df["origin_lat"].to_numpy(),
        df["origin_lon"].to_numpy(),
        df["destination_lat"].to_numpy(),
        df["destination_lon"].to_numpy(),
    )

    # Fixed road-circuity constant representing average highway detour
    # factor on Indian NH/SH corridors relative to straight-line distance.
    ROAD_CIRCUITY_FACTOR = 1.25
    df["route_distance_km"] = np.round(straight_line_km * ROAD_CIRCUITY_FACTOR, 2)

    return df


def clean_shipment_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw telemetry DataFrame:

    - Imputes missing `actual_hours` (simulated offline-GPS sensor gaps)
      using a *conditional* transform: the median actual_hours of that
      specific `fleet_type`, rather than a single global median. This
      preserves the fleet-specific runtime distribution instead of biasing
      slower/faster fleets toward an unrealistic blended average.

    Returns a new DataFrame (does not mutate the input in place).
    """
    df = df.copy()

    fleet_median_hours = df.groupby("fleet_type")["actual_hours"].transform("median")
    df["actual_hours"] = df["actual_hours"].fillna(fleet_median_hours)

    # Defensive fallback: if an entire fleet group were somehow all-NaN,
    # fall back to the global median so no NaNs ever leak downstream.
    df["actual_hours"] = df["actual_hours"].fillna(df["actual_hours"].median())

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers the modelling-ready feature set:

      - route_distance_km  : vectorized haversine * road circuity factor
      - avg_speed_kmh       : route_distance_km / actual_hours
      - is_delayed          : binary target, 1 if actual_hours > scheduled_hours

    Assumes `clean_shipment_data` has already been run (no NaNs in
    actual_hours).
    """
    df = add_route_distance(df)
    df = df.copy()

    # Guard against divide-by-zero on any pathological zero-hour record.
    safe_hours = df["actual_hours"].replace(0, np.nan)
    df["avg_speed_kmh"] = np.round(df["route_distance_km"] / safe_hours, 2)
    df["avg_speed_kmh"] = df["avg_speed_kmh"].fillna(df["avg_speed_kmh"].median())

    df["is_delayed"] = (df["actual_hours"] > df["scheduled_hours"]).astype(int)

    df["transit_variance_hours"] = np.round(df["actual_hours"] - df["scheduled_hours"], 2)

    return df


def run_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper chaining clean -> feature engineering in the
    correct order. This is the single entry point app.py should call."""
    cleaned = clean_shipment_data(raw_df)
    engineered = engineer_features(cleaned)
    return engineered


if __name__ == "__main__":
    from data_generator import generate_shipment_telemetry

    raw = generate_shipment_telemetry(3000)
    processed = run_pipeline(raw)
    print(processed[[
        "fleet_type", "route_distance_km", "scheduled_hours",
        "actual_hours", "avg_speed_kmh", "is_delayed"
    ]].head(10))
    print(f"\nRemaining NaNs:\n{processed.isna().sum()[processed.isna().sum() > 0]}")
