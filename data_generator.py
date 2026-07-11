"""
data_generator.py
==============================================================================
Synthetic Telemetry Generation Engine
------------------------------------------------------------------------------
Produces a highly realistic, reproducible synthetic dataset simulating GPS
and operational telemetry logs for inter-city freight shipments moving
between India's Multi-Modal Logistics Parks (MMLPs) and major freight hubs.

The generator deliberately injects real-world messiness (sensor dropout,
weather variance, fleet-specific runtime behaviour) so that the downstream
cleaning / feature-engineering / modelling stages have genuine signal to
work with, rather than a toy dataset.
==============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------------------
RANDOM_SEED = 42
_rng = np.random.default_rng(RANDOM_SEED)

# ------------------------------------------------------------------------------
# Real-world geographical anchors: major Indian MMLPs / freight hubs
# (lat, lon) — approximate coordinates of the logistics park / port complex
# ------------------------------------------------------------------------------
INDIAN_FREIGHT_HUBS: dict[str, tuple[float, float]] = {
    "Delhi-NCR":      (28.4595, 77.0266),   # Gurugram / Delhi-NCR MMLP corridor
    "Mumbai-JNPT":     (18.9490, 72.9525),   # Jawaharlal Nehru Port Trust
    "Chennai-Mappedu": (13.2172, 80.0230),   # Mappedu MMLP, Chennai
    "Bengaluru-Hub":   (13.1986, 77.7066),   # Bengaluru logistics hub (KIADB)
    "Nagpur-Wardha":   (20.9463, 78.5570),   # Nagpur-Wardha MMLP (India's freight center)
    "Kolkata-Port":    (22.5726, 88.3639),   # Kolkata Port / Dock complex
}

# ------------------------------------------------------------------------------
# Categorical feature domains
# ------------------------------------------------------------------------------
FLEET_TYPES = ["Tata Ace Mini", "Ashok Leyland Rigid", "BharatBenz Multi-Axle"]

# Fleet-specific operating characteristics: (avg cruising kmh, payload cap tons)
FLEET_PROFILE = {
    "Tata Ace Mini":          {"avg_speed": 38.0, "max_payload": 0.75, "base_hours_per_100km": 2.6},
    "Ashok Leyland Rigid":    {"avg_speed": 45.0, "max_payload": 9.0,  "base_hours_per_100km": 2.2},
    "BharatBenz Multi-Axle":  {"avg_speed": 52.0, "max_payload": 25.0, "base_hours_per_100km": 1.9},
}

SEASON_CONDITIONS = ["Dry Season", "Southwest Monsoon", "Dense Fog"]

# Season-specific delay multipliers applied to actual transit time
SEASON_DELAY_FACTOR = {
    "Dry Season":         0.94,
    "Southwest Monsoon":  1.08,
    "Dense Fog":          1.16,
}

# Season occurrence probabilities (roughly reflective of an annual cycle)
SEASON_PROBABILITIES = [0.55, 0.25, 0.20]

MISSING_ACTUAL_HOURS_FRACTION = 0.04  # 4% offline-GPS sensor dropout


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Scalar Haversine helper used only at generation time for ground-truth
    distance seeding. The production vectorized version lives in
    pipeline_utils.py and is used downstream on the full DataFrame."""
    r = 6371.0088
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)
    a = np.sin(d_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2.0) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _perturb_coordinate(lat: float, lon: float, noise_std_deg: float = 0.05) -> tuple[float, float]:
    """Adds minor Gaussian noise to a hub's anchor coordinate to simulate the
    exact pickup / drop-off point within that logistics park's catchment
    area (roughly a 3-6 km jitter radius)."""
    return (
        float(lat + _rng.normal(0, noise_std_deg)),
        float(lon + _rng.normal(0, noise_std_deg)),
    )


def generate_shipment_telemetry(n_shipments: int = 3000) -> pd.DataFrame:
    """
    Generates a synthetic fleet telemetry log of `n_shipments` inter-city
    freight movements across the six major Indian MMLP hubs.

    Returns
    -------
    pd.DataFrame with columns:
        shipment_id, origin_hub, destination_hub,
        origin_lat, origin_lon, destination_lat, destination_lon,
        fleet_type, payload_tons, season_condition,
        scheduled_hours, actual_hours (4% NaN)
    """
    hub_names = list(INDIAN_FREIGHT_HUBS.keys())
    records = []

    for i in range(n_shipments):
        # --- Origin / Destination sampling (ensures origin != destination) ---
        origin_hub, destination_hub = _rng.choice(hub_names, size=2, replace=False)
        o_lat_anchor, o_lon_anchor = INDIAN_FREIGHT_HUBS[origin_hub]
        d_lat_anchor, d_lon_anchor = INDIAN_FREIGHT_HUBS[destination_hub]

        o_lat, o_lon = _perturb_coordinate(o_lat_anchor, o_lon_anchor)
        d_lat, d_lon = _perturb_coordinate(d_lat_anchor, d_lon_anchor)

        # --- Fleet & payload ---
        fleet_type = _rng.choice(FLEET_TYPES, p=[0.35, 0.40, 0.25])
        profile = FLEET_PROFILE[fleet_type]
        payload_tons = float(np.clip(_rng.uniform(0.1, profile["max_payload"]), 0.05, None))

        # --- Season ---
        season_condition = _rng.choice(SEASON_CONDITIONS, p=SEASON_PROBABILITIES)

        # --- Ground-truth road distance (haversine * road-circuity factor) ---
        straight_km = _haversine_km(o_lat, o_lon, d_lat, d_lon)
        road_circuity = _rng.uniform(1.15, 1.35)  # Indian highway network detour factor
        road_km = straight_km * road_circuity

        # --- Scheduled hours: derived from distance & fleet base rate ---
        base_hours = (road_km / 100.0) * profile["base_hours_per_100km"]
        scheduled_hours = float(np.round(base_hours + _rng.normal(0, 1.0), 2))
        scheduled_hours = max(scheduled_hours, 1.0)

        # --- Actual hours: scheduled * season delay factor * operational noise ---
        season_factor = SEASON_DELAY_FACTOR[season_condition]
        traffic_noise = _rng.normal(1.0, 0.10)
        payload_drag = 1.0 + (payload_tons / profile["max_payload"]) * 0.05
        actual_hours = scheduled_hours * season_factor * traffic_noise * payload_drag
        actual_hours = float(np.round(max(actual_hours, 0.5), 2))

        records.append({
            "shipment_id": f"SHIP-{i + 1:05d}",
            "origin_hub": origin_hub,
            "destination_hub": destination_hub,
            "origin_lat": round(o_lat, 5),
            "origin_lon": round(o_lon, 5),
            "destination_lat": round(d_lat, 5),
            "destination_lon": round(d_lon, 5),
            "fleet_type": fleet_type,
            "payload_tons": round(payload_tons, 2),
            "season_condition": season_condition,
            "scheduled_hours": scheduled_hours,
            "actual_hours": actual_hours,
        })

    df = pd.DataFrame.from_records(records)

    # --- Inject 4% missingness into actual_hours (simulated offline GPS units) ---
    n_missing = int(round(len(df) * MISSING_ACTUAL_HOURS_FRACTION))
    missing_idx = _rng.choice(df.index, size=n_missing, replace=False)
    df.loc[missing_idx, "actual_hours"] = np.nan

    return df


if __name__ == "__main__":
    sample_df = generate_shipment_telemetry(3000)
    print(sample_df.head(10))
    print(f"\nGenerated {len(sample_df)} shipment records.")
    print(f"Missing actual_hours: {sample_df['actual_hours'].isna().sum()} "
          f"({sample_df['actual_hours'].isna().mean():.2%})")
