"""
app.py
==============================================================================
Master Orchestrator — India Inter-City Freight & Logistics Network Analytics
------------------------------------------------------------------------------
Execution flow:
    1. Generate synthetic telemetry            -> data_generator
    2. Clean + engineer features                -> pipeline_utils
    3. Train + evaluate ML delay classifier      -> model_engine
    4. Render a 2x2 executive analytics canvas   -> matplotlib / seaborn
    5. Render an interactive bottleneck map      -> folium

Run with:
    python app.py
==============================================================================
"""

from __future__ import annotations

import warnings

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_generator import INDIAN_FREIGHT_HUBS, generate_shipment_telemetry
from model_engine import train_and_evaluate
from pipeline_utils import run_pipeline

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# Global style configuration
# ------------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["font.size"] = 10

N_SHIPMENTS = 3000
DASHBOARD_OUTPUT_PATH = "india_freight_dashboard.png"
MAP_OUTPUT_PATH = "india_freight_bottlenecks.html"
LONG_HAUL_DELAY_THRESHOLD_HOURS = 48


# ==============================================================================
# STEP 1-3: Data generation -> cleaning/feature engineering -> ML training
# ==============================================================================
def run_data_and_model_stage() -> tuple[pd.DataFrame, object]:
    print("\n[1/5] Generating synthetic shipment telemetry (n=3000)...")
    raw_df = generate_shipment_telemetry(N_SHIPMENTS)
    print(f"      -> {len(raw_df)} raw records | "
          f"{raw_df['actual_hours'].isna().sum()} missing actual_hours "
          f"({raw_df['actual_hours'].isna().mean():.2%})")

    print("\n[2/5] Cleaning data & engineering features...")
    processed_df = run_pipeline(raw_df)
    print(f"      -> route_distance_km, avg_speed_kmh, is_delayed engineered")
    print(f"      -> Overall delay rate: {processed_df['is_delayed'].mean():.2%}")

    print("\n[3/5] Training GradientBoostingClassifier delay predictor...")
    artifacts = train_and_evaluate(processed_df)

    return processed_df, artifacts


# ==============================================================================
# STEP 4: 2x2 Executive Analytics Dashboard
# ==============================================================================
def build_executive_dashboard(df: pd.DataFrame, artifacts, output_path: str = DASHBOARD_OUTPUT_PATH) -> None:
    print("\n[4/5] Rendering 2x2 executive analytics dashboard...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "India Inter-City Freight & Logistics Network — Executive Analytics Dashboard",
        fontsize=17, fontweight="bold", y=0.995
    )

    # --- Panel 1: Delay ratio by season/weather, grouped by fleet class ---
    ax1 = axes[0, 0]
    delay_ratio = (
        df.groupby(["season_condition", "fleet_type"])["is_delayed"]
        .mean()
        .reset_index()
        .rename(columns={"is_delayed": "delay_ratio"})
    )
    sns.barplot(
        data=delay_ratio,
        x="season_condition", y="delay_ratio", hue="fleet_type",
        ax=ax1, palette="rocket"
    )
    ax1.set_title("Delay Ratio by Weather/Season × Fleet Class", fontweight="bold")
    ax1.set_xlabel("Season / Weather Condition")
    ax1.set_ylabel("Delay Ratio")
    ax1.legend(title="Fleet Type", fontsize=8, title_fontsize=9, loc="upper left")
    ax1.set_ylim(0, max(delay_ratio["delay_ratio"].max() * 1.25, 0.1))

    # --- Panel 2: Fleet velocity spreads (box plot) ---
    ax2 = axes[0, 1]
    sns.boxplot(
        data=df, x="fleet_type", y="avg_speed_kmh",
        ax=ax2, palette="mako", hue="fleet_type", legend=False
    )
    ax2.set_title("Fleet Velocity Distribution (Avg Speed km/h)", fontweight="bold")
    ax2.set_xlabel("Fleet Type")
    ax2.set_ylabel("Average Speed (km/h)")
    ax2.tick_params(axis="x", rotation=12)

    # --- Panel 3: Geographical distance vs transit hours ---
    ax3 = axes[1, 0]
    sns.scatterplot(
        data=df, x="route_distance_km", y="actual_hours",
        hue="is_delayed", palette={0: "#2E86AB", 1: "#E63946"},
        alpha=0.55, s=28, ax=ax3
    )
    ax3.set_title("Route Distance vs. Actual Transit Hours", fontweight="bold")
    ax3.set_xlabel("Route Distance (km)")
    ax3.set_ylabel("Actual Transit Hours")
    ax3.legend(title="Delayed", labels=["On-Time", "Delayed"], fontsize=8, title_fontsize=9)

    # --- Panel 4: Confusion matrix heatmap ---
    ax4 = axes[1, 1]
    sns.heatmap(
        artifacts.confusion_matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=["On-Time", "Delayed"], yticklabels=["On-Time", "Delayed"],
        cbar=True, ax=ax4, linewidths=0.5, linecolor="white",
        annot_kws={"fontsize": 13, "fontweight": "bold"}
    )
    ax4.set_title(
        f"ML Confusion Matrix (Accuracy: {artifacts.accuracy:.2%})", fontweight="bold"
    )
    ax4.set_xlabel("Predicted Label")
    ax4.set_ylabel("Actual Label")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"      -> Dashboard saved to '{output_path}'")


# ==============================================================================
# STEP 5: Interactive Folium Bottleneck Map
# ==============================================================================
def build_bottleneck_map(df: pd.DataFrame, output_path: str = MAP_OUTPUT_PATH) -> None:
    print("\n[5/5] Rendering interactive Folium bottleneck map...")

    # Center the map over India, anchored on Nagpur (India's logistical
    # geographic center and home to the Nagpur-Wardha MMLP).
    nagpur_coords = INDIAN_FREIGHT_HUBS["Nagpur-Wardha"]
    freight_map = folium.Map(
        location=nagpur_coords,
        zoom_start=5,
        tiles="CartoDB positron",
    )

    # --- Plot hub markers ---
    for hub_name, (lat, lon) in INDIAN_FREIGHT_HUBS.items():
        folium.Marker(
            location=[lat, lon],
            popup=f"<b>{hub_name}</b><br>Multi-Modal Logistics Hub",
            tooltip=hub_name,
            icon=folium.Icon(color="darkblue", icon="warehouse", prefix="fa"),
        ).add_to(freight_map)

    # --- Identify severe long-haul bottlenecks ---
    bottlenecks = df[
        (df["is_delayed"] == 1) & (df["actual_hours"] > LONG_HAUL_DELAY_THRESHOLD_HOURS)
    ].copy()

    print(f"      -> {len(bottlenecks)} severe long-haul bottleneck shipments identified "
          f"(delayed & transit > {LONG_HAUL_DELAY_THRESHOLD_HOURS}h)")

    bottleneck_layer = folium.FeatureGroup(name="Severe Long-Haul Bottlenecks")

    for _, row in bottlenecks.iterrows():
        origin_point = [row["origin_lat"], row["origin_lon"]]
        dest_point = [row["destination_lat"], row["destination_lon"]]

        folium.PolyLine(
            locations=[origin_point, dest_point],
            color="#E63946",
            weight=2.0,
            opacity=0.45,
            tooltip=(
                f"{row['origin_hub']} → {row['destination_hub']}<br>"
                f"Fleet: {row['fleet_type']}<br>"
                f"Distance: {row['route_distance_km']:.1f} km<br>"
                f"Actual Transit: {row['actual_hours']:.1f} hrs "
                f"(Scheduled: {row['scheduled_hours']:.1f} hrs)<br>"
                f"Season: {row['season_condition']}"
            ),
        ).add_to(bottleneck_layer)

    bottleneck_layer.add_to(freight_map)
    folium.LayerControl(collapsed=False).add_to(freight_map)

    # --- Legend ---
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background-color: white; padding: 12px 16px; border-radius: 8px;
                border: 2px solid #444; font-size: 13px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <b>India Freight Network — Bottleneck Analysis</b><br>
        <span style="color:#00008B;">&#9679;</span> MMLP / Freight Hub<br>
        <span style="color:#E63946;">&#9473;&#9473;</span> Delayed Long-Haul Route (&gt;48h)
    </div>
    """
    freight_map.get_root().html.add_child(folium.Element(legend_html))

    freight_map.save(output_path)
    print(f"      -> Interactive map saved to '{output_path}'")


# ==============================================================================
# Orchestration entry point
# ==============================================================================
def main() -> None:
    print("=" * 78)
    print(" INDIA INTER-CITY FREIGHT & LOGISTICS NETWORK ANALYTICS")
    print(" End-to-End Pipeline Execution")
    print("=" * 78)

    processed_df, artifacts = run_data_and_model_stage()
    build_executive_dashboard(processed_df, artifacts)
    build_bottleneck_map(processed_df)

    print("\n" + "=" * 78)
    print(" PIPELINE COMPLETE")
    print(f"   - Dashboard : {DASHBOARD_OUTPUT_PATH}")
    print(f"   - Map       : {MAP_OUTPUT_PATH}")
    print("=" * 78)


if __name__ == "__main__":
    main()
