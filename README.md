# 🚛 India Inter-City Freight & Logistics Network Analytics

**An end-to-end data science system for modeling, predicting, and visualizing delay risk across India's Multi-Modal Logistics Park freight network.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/ML-GradientBoosting-orange?logo=scikitlearn)
![Folium](https://img.shields.io/badge/GeoViz-Folium-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

India's freight economy moves across a small number of critical **Multi-Modal Logistics Parks (MMLPs)** — Delhi-NCR, Mumbai-JNPT, Chennai-Mappedu, Bengaluru-Hub, Nagpur-Wardha, and Kolkata-Port — connected by a highway network that is highly exposed to **monsoon flooding, dense winter fog, and vehicle-class-dependent throughput limits**.

This project simulates a realistic fleet telemetry environment (3,000 shipments), builds a clean feature-engineering + ML pipeline to **predict shipment delay risk**, and produces **executive-ready visual analytics** — a static 2×2 dashboard and an interactive geospatial bottleneck map — entirely reproducible with one command.

It is designed to demonstrate production-grade data science engineering practices: modular architecture, vectorized geospatial computation, leakage-safe preprocessing, and stakeholder-facing reporting.

---

## 🏗️ Architecture

The project is deliberately split into single-responsibility modules rather than one monolithic script — mirroring how a real ML platform team would structure a production repo.

```
india_freight_analytics/
│
├── data_generator.py      # Synthetic telemetry generation engine
├── pipeline_utils.py      # Vectorized distance calc, cleaning, feature engineering
├── model_engine.py        # Scikit-Learn ColumnTransformer + GradientBoosting pipeline
├── app.py                 # Master orchestrator — runs the full pipeline end-to-end
├── requirements.txt       # Pinned third-party dependencies
└── README.md               # You are here
```

### Data & Control Flow

```
data_generator.py  ──►  raw telemetry DataFrame (3,000 shipments, 4% missing GPS)
        │
        ▼
pipeline_utils.py   ──►  haversine distance → fleet-conditional imputation
        │                 → is_delayed target → avg_speed_kmh feature
        ▼
model_engine.py     ──►  stratified train/test split → ColumnTransformer
        │                 (StandardScaler + OneHotEncoder) → GradientBoostingClassifier
        ▼
app.py               ──►  2×2 Matplotlib/Seaborn dashboard (.png)
        │                 + interactive Folium bottleneck map (.html)
        ▼
    Final Artifacts:  india_freight_dashboard.png
                        india_freight_bottlenecks.html
```

---

## ✨ Key Features

- **Realistic Synthetic Data Engine** — Six real Indian MMLP hub coordinates, Gaussian-jittered pickup/drop-off points, fleet-specific speed/payload profiles, and season-driven delay dynamics (Dry Season / Southwest Monsoon / Dense Fog).
- **Sensor-Realistic Missingness** — 4% of `actual_hours` records are deliberately nulled to simulate offline GPS transponders, a genuine operational pain point in Indian fleet telemetry.
- **Vectorized Geospatial Computation** — A pure NumPy, fully vectorized Haversine implementation computes great-circle route distance across all 3,000 shipments in a single array operation (no `.apply()` loops).
- **Leakage-Safe, Fleet-Conditional Imputation** — Missing runtime values are imputed using the *median of that specific fleet type*, not a single global average, preserving true fleet-level runtime distributions.
- **Production-Style ML Pipeline** — A single `sklearn.pipeline.Pipeline` wraps a `ColumnTransformer` (StandardScaler for numerics, OneHotEncoder for categoricals) directly into a `GradientBoostingClassifier`, trained on a stratified split to preserve class balance.
- **Executive Analytics Dashboard** — A single 2×2 Matplotlib/Seaborn canvas combining delay-driver analysis, fleet performance spread, distance/transit correlation, and live model diagnostics.
- **Interactive Geospatial Bottleneck Map** — A Folium map centered on Nagpur (India's logistical geographic center) plotting every severe long-haul delayed route (>48h transit) as an interactive polyline layer.

---

## 🧰 Tech Stack

| Layer                  | Tooling                                   |
|-------------------------|--------------------------------------------|
| Data generation          | NumPy, Pandas                              |
| Geospatial computation   | Vectorized NumPy Haversine                 |
| Data cleaning / features | Pandas (`groupby().transform()`)           |
| Machine Learning         | Scikit-Learn (`Pipeline`, `ColumnTransformer`, `GradientBoostingClassifier`) |
| Static visualization     | Matplotlib, Seaborn                        |
| Interactive geovisualization | Folium                                 |

---

## 📊 Output Artifacts

Running the pipeline produces two portfolio-ready deliverables:

### 1. `india_freight_dashboard.png` — Executive Analytics Canvas (2×2)
| Panel | Insight |
|---|---|
| **Delay Ratio by Season × Fleet** | Which fleet classes suffer most under monsoon/fog conditions |
| **Fleet Velocity Spread (Boxplot)** | Operational speed consistency per vehicle class |
| **Distance vs. Transit Hours (Scatter)** | Correlation between haul length and delay outcome |
| **Confusion Matrix (Heatmap)** | Live model diagnostic — precision/recall trade-off at a glance |

### 2. `india_freight_bottlenecks.html` — Interactive Bottleneck Map
An interactive Folium map (open directly in any browser) plotting:
- All six MMLP hub locations as markers
- Every shipment that was **both delayed and exceeded 48 hours of transit**, rendered as a polyline route with a hover tooltip showing fleet type, distance, and season condition

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/india-freight-analytics.git
cd india-freight-analytics
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python app.py
```

This single command will:
1. Generate 3,000 synthetic shipment records
2. Clean the data and engineer modeling features
3. Train and evaluate the `GradientBoostingClassifier`
4. Print a structured `classification_report` and confusion matrix to the console
5. Save `india_freight_dashboard.png` to the project root
6. Save `india_freight_bottlenecks.html` to the project root

### Run Individual Modules

Each module is independently executable for debugging / inspection:

```bash
python data_generator.py     # Preview raw synthetic telemetry
python pipeline_utils.py     # Preview cleaned + engineered dataset
python model_engine.py       # Train the model and print evaluation metrics only
```

---

## 🧪 Model Performance Snapshot

The `GradientBoostingClassifier` is trained on an 75/25 stratified split over engineered features (`route_distance_km`, `payload_tons`, `scheduled_hours`, `avg_speed_kmh`, `fleet_type`, `season_condition`, `origin_hub`, `destination_hub`) and typically achieves **~89–93% test accuracy**, with strong recall on the delayed class — the operationally critical error mode for a logistics dispatch team to minimize.

*(Exact metrics are regenerated and printed to console on every run — see the `classification_report` output.)*

---

## 🗺️ Simulated Freight Hub Network

| Hub | Region | Coordinates (approx.) |
|---|---|---|
| Delhi-NCR | North India | 28.4595° N, 77.0266° E |
| Mumbai-JNPT | West India (Port) | 18.9490° N, 72.9525° E |
| Chennai-Mappedu | South India | 13.2172° N, 80.0230° E |
| Bengaluru-Hub | South India | 13.1986° N, 77.7066° E |
| Nagpur-Wardha | Central India | 20.9463° N, 78.5570° E |
| Kolkata-Port | East India (Port) | 22.5726° N, 88.3639° E |

---

## 📁 Repository Structure Rationale

This project intentionally avoids a single notebook or script. Each file owns a distinct responsibility — data simulation, numerical/geospatial utilities, ML modeling, and orchestration — so that any layer (e.g., swapping `GradientBoostingClassifier` for `XGBoost`, or the synthetic generator for a real telemetry feed) can be modified in isolation without touching the rest of the codebase. This mirrors how ML systems are structured in production environments.

---

## 📄 License

This project is released under the MIT License. Synthetic data only — no real shipment, customer, or carrier data is used or represented.

---

## 👤 Author

Built as a portfolio demonstration of production-grade data science engineering: modular pipeline design, vectorized geospatial computation, leakage-aware feature engineering, and stakeholder-ready analytics reporting.
