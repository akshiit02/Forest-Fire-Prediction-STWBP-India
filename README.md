# Forest Fire Prediction System using Multi-Task STW-BP

A multi-task deep learning system that predicts both **wildfire occurrence** and **wildfire intensity** across India, built on NASA MODIS satellite fire-detection data and a custom Spatio-Temporal Weighted (STW-BP) neural network architecture.

## Overview

Most fire-risk models treat "will a fire happen" and "how severe will it be" as separate problems. This project trains a single network to predict both jointly, while learning region-specific feature weighting along the way.

## Pipeline

1. **Data preprocessing** (`src/data_preprocessing.py`) — cleans NASA FIRMS fire-detection records for India (2012–2019): filters to high-confidence detections, parses timestamps into year/month/day-of-year/hour, removes duplicates, and clips outliers in brightness/FRP.
2. **Non-fire sample generation** (`src/nonfire_generation.py`) — the raw FIRMS data only contains confirmed fire detections, so there's no negative class. This generates synthetic non-fire samples by spatially jittering real fire points (within India's bounding box) and scaling down their driver features, producing a balanced binary dataset.
3. **Intensity labeling** (`add_intensity.py`) — derives a 1–5 fire intensity scale from Fire Radiative Power (FRP), used as the second task target.
4. **Baseline models** — Random Forest (`src/baseline_random_forest.py`) and XGBoost (`src/xgboost_model.py`), trained on the same feature set as a comparison point for the main model.
5. **STW-BP model** (`src/stwbp_model.py`) — the core architecture. Spatial-temporal features (latitude, longitude, month) pass through a small subnetwork that learns adaptive sigmoid weights, which scale the satellite driver features (brightness, brightness temperature, FRP, scan, track) before they reach the main classifier. Two output heads are trained jointly: fire occurrence (binary) and fire intensity (5-class).
6. **Regional weight analysis** (`src/weight_analysis.py`) — inspects what the adaptive-weighting layer learned per region (Central India, Himalayan, North-East, Western Ghats), showing which satellite-derived features drive predictions differently across the country.
7. **Live inference** (`live_test.py`) — runs the trained model on fresh satellite data, filtered to India's bounding box with realism filters on confidence, brightness, and FRP.

## Results

On a held-out test split, the fire-occurrence head reached ~98% accuracy and ~0.998 ROC-AUC; see the evaluation block in `src/stwbp_model.py` for the exact metrics computed.

## Data source

[NASA FIRMS / MODIS Active Fire Data](https://firms.modaps.eosdis.nasa.gov/) — active fire detections, filtered to India.

## Built with

Python, TensorFlow / Keras, scikit-learn, XGBoost, Pandas, NumPy

## Project structure

```
├── src/
│   ├── data_preprocessing.py    # clean raw FIRMS data
│   ├── nonfire_generation.py    # synthetic negative sampling
│   ├── baseline_random_forest.py
│   ├── xgboost_model.py
│   ├── stwbp_model.py           # main multi-task model
│   └── weight_analysis.py       # regional feature-weight analysis
├── add_intensity.py             # derives fire intensity labels
├── check_dataset.py             # quick dataset sanity checks
├── check_intensity.py           # intensity distribution check
├── live_test.py                 # inference on live satellite data
└── results/
    └── regional_analysis/       # learned feature weights by region
```

## Setup

```bash
git clone https://github.com/akshiit02/Forest-Fire-Prediction-STWBP-India.git
cd Forest-Fire-Prediction-STWBP-India
pip install -r requirements.txt
```

Datasets and trained model weights aren't included here due to size. Download MODIS/FIRMS active fire data for India from the link above, place it at `data/raw/`, then run the pipeline in order:

```bash
python src/data_preprocessing.py
python src/nonfire_generation.py
python add_intensity.py
python src/stwbp_model.py
```

## What I'd add next

- Package the trained model behind a small API, similar to the deployment layer in my [AgroVision project](https://github.com/akshiit02/AI-Vision-System-for-Crop-and-Pest-Monitoring)
- Add a notebook reproducing the regional weight-analysis plots
- Try a small transformer-based spatio-temporal encoder in place of the current dense adaptive-weighting layer
