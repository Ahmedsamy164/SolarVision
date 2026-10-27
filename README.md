# SolarVision ☀️

**SolarVision** is a solar-panel monitoring project that explores two complementary AI approaches:

1. **Time-series forecasting and operational anomaly alerts** using solar power and weather-related data.
2. **Thermal image classification** to classify infrared images of solar modules into anomaly categories.

The two modules are currently maintained as separate pipelines in this repository. They should not be described as one fully integrated service until their inference workflows and application have been connected and tested together.

## Project Modules

### 1. Time-Series Forecasting & Anomaly Intelligence

This module works with the hourly Laghouat, Algeria solar PV and dust dataset. It includes preprocessing, feature engineering, 24-hour forecasting, an exploratory 48-hour experiment, residual-based under-production alerts, and a scenario-based cleaning cost-benefit calculation.

Main components:

- `01_preprocessing.ipynb` — data preparation, target construction, and time-aware features.
- `02_training.ipynb` — model comparison and 24-hour forecasting experiments, including MLflow/Optuna/SHAP code.
- `02b_training_48h.ipynb` — 48-hour forecasting experiment.
- `03_anomaly_cleaning.ipynb` — anomaly rules and cleaning decision analysis.
- `api.py` — FastAPI endpoints for forecast, anomaly, and cleaning functions.
- `app_streamlit.py` — Streamlit dashboard based on saved test predictions and demo scenarios.

Read the module-specific guide here: [`Time-Series README`](SolarVision_TimeSeries_Feature-main/SolarVision_TimeSeries_Feature-main/README.md).

**Important limitations:** the dataset represents one site in Laghouat, Algeria, during part of 2022. The dust-density series is partly reconstructed, and a forecast residual is not proof that dust caused under-production. The 48-hour experiment is not currently connected to the API/dashboard. Review the module README for the reported evaluation results and assumptions.

### 2. Thermal Solar-Panel Anomaly Classification

This module uses infrared thermal images and deep-learning models to classify solar-module conditions. It contains a notebook, class mapping, evaluation plots, model exports, and MLflow logging code.

Main components:

- `notebooks/solar-panel-anomaly-detection.ipynb` — model training and evaluation workflow.
- `class_mapping.json` — mapping between class labels and model outputs.
- `models/` — exported model files in Keras/H5/ONNX formats.
- `*CV*.png`, `*OUT*.png`, and training/validation plots — experiment evidence and visual results.
- `log_to_mlflow.py` — MLflow logging helper.

Read the module-specific guide here: [`Thermal Classification README`](Solar-Panel-Anomaly-Detection-main/Solar-Panel-Anomaly-Detection-main/README.md).

The module README documents the dataset, class list, training approach, reported metrics, and limitations. Verify the results and model files by rerunning the notebook before treating them as final release evidence.

## Repository Layout

The source projects are currently kept in separate folders to preserve their original files and instructions:

```text
SolarVision/
├── README.md
├── SolarVision_TimeSeries_Feature-main/
│   └── SolarVision_TimeSeries_Feature-main/
│       ├── 01_preprocessing.ipynb
│       ├── 02_training.ipynb
│       ├── 02b_training_48h.ipynb
│       ├── 03_anomaly_cleaning.ipynb
│       ├── api.py
│       ├── app_streamlit.py
│       └── README.md
└── Solar-Panel-Anomaly-Detection-main/
    └── Solar-Panel-Anomaly-Detection-main/
        ├── notebooks/
        ├── models/
        └── README.md
```

The repeated folder names came from extracting the original project archives. They can be simplified later, but after moving files, update relative paths in notebooks and applications and test both modules again.

## Getting Started

Each module has its own dependencies and run instructions. Start with the relevant module README rather than installing every library at once:

- [Time-Series setup and usage](SolarVision_TimeSeries_Feature-main/SolarVision_TimeSeries_Feature-main/README.md)
- [Thermal Classification setup and usage](Solar-Panel-Anomaly-Detection-main/Solar-Panel-Anomaly-Detection-main/README.md)

Use separate Python environments if the modules require incompatible library versions. Do not commit virtual environments, API keys, `.env` files, or other secrets.

## Data, Models, and Reproducibility

- Follow the dataset source and licensing terms documented by each module.
- Large datasets and generated files may be better stored outside Git or managed with Git LFS/DVC, depending on repository limits and team needs.
- Keep model artifacts that are needed for inference; remove duplicate exports only after confirming which format the code actually loads.
- Re-run notebooks from top to bottom and record the environment and evaluation split before presenting metrics as reproducible results.
- The two modules are not yet a unified end-to-end application. Any future integration should document the interface between time-series alerts and image-classification outputs and test the combined workflow.

## Contributing

1. Create a branch for your change, for example `feature/eda` or `fix/forecast-api`.
2. Make a focused change and test the affected notebook or module.
3. Commit with a descriptive message.
4. Open a Pull Request to `main` and ask a teammate to review it before merging.

Avoid changing the same notebook or shared configuration simultaneously where possible. Keep dataset, model, and dependency changes documented.

## Team

Add the project members and their technical responsibilities here before the final submission.

## Project Status

This repository brings together two existing project modules. Repository cleanup, reproducibility checks, and integration between the modules should be completed and verified before describing the project as a single production-ready system.
