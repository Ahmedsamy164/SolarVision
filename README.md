# SolarVision ☀️

**Solar-panel production monitoring, thermal inspection, and maintenance decision support in one Streamlit application.**

SolarVision combines hourly production analysis with a thermal-image classifier. The application identifies sustained production shortfalls, predicts a panel condition from a thermal image, and presents a maintenance suggestion when the user confirms that both inputs refer to the same panel/site and inspection period.

The unified entry point is **`solarvision_app.py`**. Existing model and data files remain required; they are not embedded in the application (the Docker image copies in only the few files it needs). The supplied SolarVision logo is embedded as the browser-tab icon.

**Quick start:** [run it with Docker](#run-with-docker) (recommended) or [install it locally](#local-installation-without-docker).

## Features

- Compare actual and expected hourly power on a chart.
- Use historical test predictions or upload an hourly CSV.
- Detect sustained under-production alerts and inspect affected readings.
- Download monitoring results as CSV.
- Classify thermal module images into 12 conditions and display model scores.
- Combine linked production and thermal evidence into a maintenance suggestion.
- Estimate cleaning economics from user-supplied assumptions.
- Optionally generate a 24-hour-ahead power prediction from a complete feature JSON.
- Keep detailed tables, calculation explanations, and advanced controls in optional sections.

## Repository layout

The application sits at the repository root. Preserve the existing nested directories because the application resolves artifacts relative to its own location.

```text
SolarVision/
├── README.md
├── solarvision_app.py
├── requirements.txt          # pinned dependencies for the unified app
├── Dockerfile                # container image for the unified app
├── .dockerignore             # keeps notebooks/datasets/media out of the image
├── SolarVision_TimeSeries_Feature-main/
│   └── SolarVision_TimeSeries_Feature-main/
│       ├── forecast_test_predictions_24h.csv
│       ├── api.py
│       ├── app_streamlit.py
│       ├── README.md
│       ├── models/
│       │   ├── best_forecast_24h.joblib
│       │   └── best_forecast_24h_features.joblib
│       └── ...
└── Solar-Panel-Anomaly-Detection-main/
    └── Solar-Panel-Anomaly-Detection-main/
        ├── class_mapping.json
        ├── data.zip
        ├── README.md
        ├── models/
        │   └── best_solar_cnn.keras
        └── ...
```

The original notebooks, API, and standalone dashboard remain separate files. The unified application loads the saved models directly; it does not require the FastAPI server to be running.

## Run with Docker

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) or Docker Engine (Linux), **running**. Commands are the same on every OS. Run them from the repository root.

```bash
docker build -t solarvision .
docker run --rm -p 8501:8501 solarvision
```

Open **http://localhost:8501**. Stop with **Ctrl+C**.

Useful variations:

```bash
# Different host port (app still listens on 8501 inside the container)
docker run --rm -p 8080:8501 solarvision          # -> http://localhost:8080

# Run in the background, follow logs, stop
docker run -d --rm --name solarvision -p 8501:8501 solarvision
docker logs -f solarvision
docker stop solarvision
```

What the image contains:

- `python:3.12-slim`, the pinned packages from `requirements.txt` (CPU-only TensorFlow), and the app running as a non-root user.
- Only the runtime files: `solarvision_app.py`, `forecast_test_predictions_24h.csv`, the 24-hour forecast model and its feature list, `class_mapping.json`, and `best_solar_cnn.keras`.
- **Not** included (see `.dockerignore`): notebooks, `data.zip`, the H5/ONNX model exports, figures, videos, and the raw/processed datasets. Thermal test images are uploaded from your computer through the browser, so extract `data.zip` on the host.
- A health check on `/_stcore/health`; `docker ps` shows `healthy` after about a minute.

The first build downloads TensorFlow and takes several minutes; later builds reuse the cached dependency layer unless `requirements.txt` changes. The first thermal classification after startup is slower because TensorFlow and the model load on first use.

### Checking that the container works

1. `docker ps` reports the container as `healthy` (allow ~60 seconds after start).
2. At http://localhost:8501, **Historical test demo** shows the actual/expected chart and three summary cards. This exercises the CSV and pandas path.
3. Upload a thermal image (see [Finding a labeled thermal example](#finding-a-labeled-thermal-example)). A detected condition, a score, and the Grad-CAM panels should appear. This exercises TensorFlow and the Keras model.
4. Optional: tick **Show advanced forecasting** in the sidebar and upload a feature JSON. This exercises the scikit-learn model. One valid input can be generated on the host (`pip install pyarrow` first):

   ```python
   import pandas as pd, json
   ts = "SolarVision_TimeSeries_Feature-main/SolarVision_TimeSeries_Feature-main/"
   names = json.load(open(ts + "processed_data/preprocessing_meta.json"))["features_24h"]
   row = pd.read_parquet(ts + "processed_data/forecast_ready.parquet").iloc[1000][names]
   json.dump({n: float(row[n]) for n in names}, open("sample_features.json", "w"))
   ```

## Local installation (without Docker)

Use **Python 3.12**, which is what the Docker image uses. `requirements.txt` pins the versions the app was built with. `scikit-learn` must stay at the pinned version because the saved forecast model was trained with it.

### Windows — Command Prompt

Open a terminal in the repository root:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run solarvision_app.py
```

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run solarvision_app.py
```

Open the local URL shown in the terminal, usually **http://localhost:8501**. Stop the application with **Ctrl+C**.

To launch from another working directory, supply the full path to `solarvision_app.py`.

## Interface design

The dashboard uses a navy, blue and gold visual theme with the supplied SolarVision logo in the page header, sidebar and browser tab. Production summary cards show mean actual output, mean expected output and alert hours. Thermal scores and cleaning benefit use separate cards, while maintenance suggestions appear in contextual messages. Detailed tables and settings remain expandable.

## Using the dashboard

### 1. Production data and sidebar settings

| Control | Meaning |
| --- | --- |
| Historical test demo | Loads saved historical test predictions from the time-series module. |
| Upload hourly CSV | Analyzes an uploaded table of actual and expected hourly power. |
| Shortfall threshold (%) | Required percentage drop below expected power; default 20%. |
| Electricity price / kWh | Value of recovered electricity in the selected currency. |
| Cleaning cost | Total cleaning cost in the same currency. |
| Expected recovery after cleaning | Assumed fraction of lost energy that cleaning could recover; default 0.8. |
| Number of equivalent modules (estimate) | Multiplier for extrapolating a single-module scenario; default 1. |
| Show advanced forecasting | Reveals the feature-JSON forecasting workflow. |

Uploaded expected-power values are **supplied by the CSV**. Uploading a CSV does not automatically run the forecasting model or create features from raw sensor data.

If the selected data includes `predicted_blend_W`, **Use saved blended forecast** can replace the expected-power series with those saved values. This does not train or fit a new blend.

### 2. Production monitoring

The chart compares **Actual power (W)** with **Expected power (W)**.

A reading is anomalous when:

- Expected power is at least **5 W**.
- Actual power is less than or equal to `expected × (1 − threshold)`.
- If `is_daytime` is supplied, it must also indicate daytime.

A sustained alert is raised when at least **3 of the last 6 hourly readings** are anomalous. They do not need to be consecutive. During startup, the window uses the available readings, so the first alert can occur on the third reading.

**Hours with sustained alerts** counts flagged rows, not separate incidents. **View alert readings** opens the affected readings, and **Download monitoring results** exports the analyzed table.

### 3. Thermal inspection

Upload a thermal module image (`.png`, `.jpg`, or `.jpeg`). Use images compatible with the training dataset, rather than ordinary visible-light photographs.

The application loads `best_solar_cnn.keras` and the labels in `class_mapping.json`. It converts the image to RGB, resizes it with TensorFlow bilinear interpolation to the model input dimensions (the current architecture uses 40 × 24 pixels), and divides pixel values by 255, matching the notebook's normalization.

The supported labels are:

| Label | Condition |
| --- | --- |
| Cell | Single-cell anomaly |
| Cell-Multi | Multiple-cell anomalies |
| Cracking | Cracking |
| Diode | Diode anomaly |
| Diode-Multi | Multiple-diode anomalies |
| Hot-Spot | Hot spot |
| Hot-Spot-Multi | Multiple hot spots |
| No-Anomaly | No anomaly predicted |
| Offline-Module | Offline module |
| Shadowing | Shadowing |
| Soiling | Soiling / dirt accumulation |
| Vegetation | Vegetation-related obstruction |

**Detected condition** is the highest-scoring class. **Model score** is a model output, not a calibrated probability of a correct diagnosis. **View all image results** shows the remaining class scores.

### Visual explanation (Grad-CAM)

After classification, the app displays the original image, a Grad-CAM heatmap, and a blended overlay. Warmer colors indicate stronger positive contribution to the predicted class. The explanation uses the Keras model and gradients from its last convolutional layer. It does not mark confirmed fault boundaries or measure temperature. If no positive activation exists, the app reports that rather than showing a misleading highlight. Explanation errors do not discard the classification result. Heatmap rendering was checked locally; model-based gradient execution still needs verification in the TensorFlow environment.

### 4. Maintenance recommendation

Confirm **The image and readings are from the same panel/site and inspection period** only when that relationship is established. This is a user confirmation; the application does not verify panel identity or capture time automatically.

Under **Assessment settings**, the minimum image model score defaults to **0.70**.

| Available evidence | Application behavior |
| --- | --- |
| No linked image | Reports production-monitoring guidance without attributing a cause to the image. |
| Image score below threshold | Requests manual image review. |
| Linked Soiling prediction + sustained alert + positive net benefit | Suggests considering cleaning. |
| Linked Soiling prediction without both alert and positive economics | Suggests inspection rather than recommending cleaning. |
| Linked No-Anomaly prediction + production alert | Suggests investigating weather, sensors, or other causes. |
| Another linked thermal condition | Suggests inspecting the predicted condition and displays mapped maintenance guidance. |

**Maintenance priority** and **Suggested action** come from the thermal class mapping. They are suggested inspection guidance, not a confirmed fault diagnosis or an automated maintenance order.

#### Cleaning estimate

The application sums positive hourly power shortfall on daytime readings belonging to days with a sustained alert. With one-hour intervals, dividing the summed watts by 1,000 gives module-scale kWh.

```text
Estimated plant loss = module shortfall kWh × equivalent modules
Recoverable energy  = estimated plant loss × assumed recovery fraction
Recoverable revenue = recoverable energy × electricity price
Net cleaning benefit = recoverable revenue − cleaning cost
```

**Estimated net cleaning benefit** is a scenario based on the analyzed period and supplied assumptions. It is not a prediction of future savings. A positive value alone does not trigger a cleaning recommendation; linked thermal evidence and a sustained alert are also required.

### 5. Advanced next-day forecasting

Enable **Show advanced forecasting**, expand **Advanced: forecast next-day power**, upload a feature JSON, and select **Predict power**.

The JSON must contain a complete feature-name/value object using the names in `best_forecast_24h_features.joblib`. Every required feature must be supplied as a finite numeric value. The app does not silently replace missing features with zero.

This workflow loads the saved 24-hour forecasting model and returns a module-scale power prediction in watts. It does not construct lag/rolling features from a raw CSV, call a weather service, generate a 24-point forecast curve, or automatically add its result to the monitoring chart. The 48-hour experiment is not connected to this interface.

## Hourly CSV format

The first column is the timestamp. Required data columns:

- `actual_power_W`
- `predicted_power_W`

Optional columns:

- `is_daytime`: `true`/`false` or `0`/`1`.
- `predicted_blend_W`: saved blended expected-power values.

Timestamps must be valid and unique, and the sorted readings must have consecutive one-hour intervals. Power values must be finite and nonnegative. Timestamps are normalized to UTC; timestamps without a timezone are treated as UTC. Include an explicit offset when your source uses local time.

```csv
timestamp,actual_power_W,predicted_power_W,is_daytime
2026-10-28T08:00:00Z,50,100,1
2026-10-28T09:00:00Z,50,100,1
2026-10-28T10:00:00Z,50,100,1
2026-10-28T11:00:00Z,50,100,1
2026-10-28T12:00:00Z,50,100,1
2026-10-28T13:00:00Z,50,100,1
```

## Reproduce the synthetic monitoring example

1. Save the CSV above as `demo_production.csv`.
2. Select **Upload hourly CSV** and upload it.
3. Use these settings:

| Setting | Value |
| --- | ---: |
| Shortfall threshold | 20% |
| Electricity price | 2 |
| Cleaning cost | 50 |
| Expected recovery | 0.8 |
| Equivalent modules | 1000 |

Expected monitoring results:

- **4 sustained-alert hours**: the third through sixth readings.
- Module shortfall on the alert day: `6 × 50 W × 1 h / 1000 = 0.3 kWh`.
- Scaled shortfall: `0.3 × 1000 = 300 kWh`.
- Net benefit: `300 × 0.8 × 2 − 50 = 430` currency units.

This example uses **synthetic production values**. It demonstrates alert and economic calculations; it does not establish a real link to a thermal image. Leave the matching-input confirmation unchecked for unrelated data. A real joint recommendation requires a corresponding image and production record.

### Finding a labeled thermal example

After extracting the thermal module's `data.zip`, the dataset contains:

```text
InfraredSolarModules/
├── module_metadata.json
└── images/
    └── ... numbered JPG images ...
```

`module_metadata.json` maps numbered images to their dataset labels. For example, `images/8291.jpg`, `images/8290.jpg`, and `images/8293.jpg` are labeled **Soiling** in the supplied archive. Model predictions can differ from these labels. These images may have been used during training, so classifying them is not an independent evaluation of accuracy.

## Data and evaluation context

The time-series module uses hourly single-module solar PV and weather-related data from **Laghouat, Algeria, during part of 2022**. Dust density is partly reconstructed. Forecast residuals do not prove soiling: shading, weather, sensor issues, and other faults can also reduce output.

The module documentation reports that the deployed Random Forest does not beat persistence on test MAE; a saved blend improves some other metrics. Adding the unified interface does not retrain the models or improve their measured accuracy.

See the existing module documentation for datasets, notebooks, model comparisons, and reproducibility details:

- [Time-series module](SolarVision_TimeSeries_Feature-main/SolarVision_TimeSeries_Feature-main/README.md)
- [Thermal classification module](Solar-Panel-Anomaly-Detection-main/Solar-Panel-Anomaly-Detection-main/README.md)

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `File does not exist: solarvision_app.py` | Open a terminal in the folder containing the file, or pass its full path. On Windows, check that it is not saved as `.py.txt`. |
| Missing Python module | Activate the environment and install the required package with `python -m pip`. |
| Historical CSV or model not found | Preserve the directory layout shown above. The Python file alone does not include datasets or models. |
| Saved model fails to load | Check the full error and use dependency versions compatible with the training environment. |
| CSV rejected | Check required columns, numeric values, duplicate timestamps, and missing hourly readings. |
| No cleaning recommendation | Check the sustained alert, predicted class, score threshold, net benefit, and whether the inputs really correspond. |
| `docker: command not found` / cannot connect to the Docker daemon | Install Docker and make sure Docker Desktop is running before `docker build`. |
| `port is already allocated` | Another process uses 8501. Map a different host port, e.g. `-p 8080:8501`. |
| Container shows `starting` or `unhealthy` | Wait about a minute after launch; if it persists, read `docker logs solarvision`. |
| `COPY failed: file not found` during build | Run `docker build` from the repository root and keep the nested folder layout. |
| Old browser-tab icon | Restart the application and reopen the tab; the browser may retain the previous icon. |

## Current scope and limitations

- One application now presents both pipelines and combines their outputs through explicit decision rules.
- There is no live sensor, camera, weather-service, scheduling, or cleaning-system connection.
- The app does not automatically associate images with a panel or date.
- The forecasting and thermal models were developed separately; the combined maintenance recommendation has not been validated against a matched real-world dataset.
- Financial outputs depend on user assumptions and hypothetical scaling; they are not guaranteed savings.
- The 48-hour forecasting experiment remains separate.
- Basic logic checks cover the synthetic alert/energy calculation and key maintenance decision branches. Python syntax was checked after interface updates. Complete end-to-end model and UI validation remains dependent on a compatible local or Docker environment (see [Checking that the container works](#checking-that-the-container-works)); no new accuracy benchmark is claimed.

## Project modules

SolarVision brings together two separately developed pipelines, which `solarvision_app.py` now presents through a single interface.

| Module | Folder | What it provides |
| --- | --- | --- |
| Time-series forecasting & anomaly intelligence | `SolarVision_TimeSeries_Feature-main/` | Preprocessing, 24-hour forecasting, residual-based under-production alerts, cleaning cost-benefit analysis, FastAPI service, original Streamlit dashboard. |
| Thermal solar-panel classification | `Solar-Panel-Anomaly-Detection-main/` | CNN classifier for 12 thermal conditions, evaluation plots, model exports (Keras/H5/ONNX), MLflow logging. |

The original notebooks, `api.py` and `app_streamlit.py` are kept for reproducibility and are not needed to run the unified app. The repeated folder names come from extracting the original project archives; if they are ever flattened, update the paths in `solarvision_app.py`, the `Dockerfile` and the notebooks, and test again.
