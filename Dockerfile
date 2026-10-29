# SolarVision unified app (solarvision_app.py) - run from the repository root.
#   docker build -t solarvision .
#   docker run --rm -p 8501:8501 solarvision        ->  http://localhost:8501
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Dependencies first (cached layer: rebuilds are fast when only code changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The app resolves its data/models relative to its own location, so the nested
# folder layout must be kept exactly as in the repository.
COPY solarvision_app.py ./
COPY SolarVision_TimeSeries_Feature-main/SolarVision_TimeSeries_Feature-main/forecast_test_predictions_24h.csv \
     ./SolarVision_TimeSeries_Feature-main/SolarVision_TimeSeries_Feature-main/
COPY SolarVision_TimeSeries_Feature-main/SolarVision_TimeSeries_Feature-main/models/best_forecast_24h.joblib \
     SolarVision_TimeSeries_Feature-main/SolarVision_TimeSeries_Feature-main/models/best_forecast_24h_features.joblib \
     ./SolarVision_TimeSeries_Feature-main/SolarVision_TimeSeries_Feature-main/models/
COPY Solar-Panel-Anomaly-Detection-main/Solar-Panel-Anomaly-Detection-main/class_mapping.json \
     ./Solar-Panel-Anomaly-Detection-main/Solar-Panel-Anomaly-Detection-main/
COPY Solar-Panel-Anomaly-Detection-main/Solar-Panel-Anomaly-Detection-main/models/best_solar_cnn.keras \
     ./Solar-Panel-Anomaly-Detection-main/Solar-Panel-Anomaly-Detection-main/models/

# Don't run as root
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health').status == 200 else 1)"

CMD ["streamlit", "run", "solarvision_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
