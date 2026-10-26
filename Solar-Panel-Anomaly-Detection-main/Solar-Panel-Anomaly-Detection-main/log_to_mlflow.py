import os
import json
import mlflow
import mlflow.keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

# Set project paths
project_dir = r"D:\Samsung\Solar Panel Project"
models_dir = os.path.join(project_dir, "models")
data_dir = os.path.join(project_dir, "data") 
# Set MLflow experiment name
mlflow.set_experiment("Solar_Panel_Anomaly_Detection")

with mlflow.start_run(run_name="Solar_Models_With_Full_Metrics"):
    
    # 1. Log class mapping JSON file as an artifact
    json_path = os.path.join(project_dir, "class_mapping.json")
    if os.path.exists(json_path):
        mlflow.log_artifact(json_path, artifact_path="config")
        print("Logged class_mapping.json successfully.")

    # -------------------------------------------------------------
    # Baseline Model Metrics
    # -------------------------------------------------------------
    mlflow.log_param("baseline_model_type", "Custom CNN Baseline")
    mlflow.log_metric("baseline_test_accuracy", 0.5918)
    mlflow.log_metric("baseline_precision", 0.621)
    mlflow.log_metric("baseline_recall", 0.592)
    mlflow.log_metric("baseline_f1_score", 0.587)

    # -------------------------------------------------------------
    # Best Model Metrics (Custom Enhanced CNN)
    # -------------------------------------------------------------
    mlflow.log_param("best_model_type", "Custom Enhanced CNN")
    mlflow.log_metric("best_test_accuracy", 0.9290)
    mlflow.log_metric("best_precision", 0.9289)
    mlflow.log_metric("best_recall", 0.9290)
    mlflow.log_metric("best_f1_score", 0.9283)

    # 3. Load and log the Keras models to MLflow
    baseline_keras_path = os.path.join(models_dir, "baseline_solar_cnn.keras")
    if os.path.exists(baseline_keras_path):
        baseline_model = tf.keras.models.load_model(baseline_keras_path)
        mlflow.keras.log_model(baseline_model, name="baseline_model")

    best_keras_path = os.path.join(models_dir, "best_solar_cnn.keras")
    if os.path.exists(best_keras_path):
        best_model = tf.keras.models.load_model(best_keras_path)
        mlflow.keras.log_model(best_model, name="best_model")

    # 4. Log all model files (H5, ONNX, Keras) as artifacts
    mlflow.log_artifacts(models_dir, artifact_path="model_files_all_formats")
    print("All model files logged as artifacts successfully.")

    # 5. Generate and log Comparison Chart (Accuracy, Precision, Recall, F1)
    plt.figure(figsize=(10, 5))
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    baseline_scores = [0.5918, 0.621, 0.592, 0.587]
    best_scores = [0.9290, 0.9289, 0.9290, 0.9283]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    plt.bar(x - width/2, baseline_scores, width, label='Baseline CNN', color='navy')
    plt.bar(x + width/2, best_scores, width, label='Best Custom CNN', color='orangered')
    
    plt.ylabel('Score')
    plt.title('Performance Comparison Between Baseline and Best Models')
    plt.xticks(x, metrics_names)
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    comparison_chart_path = os.path.join(project_dir, "models_comparison_metrics.png")
    plt.savefig(comparison_chart_path)
    plt.close()
    
    mlflow.log_artifact(comparison_chart_path, artifact_path="charts")
    print("Comparison metrics chart logged successfully.")

print("MLflow Logging with Full Metrics Completed Successfully!")