from pathlib import Path
from datetime import datetime
import csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LOG_FILE = RESULTS_DIR / "prediction_log.csv"


def log_prediction(result):

    file_exists = LOG_FILE.exists()

    with open(LOG_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Input Type",
                "Selected Model",
                "Prediction",
                "Confidence (%)",
                "Confidence Level",
                "Recommended Action",
                "Severity",
                "Inference Time (ms)"
            ])

        writer.writerow([
            result["timestamp"],
            result["input_type"],
            result["selected_model"],
            result["prediction"],
            result["confidence"],
            result["confidence_level"],
            result["recommended_action"],
            result["severity"],
            result["inference_time_ms"]
        ])
