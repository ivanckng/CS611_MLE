import os
from datetime import date, datetime, timedelta
import pandas as pd
import argparse


# --------------------------------
# Parse Command Line Arguments
# --------------------------------
parser = argparse.ArgumentParser(description="Model Inference Completion.")
parser.add_argument('--snapshotdate', type=str, help='The snapshot date for model inference completion (YYYY-MM-DD).')
args = parser.parse_args()
current_date_str = args.snapshotdate
current_date = datetime.strptime(current_date_str, '%Y-%m-%d').date()

# --------------------------------
# Model Inference Result Check
# --------------------------------
data_dir = "datamart"
gold_model_predictions_directory = os.path.join(data_dir, "gold", "model_predictions")
if not os.path.exists(gold_model_predictions_directory):
    print(f"Gold Model Predictions Directory '{gold_model_predictions_directory}' does not exist. Please check the data source.")
    raise SystemExit("Exiting the program due to missing gold model predictions directory.")
print(f"Gold Model Predictions Directory '{gold_model_predictions_directory}' exists.")

print("Gold model predictions directory checks passed successfully.")