import os
from datetime import date, datetime, timedelta
import pandas as pd
import argparse


# --------------------------------
# Parse Command Line Arguments
# --------------------------------
parser = argparse.ArgumentParser(description="Model Monitor Completion.")
parser.add_argument('--snapshotdate', type=str, help='The snapshot date for model monitor completion (YYYY-MM-DD).')
args = parser.parse_args()
current_date_str = args.snapshotdate
current_date = datetime.strptime(current_date_str, '%Y-%m-%d').date()

# --------------------------------
# Model Monitor Result Check
# --------------------------------
data_dir = "datamart"
gold_model_monitor_directory = os.path.join(data_dir, "gold", "model_monitor")
if not os.path.exists(gold_model_monitor_directory):
    print(f"Gold Model Monitor Directory '{gold_model_monitor_directory}' does not exist. Please check the data source.")
    raise SystemExit("Exiting the program due to missing gold model monitor directory.")
print(f"Gold Model Monitor Directory '{gold_model_monitor_directory}' exists.")

print("Gold model monitor directory checks passed successfully.")