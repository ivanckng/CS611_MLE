import os
from datetime import date, datetime, timedelta
import pandas as pd
import argparse


# --------------------------------
# Parse Command Line Arguments
# --------------------------------
parser = argparse.ArgumentParser(description="Feature Store Check.")
parser.add_argument('--snapshotdate', type=str, help='The snapshot date for feature store check (YYYY-MM-DD).')
args = parser.parse_args()
current_date_str = args.snapshotdate
current_date = datetime.strptime(current_date_str, '%Y-%m-%d').date()

# --------------------------------
# Data Source Directory Check
# --------------------------------
data_dir = "datamart"

print(f"Checking Data Source Directory: {data_dir}...")
if not os.path.exists(data_dir):
    print(f"Data Source Directory '{data_dir}' does not exist. Please check the data source path.")
    raise SystemExit("Exiting the program due to missing data source directory.")
print(f"Data Source Directory '{data_dir}' exists.")

# --------------------------------
# Gold Feature Data Store Check
# --------------------------------
print("Checking Gold Feature Data Store...")
gold_feature_store_directory = os.path.join(data_dir, "gold", "feature_store")
if not os.path.exists(gold_feature_store_directory):
    print(f"Gold Feature Data Store '{gold_feature_store_directory}' does not exist. Please check the data source.")
    raise SystemExit("Exiting the program due to missing gold feature data store.")
print(f"Gold Feature Data Store '{gold_feature_store_directory}' exists.")

print("Gold feature data store checks passed successfully.")