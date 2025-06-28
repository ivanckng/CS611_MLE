import os
from datetime import date, datetime, timedelta
import pandas as pd
import argparse


# --------------------------------
# Parse Command Line Arguments
# --------------------------------
parser = argparse.ArgumentParser(description="Data pipeline start script for feature source checks.")
parser.add_argument('--snapshotdate', type=str, help='The snapshot date for the feature source checks (YYYY-MM-DD).')
args = parser.parse_args()
current_date_str = args.snapshotdate
current_date = datetime.strptime(current_date_str, '%Y-%m-%d').date()

# --------------------------------
# Data Source Directory Check
# --------------------------------
data_dir = "data"

print(f"Checking Data Source Directory: {data_dir}...")
if not os.path.exists(data_dir):
    print(f"Data Source Directory '{data_dir}' does not exist. Please check the data source path.")
    raise SystemExit("Exiting the program due to missing data source directory.")
print(f"Data Source Directory '{data_dir}' exists.")


# --------------------------------
# Feature Data Source Check
# --------------------------------
print("Checking Feature Data Sources...")
feature_files = {
    "Clickstream": "feature_clickstream.csv",
    "Attributes": "features_attributes.csv",
    "Financials": "features_financials.csv"
}

for feature_name, file_name in feature_files.items():
    feature_csv_path = os.path.join(data_dir, file_name)
    if not os.path.exists(feature_csv_path):
        print(f"Feature - {feature_name} Data Source '{feature_csv_path}' does not exist. Please check the data source.")
        raise SystemExit(f"Exiting the program due to missing Feature - {feature_name} data source.")
    print(f"Feature - {feature_name} Data Source '{feature_csv_path}' exists.")


click_csv_path = os.path.join(data_dir, "feature_clickstream.csv")
click_pdf = pd.read_csv(click_csv_path)
click_pdf['snapshot_date'] = pd.to_datetime(click_pdf['snapshot_date'])

if click_pdf['snapshot_date'].max().date() < current_date:
    print(f"Current date's feature data is not ready. Last available date is {click_pdf['snapshot_date'].max().date()}. Expected at least {current_date}.")
    raise SystemExit("Exiting the program due to missing Current date's feature data.")
else:
    print(f"Current date's feature data is ready. Last available date is {click_pdf['snapshot_date'].max().date()}.")

print("All feature data source checks passed successfully.")