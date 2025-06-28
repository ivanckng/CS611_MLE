import os
from datetime import date, datetime, timedelta
import pandas as pd
import argparse


# --------------------------------
# Parse Command Line Arguments
# --------------------------------
parser = argparse.ArgumentParser(description="Data pipeline start script for label source checks.")
parser.add_argument('--snapshotdate', type=str, help='The snapshot date for the label source checks (YYYY-MM-DD).')
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
# Label Data Source Check
# --------------------------------
print("Checking Label Data Source...")
label_csv_path = os.path.join(data_dir, "lms_loan_daily.csv")
if not os.path.exists(label_csv_path):
    print(f"Label Data Source '{label_csv_path}' does not exist. Please check the data source.")
    raise SystemExit("Exiting the program due to missing label data source.")
print(f"Label Data Source '{label_csv_path}' exists.")


# assuming the business t-1 timeliness
# check if current date's data is ready
label_pdf = pd.read_csv(label_csv_path)
label_pdf['snapshot_date'] = pd.to_datetime(label_pdf['snapshot_date'])

if label_pdf['snapshot_date'].max().date() < current_date:
    print(f"Current date's data is not ready. Last available date is {label_pdf['snapshot_date'].max().date()}. Expected at least {current_date}.")
    raise SystemExit("Exiting the program due to missing Current date's data.")
else:
    print(f"Current date's data is ready. Last available date is {label_pdf['snapshot_date'].max().date()}.")

print("All label data source checks passed successfully.")