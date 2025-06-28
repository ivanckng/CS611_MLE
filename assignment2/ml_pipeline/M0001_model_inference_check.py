import os
from datetime import date, datetime, timedelta
import pandas as pd
import argparse


# --------------------------------
# Parse Command Line Arguments
# --------------------------------
parser = argparse.ArgumentParser(description="Model Inference Check.")
parser.add_argument('--snapshotdate', type=str, help='The snapshot date for model inference check (YYYY-MM-DD).')
args = parser.parse_args()
current_date_str = args.snapshotdate
current_date = datetime.strptime(current_date_str, '%Y-%m-%d').date()

# --------------------------------
# Model Directory Check
# --------------------------------
model_dir = "model_bank"

print(f"Checking Model Directory: {model_dir}...")
if not os.path.exists(model_dir):
    print(f"Model Directory '{model_dir}' does not exist. Please check the model directory path.")
    raise SystemExit("Exiting the program due to missing model directory.")
print(f"Model Directory '{model_dir}' exists.")

# --------------------------------
# Model Source Check
# --------------------------------
print("Checking Model Source...")
model_pkl_path = os.path.join(model_dir, "credit_model_2024_09_01.pkl")
if not os.path.exists(model_pkl_path):
    print(f"Model '{model_pkl_path}' does not exist. Please check the model source.")
    raise SystemExit("Exiting the program due to missing model source.")
print(f"Model '{model_pkl_path}' exists.")