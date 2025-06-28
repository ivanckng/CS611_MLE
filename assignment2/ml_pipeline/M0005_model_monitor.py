import sys
import os
sys.path.append('/opt/airflow/scripts')
import argparse
import os
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import utils.model_monitor

# --------------------------------
# Parse Command Line Arguments
# --------------------------------
parser = argparse.ArgumentParser(description="Model Monitor.")
parser.add_argument('--snapshotdate', type=str, help='The snapshot date for model monitor (YYYY-MM-DD).')
args = parser.parse_args()
current_date_str = args.snapshotdate
current_date = datetime.strptime(current_date_str, '%Y-%m-%d').date()


# set up config
start_date_str = "2023-07-01"
end_date_str = current_date_str


# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

accuracy_lst = []
f1_lst = []
roc_auc_lst = []

for snapshot_date in dates_str_lst:
    print(snapshot_date)
    accuracy, f1, roc_auc = utils.model_monitor.main(snapshot_date)
    accuracy_lst.append(accuracy)
    f1_lst.append(f1)
    roc_auc_lst.append(roc_auc)

monitoring_results = {
    "snapshot_date": dates_str_lst,
    "accuracy": accuracy_lst,
    "f1_score": f1_lst,
    "roc_auc_score": roc_auc_lst
}

monitoring_results_pdf = pd.DataFrame(monitoring_results)

# --- save model inference to datamart gold table ---
# create datalake
monitor_dir = f"datamart/gold/model_monitor/"
gold_csv_directory = f"datamart/gold/model_monitor/csv"
gold_plt_directory = f"datamart/gold/model_monitor/plt"

if not os.path.exists(monitor_dir):
    os.makedirs(monitor_dir)
if not os.path.exists(gold_csv_directory): 
    os.makedirs(gold_csv_directory)
if not os.path.exists(gold_plt_directory): 
    os.makedirs(gold_plt_directory)

# save gold table - IRL connect to database to write
csv_filepath = gold_csv_directory + current_date_str
plt_filepath = gold_plt_directory + current_date_str
monitoring_results_pdf.to_csv(csv_filepath + '.csv', index=False)
print('saved to:', csv_filepath)

def accuracy_plot(accuracy, date, plt_path):
    if len(accuracy) != len(date):
        raise ValueError("Accuracy list and Date list must have the same length.")

    plt.figure(figsize=(12, 6))
    plt.plot(date, accuracy, marker='o') 
    plt.title('Model Accuracy Over Time')
    plt.xlabel('Date')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.ylim(0, 1) 
    plt.xticks(rotation=45, ha='right') 
    plt.tight_layout() 
    plt.savefig(plt_path)
    plt.close()
    print(f"Accuracy plot saved to {plt_path}")

accuracy_plot(accuracy_lst, dates_str_lst, plt_filepath)