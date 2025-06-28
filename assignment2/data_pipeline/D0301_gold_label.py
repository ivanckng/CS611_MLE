import sys
import os
sys.path.append('/opt/airflow/scripts')
import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, BooleanType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table

# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# --------------------------------
# Parse Command Line Arguments
# --------------------------------
parser = argparse.ArgumentParser(description="Gold Label Data Pipeline.")
parser.add_argument('--snapshotdate', type=str, help='The snapshot date for gold label data pipeline (YYYY-MM-DD).')
args = parser.parse_args()
current_date_str = args.snapshotdate
current_date = datetime.strptime(current_date_str, '%Y-%m-%d').date()


# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
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

# Config Directory
bronze_lms_directory = "datamart/bronze/lms/"
bronze_clks_directory = "datamart/bronze/clks/"
bronze_attr_directory = "datamart/bronze/attr/"
bronze_fin_directory = "datamart/bronze/fin/"
silver_lms_directory = "datamart/silver/lms/"
silver_clks_directory = "datamart/silver/clks/"
silver_attr_directory = "datamart/silver/attr/"
silver_fin_directory = "datamart/silver/fin/"
gold_clks_directory = "datamart/gold/feature_store/eng/"
gold_fin_directory = "datamart/gold/feature_store/cust_fin_risk/"
gold_label_store_directory = "datamart/gold/label_store/"

# ---------------------- 
# Build Gold Tables
# ---------------------- 

# Build Label Store (based on Loan Mgmt System Data)
gold_label_store_directory = "datamart/gold/label_store/"

if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

# run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_lms_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)

# Inspect Label Store 
folder_path = gold_label_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())
df.show()

# ---------------------- 
# Stop Spark Session
# ---------------------- 

spark.stop()
print("Script finished execution. Stop Spark session.")
