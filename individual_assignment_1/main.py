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
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

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

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
clickstream_end_date_str = "2024-12-01"
attributes_end_date_str = "2025-01-01"
financials_end_date_str = "2025-01-01"
loan_end_date_str = "2025-11-01"

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

clickstream_dates_str_lst = generate_first_of_month_dates(start_date_str, clickstream_end_date_str)
attributes_dates_str_lst = generate_first_of_month_dates(start_date_str, attributes_end_date_str)
financials_dates_str_lst = generate_first_of_month_dates(start_date_str, financials_end_date_str)
loan_dates_str_lst = generate_first_of_month_dates(start_date_str, loan_end_date_str)

print(clickstream_dates_str_lst)
print(attributes_dates_str_lst)
print(financials_dates_str_lst)
print(loan_dates_str_lst)


# csv path
clickstream_path = 'data/feature_clickstream.csv'
attributes_path = 'data/features_attributes.csv'
financials_path = 'data/features_financials.csv'
loan_path = 'data/lms_loan_daily.csv'


# create bronze datalake
bronze_loan_directory = "datamart_assignment_1/bronze/loan_lms/"
bronze_clickstream_directory = "datamart_assignment_1/bronze/clickstream/"
bronze_attributes_directory = "datamart_assignment_1/bronze/attributes/"
bronze_financials_directory = "datamart_assignment_1/bronze/financials/"

if not os.path.exists(bronze_loan_directory):
    os.makedirs(bronze_loan_directory)

if not os.path.exists(bronze_clickstream_directory):
    os.makedirs(bronze_clickstream_directory)

if not os.path.exists(bronze_attributes_directory):
    os.makedirs(bronze_attributes_directory)

if not os.path.exists(bronze_financials_directory):
    os.makedirs(bronze_financials_directory)

# run bronze backfill
for date_str in clickstream_dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_clickstream_directory, spark, clickstream_path, 'clickstream')
for date_str in attributes_dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_attributes_directory, spark, attributes_path, 'attributes')
for date_str in financials_dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_financials_directory, spark, financials_path, 'financials')
for date_str in loan_dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_loan_directory, spark, loan_path, 'loan')


# create silver datalake
silver_loan_directory = "datamart_assignment_1/silver/loan_lms/"
silver_clickstream_directory = "datamart_assignment_1/silver/clickstream/"
silver_attributes_directory = "datamart_assignment_1/silver/attributes/"
silver_financials_directory = "datamart_assignment_1/silver/financials/"

if not os.path.exists(silver_loan_directory):
    os.makedirs(silver_loan_directory)

if not os.path.exists(silver_clickstream_directory):
    os.makedirs(silver_clickstream_directory)

if not os.path.exists(silver_attributes_directory):
    os.makedirs(silver_attributes_directory)

if not os.path.exists(silver_financials_directory):
    os.makedirs(silver_financials_directory)

# run silver backfill to
for date_str in loan_dates_str_lst:
    utils.data_processing_silver_table.process_loan_silver_table(date_str, bronze_loan_directory, silver_loan_directory, spark)
for date_str in clickstream_dates_str_lst:
    utils.data_processing_silver_table.process_clickstream_silver_table(date_str, bronze_clickstream_directory, silver_clickstream_directory, spark)
for date_str in attributes_dates_str_lst:
    utils.data_processing_silver_table.process_attributes_silver_table(date_str, bronze_attributes_directory, silver_attributes_directory, spark)
for date_str in financials_dates_str_lst:
    utils.data_processing_silver_table.process_financials_silver_table(date_str, bronze_financials_directory, silver_financials_directory, spark)


# create gold datalake
gold_label_store_directory = "datamart_assignment_1/gold/label_store/"
gold_clickstream_store_directory = "datamart_assignment_1/gold/clickstream_store/"
gold_attributes_store_directory = "datamart_assignment_1/gold/attributes_store/"
gold_financials_store_directory = "datamart_assignment_1/gold/financials_store/"
gold_attr_fin_store_directory = "datamart_assignment_1/gold/attr_fin_store/"

if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

if not os.path.exists(gold_clickstream_store_directory):
    os.makedirs(gold_clickstream_store_directory)

if not os.path.exists(gold_attributes_store_directory):
    os.makedirs(gold_attributes_store_directory)

if not os.path.exists(gold_financials_store_directory):
    os.makedirs(gold_financials_store_directory)

if not os.path.exists(gold_attr_fin_store_directory):
    os.makedirs(gold_attr_fin_store_directory)

if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

# run gold backfill
for date_str in loan_dates_str_lst:
    utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_loan_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)
for date_str in clickstream_dates_str_lst:
    utils.data_processing_gold_table.process_clickstream_gold_table(date_str, silver_clickstream_directory, gold_clickstream_store_directory, spark)
for date_str in attributes_dates_str_lst:
    utils.data_processing_gold_table.process_attributes_gold_table(date_str, silver_attributes_directory, gold_attributes_store_directory, spark)
for date_str in financials_dates_str_lst:
    utils.data_processing_gold_table.process_financials_gold_table(date_str, silver_financials_directory, gold_financials_store_directory, spark)
for date_str in financials_dates_str_lst:
    utils.data_processing_gold_table.process_attr_fin_gold_table(date_str, gold_attributes_store_directory, gold_financials_store_directory, gold_attr_fin_store_directory, spark)



# print gold datalake
folder_path = gold_label_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
label_df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",label_df.count())

label_df.show()
label_df.printSchema()

folder_path = gold_clickstream_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
clickstream_df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",clickstream_df.count())

clickstream_df.show()
clickstream_df.printSchema()

folder_path = gold_attributes_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
attributes_df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",attributes_df.count())

attributes_df.show()
attributes_df.printSchema()

folder_path = gold_financials_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
financials_df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",financials_df.count())

financials_df.show()
financials_df.printSchema()

folder_path = gold_attr_fin_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
attr_fin_df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",attr_fin_df.count())

attr_fin_df.show()
attr_fin_df.printSchema()



    