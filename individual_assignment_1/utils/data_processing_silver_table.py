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
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_loan_silver_table(snapshot_date_str, bronze_loan_directory, silver_loan_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "loan_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_loan_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def process_clickstream_silver_table(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "fe_1": IntegerType(),
        "fe_2": IntegerType(),
        "fe_3": IntegerType(),
        "fe_4": IntegerType(),
        "fe_5": IntegerType(),
        "fe_6": IntegerType(),
        "fe_7": IntegerType(),
        "fe_8": IntegerType(),
        "fe_9": IntegerType(),
        "fe_10": IntegerType(),
        "fe_11": IntegerType(),
        "fe_12": IntegerType(),
        "fe_13": IntegerType(),
        "fe_14": IntegerType(),
        "fe_15": IntegerType(),
        "fe_16": IntegerType(),
        "fe_17": IntegerType(),
        "fe_18": IntegerType(),
        "fe_19": IntegerType(),
        "fe_20": IntegerType(),
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # save silver table - IRL connect to database to write
    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_attributes_silver_table(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attributes_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    df = df.withColumn('Occupation',F.when(df['Occupation'] == '_______', None).otherwise(df['Occupation']))
    
    ssn_p = r'\d+-\d+-\d+'
    extract_ssn = F.regexp_extract(col("SSN"), ssn_p, 0)
    df = df.withColumn("SSN",F.when(extract_ssn == "", F.lit(None)).otherwise(extract_ssn))

    name_p = r'(\w+\.*\s*\w*)'
    extract_name = F.regexp_extract(col("Name"), name_p, 0)
    df = df.withColumn("Name",F.when(extract_name == "", F.lit(None)).otherwise(extract_name))

    age_p = r'(\d+)'
    extract_age = F.regexp_extract(col("Age"), age_p, 0)
    df = df.withColumn("Age",F.when(extract_age == "", F.lit(np.nan)).otherwise(extract_age))

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "Customer_ID": StringType(),
        "snapshot_date": DateType()        
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))



    # save silver table - IRL connect to database to write
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_financials_silver_table(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_financials_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    income_p = r'(\d+\.*\d*)'
    extract_income = F.regexp_extract(col("Annual_Income"), income_p, 0)
    df = df.withColumn("Annual_Income",F.when(extract_income == "", F.lit(np.nan)).otherwise(extract_income))

    numloan_p = r'(\d+)'
    extract_numloan = F.regexp_extract(col("Num_of_Loan"), numloan_p, 0)
    df = df.withColumn("Num_of_Loan",F.when(extract_numloan == "", F.lit(np.nan)).otherwise(extract_numloan))
    
    df = df.withColumn('Type_of_Loan',F.when(df['Type_of_Loan'] == 'NaN', None).otherwise(df['Type_of_Loan']))

    delaypayment_p = r'(\d+)'
    extract_delaypayment = F.regexp_extract(col("Num_of_Delayed_Payment"), delaypayment_p, 0)
    df = df.withColumn("Num_of_Delayed_Payment",F.when(extract_delaypayment == "", F.lit(np.nan)).otherwise(extract_delaypayment))

    changedlimit_p = r'(\d+\.*\d*)'
    extract_changedlimit = F.regexp_extract(col("Changed_Credit_Limit"), changedlimit_p, 0)
    df = df.withColumn("Changed_Credit_Limit",F.when(extract_changedlimit == "", F.lit(np.nan)).otherwise(extract_changedlimit))

    df = df.withColumn('Credit_Mix',F.when(df['Credit_Mix'] == '_', None).otherwise(df['Credit_Mix']))

    od_p = r'(\d+\.*\d*)'
    extract_od = F.regexp_extract(col("Outstanding_Debt"), od_p, 0)
    df = df.withColumn("Outstanding_Debt",F.when(extract_od == "", F.lit(np.nan)).otherwise(extract_od))

    aim_p = r'(\d+\.*\d*)'
    extract_aim = F.regexp_extract(col("Amount_invested_monthly"), aim_p, 0)
    df = df.withColumn("Amount_invested_monthly",F.when(extract_aim == "", F.lit(np.nan)).otherwise(extract_aim))
    
    pb_p = r'([A-Za-z]+_[A-Za-z]+_[A-Za-z]+_[A-Za-z]+_[A-Za-z]+)'
    extract_pb = F.regexp_extract(col("Payment_Behaviour"), pb_p, 0)
    df = df.withColumn("Payment_Behaviour",F.when(extract_pb == "", F.lit(None)).otherwise(extract_pb))

    mb_p = r'(\d+\.*\d*)'
    extract_mb = F.regexp_extract(col("Monthly_Balance"), mb_p, 0)
    df = df.withColumn("Monthly_Balance",F.when(extract_mb == "", F.lit(np.nan)).otherwise(extract_mb))

    cha_p = r'(\d+) .* and (\d+) .*'
    extract_cha_year = F.regexp_extract(col("Credit_History_Age"), cha_p, 1)
    extract_cha_month = F.regexp_extract(col("Credit_History_Age"), cha_p, 2)
    df = df.withColumn("Credit_History_Year",F.when(extract_cha_year == "", F.lit(np.nan)).otherwise(extract_cha_year))
    df = df.withColumn("Credit_History_Month",F.when(extract_cha_month == "", F.lit(np.nan)).otherwise(extract_cha_month))

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": FloatType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": StringType(),
        "Credit_History_Year": IntegerType(),
        "Credit_History_Month": IntegerType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": FloatType()        
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # save silver table - IRL connect to database to write
    partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df