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
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType


def process_labels_gold_table(snapshot_date_str, silver_loan_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "silver_loan_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df



def process_clickstream_gold_table(snapshot_date_str, silver_clickstream_directory, gold_clickstream_store_directory, spark):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # select columns to save
    df = df.select("Customer_ID", "snapshot_date", "fe_1", "fe_2", "fe_3", "fe_4", "fe_5", "fe_6", "fe_7", "fe_8", "fe_9", "fe_10", "fe_11", "fe_12", "fe_13", "fe_14", "fe_15", "fe_16", "fe_17", "fe_18", "fe_19", "fe_20")

    # save gold table - IRL connect to database to write
    partition_name = "gold_clickstream_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_clickstream_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_attributes_gold_table(snapshot_date_str, silver_attributes_directory, gold_attributes_store_directory, spark):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())


    # check condition
    condition_age = (F.col("Age") < 21) | (F.col("Age") > 65)

    # replace outliers with median age
    df = df.withColumn("Age", F.when(condition_age, np.nan).otherwise(F.col("Age")))

    # select columns to save: SSN and Name is for identification purposes not a useful feature, and for further table join, customer_id is the key
    df = df.select("Customer_ID", "snapshot_date", "Age", "Occupation")
    # save gold table - IRL connect to database to write
    partition_name = "gold_attributes_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_attributes_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_financials_gold_table(snapshot_date_str, silver_financials_directory, gold_financials_store_directory, spark):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # check condition
    condition_nba = F.col("Num_Bank_Accounts") < 0
    condition_ddd = F.col("Delay_from_due_date") < 0 

    # replace outliers with median value
    df = df.withColumn("Num_Bank_Accounts", F.when(condition_nba, np.nan).otherwise(F.col("Num_Bank_Accounts")))
    df = df.withColumn("Delay_from_due_date", F.when(condition_ddd, np.nan).otherwise(F.col("Delay_from_due_date")))

    # split type of loan
    split_df = df.withColumn("split_loans", F.split(col('Type_of_Loan'), ', '))
    max_loan_types = split_df.select(F.size("split_loans").alias("size")).agg(F.max("size")).collect()[0][0]
    for i in range(max_loan_types):
        split_df = split_df.withColumn(f"Type_of_Loan_{i+1}", F.col("split_loans")[i])
    stopword = "and "
    for i in range(max_loan_types):
        split_df = split_df.withColumn(f"Type_of_Loan_{i+1}", F.regexp_replace(F.col(f"Type_of_Loan_{i+1}"), f"^{stopword}", ""))

    type_set = set()
    for i in range(max_loan_types):
        unique_loan_types = split_df.select(F.col(f"Type_of_Loan_{i+1}").alias("loan_type")).distinct().collect()
        type_set.update([row.loan_type for row in unique_loan_types])
    type_set.discard(None)
    type_set.discard('Not Specified')
    type_list = list(type_set)

    for loan_type in type_list:
        condition = None
        for i in range(max_loan_types):
            col_name = f"Type_of_Loan_{i+1}"
            if condition is None:
                condition = (F.col(col_name) == loan_type)
            else:
                condition = condition | (F.col(col_name) == loan_type)
        
        split_df = split_df.withColumn(
            loan_type,
            F.when(condition, 1).otherwise(0).cast(IntegerType())
        )

    temp_columns = ["split_loans"] + [f"Type_of_Loan_{i+1}" for i in range(max_loan_types)]

    df = split_df.select([col for col in split_df.columns if col not in temp_columns])
    df = df.drop("Type_of_Loan")

    # new features
    df = df.withColumn("Debt_to_Inhand_Income_Ratio_Monthly", F.col("Total_EMI_per_Month")/F.col("Monthly_Inhand_Salary"))
    df = df.withColumn("Balance_to_Income_Ratio_Monthly", F.col("Monthly_Balance")/F.col("Monthly_Inhand_Salary"))
    df = df.withColumn("Debt_to_Income_Rate_Annual", F.col("Outstanding_Debt")/F.col("Annual_Income"))

    df = df.select("Customer_ID", 
                   "snapshot_date",
                   "Annual_Income",
                   "Monthly_Inhand_Salary",
                   "Num_Bank_Accounts",
                   "Num_Credit_Card",
                   "Interest_Rate",
                   "Num_of_Loan",
                   "Delay_from_due_date",
                   "Num_of_Delayed_Payment",
                   "Changed_Credit_Limit",
                   "Num_Credit_Inquiries",
                   "Outstanding_Debt",
                   "Credit_Utilization_Ratio",
                   "Total_EMI_per_month",
                   "Amount_invested_monthly",
                   "Monthly_Balance",
                   "Debt_to_Inhand_Income_Ratio_Monthly",
                   "Balance_to_Income_Ratio_Monthly",
                   "Debt_to_Income_Rate_Annual",
                   "Credit_History_Year",
                   "Credit_History_Month",
                   "Credit_History_Age",
                   "Credit_Mix",
                   "Payment_of_Min_Amount",
                   "Payment_Behaviour",
                   "Debt Consolidation Loan",
                   "Personal Loan",
                   "Mortgage Loan",
                   "Auto Loan",
                   "Payday Loan",
                   "Credit-Builder Loan",
                   "Home Equity Loan",
                   "Student Loan")

    # save gold table - IRL connect to database to write
    partition_name = "gold_financials_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_financials_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def process_attr_fin_gold_table(snapshot_date_str, gold_attributes_store_directory, gold_financials_store_directory, gold_attr_fin_store_directory, spark):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to gold attributes table
    at_partition_name = "gold_attributes_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    at_filepath = gold_attributes_store_directory + at_partition_name
    at_df = spark.read.parquet(at_filepath)

    # connect to gold financials table
    fin_partition_name = "gold_financials_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    fin_filepath = gold_financials_store_directory + fin_partition_name
    fin_df = spark.read.parquet(fin_filepath)


    # join two tables
    df = at_df.join(fin_df, on=["Customer_ID", "snapshot_date"], how="left")

    # save gold table - IRL connect to database to write
    partition_name = "gold_attr_fin_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_attr_fin_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df