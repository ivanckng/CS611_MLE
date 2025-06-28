import argparse
import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col, to_date
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# to call this script: python model_train.py --snapshotdate "2024-09-01"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    
    # --- set up config ---
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")


    # --- load prediction result ---
    folder_path_1 = "datamart/gold/model_predictions/credit_model_2024_09_01/"
    files_list_1 = [folder_path_1+os.path.basename(f) for f in glob.glob(os.path.join(folder_path_1, '*'))]
    
    # Load CSV into DataFrame - connect to prediction store
    prediction_sdf = spark.read.parquet(*files_list_1)
    prediction_sdf.show()

    # Ensure snapshot_date is in date format
    prediction_sdf = prediction_sdf.withColumn("snapshot_date", to_date(col("snapshot_date")))

    # Filter the DataFrame for the specific snapshot_date
    prediction_sdf = prediction_sdf.filter(col("snapshot_date") == config["snapshot_date"])
    prediction_sdf = prediction_sdf.withColumn("predicted_label", F.when(col("model_predictions") >= 0.5, 1).otherwise(0))

    # --- load label  ---
    folder_path_2 = "datamart/gold/label_store/"
    files_list_2 = [folder_path_2+os.path.basename(f) for f in glob.glob(os.path.join(folder_path_2, '*'))]
    
    # Load CSV into DataFrame - connect to label store
    label_sdf = spark.read.option("header", "true").parquet(*files_list_2)

    # Ensure snapshot_date is in date format
    label_sdf = label_sdf.withColumn("snapshot_date", to_date(col("snapshot_date")))

    # Filter the DataFrame for the specific snapshot_date
    label_sdf = label_sdf.filter(col("snapshot_date") == config["snapshot_date"])
    

    # join two feature tables
    monitor_sdf = label_sdf.join(prediction_sdf, on=["Customer_ID"], how="left")
    monitor_pdf = monitor_sdf.toPandas()

    # evaluation metrics
    y_true = monitor_pdf['label']
    y_pred = monitor_pdf['predicted_label']
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, monitor_pdf['model_predictions'])
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    return accuracy, f1, roc_auc


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.modelname)
