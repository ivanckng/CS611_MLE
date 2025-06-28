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
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# to call this script: python model_train.py --snapshotdate "2024-09-01"

def main(snapshotdate, modelname):
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
    config["model_name"] = modelname
    config["model_bank_directory"] = "model_bank/"
    config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]
    
    pprint.pprint(config)
    

    # --- load model artefact from model bank ---
    # Load the model from the pickle file
    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = pickle.load(file)
    
    print("Model loaded successfully! " + config["model_artefact_filepath"])


    # --- load feature store ---
    folder_path_1 = "datamart/gold/feature_store/eng/"
    folder_path_2 = "datamart/gold/feature_store/cust_fin_risk/"
    files_list_1 = [folder_path_1+os.path.basename(f) for f in glob.glob(os.path.join(folder_path_1, '*'))]
    files_list_2 = [folder_path_2+os.path.basename(f) for f in glob.glob(os.path.join(folder_path_2, '*'))]
    
    # Load CSV into DataFrame - connect to feature store
    feature_store_sdf_1 = spark.read.option("header", "true").parquet(*files_list_1)
    feature_store_sdf_2 = spark.read.option("header", "true").parquet(*files_list_2)

    # Ensure snapshot_date is in date format
    feature_store_sdf_1 = feature_store_sdf_1.withColumn("snapshot_date", to_date(col("snapshot_date")))
    feature_store_sdf_2 = feature_store_sdf_2.withColumn("snapshot_date", to_date(col("snapshot_date")))

    # extract feature store
    features_sdf_1 = feature_store_sdf_1.filter(col("snapshot_date") == config["snapshot_date"])
    features_sdf_2 = feature_store_sdf_2
    features_sdf_2 = features_sdf_2.drop('snapshot_date')

    # join two feature tables
    features_sdf = features_sdf_1.join(features_sdf_2, on=["Customer_ID"], how="left")

    print("extracted features_sdf", features_sdf.count(), config["snapshot_date"])
    
    features_pdf = features_sdf.toPandas()
    columns_to_exclude = ['Customer_ID', 'snapshot_date']
    columns_to_rename = [col for col in features_pdf.columns if col not in columns_to_exclude]
    rename_dict = {col: 'feature_' + col for col in columns_to_rename}
    features_pdf.rename(columns=rename_dict, inplace=True)
    features_pdf.info()
    print("features_pdf:\n", features_pdf) 

    # --- preprocess data for modeling ---
    # prepare X_inference
    feature_cols = [fe_col for fe_col in features_pdf.columns if fe_col.startswith('feature_')]
    X_inference = features_pdf[feature_cols]
    print(X_inference)
    
    # apply transformer - standard scaler
    transformer_stdscaler = model_artefact["preprocessing_transformers"]["stdscaler"]
    X_inference = transformer_stdscaler.transform(X_inference)
    
    print('X_inference', X_inference.shape[0])


    # --- model prediction inference ---
    # load model
    model = model_artefact["model"]
    
    # predict model
    y_inference = model.predict_proba(X_inference)[:, 1]
    
    # prepare output
    y_inference_pdf = features_pdf[["Customer_ID","snapshot_date"]].copy()
    y_inference_pdf["model_name"] = config["model_name"]
    y_inference_pdf["model_predictions"] = y_inference
    

    # --- save model inference to datamart gold table ---
    # create bronze datalake
    gold_directory = f"datamart/gold/model_predictions/{config['model_name'][:-4]}/"
    print(gold_directory)
    
    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)
    
    # save gold table - IRL connect to database to write
    partition_name = config["model_name"][:-4] + "_predictions_" + config["snapshot_date_str"].replace('-','_') + '.parquet'
    filepath = gold_directory + partition_name
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return y_inference_pdf


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.modelname)
