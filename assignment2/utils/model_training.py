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
from scipy.stats import randint, uniform, loguniform

from pyspark.sql.functions import col, to_date
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
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
    model_train_date_str = snapshotdate
    train_test_period_months = 12
    oot_period_months = 2
    train_test_ratio = 0.8

    config = {}
    config["model_train_date_str"] = model_train_date_str
    config["train_test_period_months"] = train_test_period_months
    config["oot_period_months"] =  oot_period_months
    config["model_train_date"] =  datetime.strptime(model_train_date_str, "%Y-%m-%d")
    config["oot_end_date"] =  config['model_train_date'] - timedelta(days = 1)
    config["oot_start_date"] =  config['model_train_date'] - relativedelta(months = oot_period_months)
    config["train_test_end_date"] =  config["oot_start_date"] - timedelta(days = 1)
    config["train_test_start_date"] =  config["oot_start_date"] - relativedelta(months = train_test_period_months)
    config["train_test_ratio"] = train_test_ratio 
    
    pprint.pprint(config)
    
    # connect to label store
    folder_path = "datamart/gold/label_store/"
    files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    label_store_sdf = spark.read.option("header", "true").parquet(*files_list)
    labels_sdf = label_store_sdf.filter((col("snapshot_date") >= config["train_test_start_date"]) & (col("snapshot_date") <= config["oot_end_date"]))

    # connect to feature store
    folder_path_1 = "datamart/gold/feature_store/eng/"
    folder_path_2 = "datamart/gold/feature_store/cust_fin_risk/"
    files_list_1 = [folder_path_1+os.path.basename(f) for f in glob.glob(os.path.join(folder_path_1, '*'))]
    files_list_2 = [folder_path_2+os.path.basename(f) for f in glob.glob(os.path.join(folder_path_2, '*'))]
    feature_store_sdf_1 = spark.read.option("header", "true").parquet(*files_list_1)
    feature_store_sdf_2 = spark.read.option("header", "true").parquet(*files_list_2)

    # extract label store
    labels_sdf = label_store_sdf.filter((col("snapshot_date") >= config["train_test_start_date"]) & (col("snapshot_date") <= config["oot_end_date"]))

    print("extracted labels_sdf", labels_sdf.count(), config["train_test_start_date"], config["oot_end_date"])

    # extract feature store
    features_sdf_1 = feature_store_sdf_1.filter((col("snapshot_date") >= config["train_test_start_date"]) & (col("snapshot_date") <= config["oot_end_date"]))
    features_sdf_2 = feature_store_sdf_2

    print("extracted features_sdf_1", features_sdf_1.count(), config["train_test_start_date"], config["oot_end_date"])
    print("extracted features_sdf_2", features_sdf_2.count(), config["train_test_start_date"], config["oot_end_date"])

    # prepare data for modeling by joining tables
    data_pdf_temp = labels_sdf.join(features_sdf_1, on=["Customer_ID", "snapshot_date"], how="left")
    features_sdf_2 = features_sdf_2.drop('snapshot_date')
    data_pdf = data_pdf_temp.join(features_sdf_2, on=["Customer_ID"], how="left").toPandas()

    # rename features
    columns_to_exclude = ['Customer_ID', 'snapshot_date', 'loan_id', 'label', 'label_def']
    columns_to_rename = [col for col in data_pdf.columns if col not in columns_to_exclude]
    rename_dict = {col: 'feature_' + col for col in columns_to_rename}
    data_pdf.rename(columns=rename_dict, inplace=True)

    oot_pdf = data_pdf[(data_pdf['snapshot_date'] >= config["oot_start_date"].date()) & (data_pdf['snapshot_date'] <= config["oot_end_date"].date())]
    train_test_pdf = data_pdf[(data_pdf['snapshot_date'] >= config["train_test_start_date"].date()) & (data_pdf['snapshot_date'] <= config["train_test_end_date"].date())]

    feature_cols = [fe_col for fe_col in data_pdf.columns if fe_col.startswith('feature_')]

    X_oot = oot_pdf[feature_cols]
    y_oot = oot_pdf["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        train_test_pdf[feature_cols], train_test_pdf["label"], 
        test_size= 1 - config["train_test_ratio"],
        random_state=88,     # Ensures reproducibility
        shuffle=True,        # Shuffle the data before splitting
        stratify=train_test_pdf["label"]           # Stratify based on the label column
    )


    # set up standard scalar preprocessing
    scaler = StandardScaler()

    transformer_stdscaler = scaler.fit(X_train) # for standardisation, we should use training set, to prevent data leakage

    # transform data
    X_train_processed = transformer_stdscaler.transform(X_train)
    X_test_processed = transformer_stdscaler.transform(X_test)
    X_oot_processed = transformer_stdscaler.transform(X_oot)

    # Define the Random Forest classifier
    rf_clf = RandomForestClassifier( random_state=88)

    # Define the hyperparameter space to search
    rf_param_dist = {
        'n_estimators': randint(50, 500),
        'max_depth': randint(3, 20),
        'max_features': ['sqrt', 'log2', 0.6, 0.8],
        'min_samples_leaf': randint(1, 10),
        'min_samples_split': randint(2, 20),
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }

    # Create a scorer based on AUC score
    auc_scorer = make_scorer(roc_auc_score)

    # Set up the random search with cross-validation
    rf_random_search = RandomizedSearchCV(
        estimator=rf_clf,
        param_distributions=rf_param_dist,
        scoring=auc_scorer,
        n_iter=100,  # Number of iterations for random search
        cv=5,       # Number of folds in cross-validation
        verbose=1,
        random_state=42,
        n_jobs=-1   # Use all available cores
    )

    # Perform the random search
    rf_random_search.fit(X_train_processed, y_train)


    # Evaluate the model on the train set
    rf_best_model = rf_random_search.best_estimator_
    y_pred_proba = rf_best_model.predict_proba(X_train_processed)[:, 1]
    rf_train_auc_score = roc_auc_score(y_train, y_pred_proba)

    # Evaluate the model on the test set
    rf_best_model = rf_random_search.best_estimator_
    y_pred_proba = rf_best_model.predict_proba(X_test_processed)[:, 1]
    rf_test_auc_score = roc_auc_score(y_test, y_pred_proba)

    # Evaluate the model on the oot set
    rf_best_model = rf_random_search.best_estimator_
    y_pred_proba = rf_best_model.predict_proba(X_oot_processed)[:, 1]
    rf_oot_auc_score = roc_auc_score(y_oot, y_pred_proba)


    # Define the XGBoost classifier
    xgb_clf = xgb.XGBClassifier(eval_metric='logloss', random_state=88)

    # Define the hyperparameter space to search
    xgb_param_dist = {
        'n_estimators': randint(25, 101),  
        'max_depth': randint(2, 6),       
        'learning_rate': loguniform(0.01, 0.2),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 0.2),
        'min_child_weight': randint(1, 6),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(1, 1.5) 
    }
    # Create a scorer based on AUC score
    auc_scorer = make_scorer(roc_auc_score)

    # Set up the random search with cross-validation
    xgb_random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=xgb_param_dist,
        scoring=auc_scorer,
        n_iter=100,  # Number of iterations for random search
        cv=10,       # Number of folds in cross-validation
        verbose=1,
        random_state=42,
        n_jobs=-1   # Use all available cores
    )

    # Perform the random search
    xgb_random_search.fit(X_train_processed, y_train)

    # Evaluate the model on the train set
    xgb_best_model = xgb_random_search.best_estimator_
    y_pred_proba = xgb_best_model.predict_proba(X_train_processed)[:, 1]
    xgb_train_auc_score = roc_auc_score(y_train, y_pred_proba)

    # Evaluate the model on the test set
    xgb_best_model = xgb_random_search.best_estimator_
    y_pred_proba = xgb_best_model.predict_proba(X_test_processed)[:, 1]
    xgb_test_auc_score = roc_auc_score(y_test, y_pred_proba)

    # Evaluate the model on the oot set
    xgb_best_model = xgb_random_search.best_estimator_
    y_pred_proba = xgb_best_model.predict_proba(X_oot_processed)[:, 1]
    xgb_oot_auc_score = roc_auc_score(y_oot, y_pred_proba)


    model_artefact = {}

    # create model_bank dir
    model_bank_directory = "model_bank/"

    if not os.path.exists(model_bank_directory):
        os.makedirs(model_bank_directory)

    if rf_test_auc_score >= xgb_test_auc_score:
        model_artefact['model'] = rf_best_model
        model_artefact['model_version'] = "credit_model_"+config["model_train_date_str"].replace('-','_')
        model_artefact['preprocessing_transformers'] = {}
        model_artefact['preprocessing_transformers']['stdscaler'] = transformer_stdscaler
        model_artefact['data_dates'] = config
        model_artefact['data_stats'] = {}
        model_artefact['data_stats']['X_train'] = X_train.shape[0]
        model_artefact['data_stats']['X_test'] = X_test.shape[0]
        model_artefact['data_stats']['X_oot'] = X_oot.shape[0]
        model_artefact['data_stats']['y_train'] = round(y_train.mean(),2)
        model_artefact['data_stats']['y_test'] = round(y_test.mean(),2)
        model_artefact['data_stats']['y_oot'] = round(y_oot.mean(),2)
        model_artefact['results'] = {}
        model_artefact['results']['auc_train'] = rf_train_auc_score
        model_artefact['results']['auc_test'] = rf_test_auc_score
        model_artefact['results']['auc_oot'] = rf_oot_auc_score
        model_artefact['results']['gini_train'] = round(2*rf_train_auc_score-1,3)
        model_artefact['results']['gini_test'] = round(2*rf_test_auc_score-1,3)
        model_artefact['results']['gini_oot'] = round(2*rf_oot_auc_score-1,3)
        model_artefact['hp_params'] = rf_random_search.best_params_
        file_path = os.path.join(model_bank_directory, model_artefact['model_version'] + '.pkl')

        # Write the model to a pickle file
        with open(file_path, 'wb') as file:
            pickle.dump(model_artefact, file)
    else:
        model_artefact['model'] = xgb_best_model
        model_artefact['model_version'] = "credit_model_"+config["model_train_date_str"].replace('-','_')
        model_artefact['preprocessing_transformers'] = {}
        model_artefact['preprocessing_transformers']['stdscaler'] = transformer_stdscaler
        model_artefact['data_dates'] = config
        model_artefact['data_stats'] = {}
        model_artefact['data_stats']['X_train'] = X_train.shape[0]
        model_artefact['data_stats']['X_test'] = X_test.shape[0]
        model_artefact['data_stats']['X_oot'] = X_oot.shape[0]
        model_artefact['data_stats']['y_train'] = round(y_train.mean(),2)
        model_artefact['data_stats']['y_test'] = round(y_test.mean(),2)
        model_artefact['data_stats']['y_oot'] = round(y_oot.mean(),2)
        model_artefact['results'] = {}
        model_artefact['results']['auc_train'] = xgb_train_auc_score
        model_artefact['results']['auc_test'] = xgb_test_auc_score
        model_artefact['results']['auc_oot'] = xgb_oot_auc_score
        model_artefact['results']['gini_train'] = round(2*xgb_train_auc_score-1,3)
        model_artefact['results']['gini_test'] = round(2*xgb_test_auc_score-1,3)
        model_artefact['results']['gini_oot'] = round(2*xgb_oot_auc_score-1,3)
        model_artefact['hp_params'] = xgb_random_search.best_params_
        file_path = os.path.join(model_bank_directory, model_artefact['model_version'] + '.pkl')

        # Write the model to a pickle file
        with open(file_path, 'wb') as file:
            pickle.dump(model_artefact, file)
        
    print(f"Model saved to {file_path}")
    return None
