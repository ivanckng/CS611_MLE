from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2024, 7, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
) as dag:

    # data pipeline

    # --- label store ---

    check_source_label_data = BashOperator(
        task_id='check_source_label_data',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D0001_label_source_check.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )


    bronze_label_store = BashOperator(
        task_id='bronze_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D0101_bronze_label.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_label_store = BashOperator(
        task_id='silver_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D0201_silver_label.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    gold_label_store = BashOperator(
        task_id='gold_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D0301_gold_label.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    label_store_completion_check = BashOperator(
        task_id='label_store_completion_check',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D0401_label_store_check.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )


    # Define task dependencies to run scripts sequentially
    check_source_label_data >> bronze_label_store >> silver_label_store >> gold_label_store >> label_store_completion_check
 
 
    # --- feature store ---

    check_source_feature_data = BashOperator(
        task_id='check_source_feature_data',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D0002_feature_source_check.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )


    bronze_table_clickstream = BashOperator(
        task_id='bronze_table_clickstream',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D0102_bronze_clickstream.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )
    
    bronze_table_attributes = BashOperator(
        task_id='bronze_table_attributes',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D0103_bronze_attributes.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    bronze_table_financials = BashOperator(
        task_id='bronze_table_financials',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D0104_bronze_financials.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_table_clickstream = BashOperator(
        task_id='silver_table_clickstream',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D0202_silver_clickstream.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_table_attributes = BashOperator(
        task_id='silver_table_attributes',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D0203_silver_attributes.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_table_financials = BashOperator(
        task_id='silver_table_financials',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D0204_silver_financials.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    gold_feature_engagment = BashOperator(
        task_id='gold_feature_engagment',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D0302_gold_engagement.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    gold_feature_finrisk = BashOperator(
        task_id='gold_feature_finrisk',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D0303_gold_finrisk.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    feature_store_completion_check = BashOperator(
        task_id='feature_store_completion_check',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D0402_feature_store_check.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )
    
    # Define task dependencies to run scripts sequentially
    check_source_feature_data >> bronze_table_clickstream >> silver_table_clickstream >> gold_feature_engagment
    check_source_feature_data >> bronze_table_financials >> silver_table_financials >> gold_feature_finrisk
    check_source_feature_data >> bronze_table_attributes >> silver_table_attributes
    gold_feature_engagment >> feature_store_completion_check
    gold_feature_finrisk >> feature_store_completion_check


    # --- model inference ---
    model_inference_start = BashOperator(
        task_id='model_inference_start',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 ml_pipeline/M0001_model_inference_check.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    model_inference = BashOperator(
        task_id='model_inference',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 ml_pipeline/M0002_model_inference.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )


    model_inference_completion = BashOperator(
        task_id='model_inference_completion',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 ml_pipeline/M0003_model_inference_completion.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )
    
    # # Define task dependencies to run scripts sequentially
    feature_store_completion_check >> model_inference_start
    model_inference_start >> model_inference >> model_inference_completion



    # # --- model monitoring ---

    model_monitor = BashOperator(
        task_id='model_monitor',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 ml_pipeline/M0005_model_monitor.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    model_monitor_completion = BashOperator(
        task_id='model_monitor_completion',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 ml_pipeline/M0006_model_monitor_completion.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )
    
    # # Define task dependencies to run scripts sequentially
    model_inference_completion >> model_monitor >> model_monitor_completion