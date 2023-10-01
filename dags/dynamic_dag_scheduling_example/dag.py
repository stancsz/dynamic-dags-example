import requests
import pandas as pd
from sklearn.externals import joblib
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.hooks.base_hook import BaseHook
from datetime import datetime
import boto3
import fastavro
from io import BytesIO
import yaml

# Load the configuration
with open('config.yml', 'r') as file:
    dag_configs = yaml.safe_load(file)['dags']

# Initialize a session using Amazon S3
s3 = boto3.session.Session().client('s3')

# Get API credentials from Airflow connections
marketing_conn = BaseHook.get_connection('marketing_api')
marketing_api_key = marketing_conn.password

weather_conn = BaseHook.get_connection('weather_api')
weather_api_key = weather_conn.password

# Define Avro schema
schema = {
    "namespace": "example.avro",
    "type": "record",
    "name": "Data",
    "fields": [
        {"name": "date", "type": "string"},
        # ... other field definitions ...
    ]
}

def create_dag(dag_id, city_id, s3_path, default_args):
    dag = DAG(
        dag_id=dag_id,
        default_args=default_args,
        description=f'A DAG to fetch data and make predictions for {dag_id}',
        schedule_interval='@daily',
    )

    def fetch_marketing_data():
        response = requests.get('https://api.hypotheticalmarketingdata.com/data', headers={'Authorization': f'Bearer {marketing_api_key}'})
        response.raise_for_status()
        marketing_data = response.json()
        df = pd.DataFrame(marketing_data)
        buffer = BytesIO()
        fastavro.writer(buffer, schema, df.to_dict('records'))
        s3.put_object(Bucket=s3_path, Key=f'{city_id}/marketing_data.avro', Body=buffer.getvalue())

    def fetch_weather_data():
        response = requests.get(f'http://api.openweathermap.org/data/2.5/weather?id={city_id}&appid={weather_api_key}')
        response.raise_for_status()
        weather_data = response.json()
        df = pd.DataFrame([weather_data])
        buffer = BytesIO()
        fastavro.writer(buffer, schema, df.to_dict('records'))
        s3.put_object(Bucket=s3_path, Key=f'{city_id}/weather_data.avro', Body=buffer.getvalue())

    def make_prediction():
        marketing_response = s3.get_object(Bucket=s3_path, Key=f'{city_id}/marketing_data.avro')
        weather_response = s3.get_object(Bucket=s3_path, Key=f'{city_id}/weather_data.avro')
        marketing_data = pd.DataFrame(fastavro.reader(BytesIO(marketing_response['Body'].read())))
        weather_data = pd.DataFrame(fastavro.reader(BytesIO(weather_response['Body'].read())))
        data = pd.merge(marketing_data, weather_data, on='date')
        model = joblib.load('model.pkl')
        prediction = model.predict(data)
        prediction_df = pd.DataFrame({'date': data['date'], 'prediction': prediction})
        buffer = BytesIO()
        fastavro.writer(buffer, schema, prediction_df.to_dict('records'))
        s3.put_object(Bucket=s3_path, Key=f'{city_id}/prediction.avro', Body=buffer.getvalue())

    fetch_marketing_task = PythonOperator(
        task_id='fetch_marketing_data',
        python_callable=fetch_marketing_data,
        dag=dag,
    )

    fetch_weather_task = PythonOperator(
        task_id='fetch_weather_data',
        python_callable=fetch_weather_data,
        dag=dag,
    )

    make_prediction_task = PythonOperator(
        task_id='make_prediction',
        python_callable=make_prediction,
        dag=dag,
    )

    fetch_marketing_task >> fetch_weather_task >> make_prediction_task

    return dag

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

# Dynamically generate DAGs based on the configuration file
for dag_config in dag_configs:
    city_name = dag_config['city_name']
    city_id = dag_config['city_id']
    s3_path = dag_config['s3_path']
    dag_id = f'marketing_weather_prediction_{city_name}'
    globals()[dag_id] = create_dag(dag_id, city_id, s3_path, default_args)