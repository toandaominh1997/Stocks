from datetime import timedelta

from airflow.models import DAG
from airflow.operators.papermill_operator import PapermillOperator
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago


default_args = {
    'owner': 'Airflow',
    'start_date': days_ago(2)
}

with DAG(
    dag_id='TS_Stock_papermill_operator',
    default_args=default_args,
    schedule_interval='0 0 * * *',
    dagrun_timeout=timedelta(minutes=3600)
) as dag:
    factory_task = BashOperator(task_id='print_hello', bash_command='echo "hello world!!!"')
    nb_task = PapermillOperator(
        task_id="training_TS",
        input_nb="/code/Stocks/modeling/TS.ipynb",
        output_nb="/home/airflow/Stocks/TS-{{ execution_date }}.ipynb",
        parameters={"msgs": "Ran from Airflow at {{ execution_date }}!"}
    )
    nb_test_task = PapermillOperator(
        task_id="testing",
        input_nb="/code/Stocks/modeling/TEST.ipynb",
        output_nb="/home/airflow/test-{{ execution_date }}.ipynb",
        parameters={"msgs": "Ran from Airflow at {{ execution_date }}!"}
    )
    ensemble_task = BashOperator(task_id='ensemble', bash_command='echo "Ensemble Task!!!"')

    # design graph
    factory_task>>nb_task
    factory_task>>nb_test_task
    nb_task>>ensemble_task
    nb_test_task>>ensemble_task
