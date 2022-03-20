# Stock Indicator

## Services

### Airflow

```
mkdir -p ./dags ./logs ./plugins
echo -e "AIRFLOW_UID=$(id -u)" > .env

# Initialize the database
docker-compose up airflow-init

# Start airflow
docker-compose up
```
