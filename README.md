# Airflow for prediction
# Descripción
Predecir el precio del BTC cada minuto 

### Instalación
Instalar:
```
    ● Docker
    ● Docker-compose
    ● Python
```
### Levantar plataforma de airflow:
```
    ● cd [Proy-dir]   
    ● mkdir -p ./plugins ./dags ./logs .dags/files/prediciones
    ● echo -e "AIRFLOW_UID=$(id -u)\nAIRFLOW_GID=0" > .env 
    ● docker-compose up airflow-init (esperar el tiempo necesario)
    ● docker-compose up -d  (esperrar el tiempo necesario)
    ● Localhost:8080 
```
### Login :
```
    ● user:       airflow
    ● password:   airflow

```


