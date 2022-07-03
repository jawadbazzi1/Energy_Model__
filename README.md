Energy Load Model
============

In order to run main.py, run this command in the terminal and then run main.py while this command is running.

```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1
```

In order to check mlflow ui run the same command

```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1
```

In order to run the docker container that contains the flask app:

```
docker compose up
```

