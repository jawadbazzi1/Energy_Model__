FROM python:3.10.5-buster

ARG MLFLOW_VERSION=1.19.0

WORKDIR /mlflow/
RUN pip install mlflow
EXPOSE 5000

ENV BACKEND_URI sqlite:////mlflow/mlflow.db
ENV ARTIFACT_ROOT /mlflow/artifacts

CMD mlflow server --backend-store-uri ${BACKEND_URI} --default-artifact-root ${ARTIFACT_ROOT} --host 127.0.0.1 --port 5000