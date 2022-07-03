FROM python:3.10.5-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . /app

ENV BACKEND_URI sqlite:///mlflow.db
ENV ARTIFACT_ROOT /artifacts

CMD [ "python", "app.py"]

