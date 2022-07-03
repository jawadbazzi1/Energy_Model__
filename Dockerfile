FROM python:3.10.5-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . /app

CMD [ "flask", "run", "--host=0.0.0.0"]

