FROM python:3.12-slim
LABEL authors="lsanchez"

RUN apt-get update && apt-get install -y curl
RUN pip install mlflow psycopg2-binary boto3 cryptography pymysql

EXPOSE 5000