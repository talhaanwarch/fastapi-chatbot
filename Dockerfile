FROM python:3.10

WORKDIR /code

COPY . /code

RUN pip install  --upgrade -r /code/requirements.txt 
#--no-cache-dir
WORKDIR /code/app

