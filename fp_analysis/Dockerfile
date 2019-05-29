FROM tensorflow/tensorflow:latest-gpu-py3-jupyter
MAINTAINER Euisoon Ahn <euisoon.ahn@gmail.com>

RUN apt-get update
# RUN apt-get install -y

ENV LANG C.UTF-8
ENV LANGUAGE C.UTF-8
ENV LC_ALL C.UTF-8

COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

VOLUME ["/data", "/fp_img", "/notebooks"]
