FROM tensorflow/tensorflow:latest-gpu-py3-jupyter
MAINTAINER Euisoon Ahn <euisoon.ahn@gmail.com>

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get -y install python3.6
RUN ln -s --force $(which python3.6) /usr/local/bin/python
RUN ln -s --force $(which python3.6) /usr/local/bin/python3

ENV LANG C.UTF-8
ENV LANGUAGE C.UTF-8
ENV LC_ALL C.UTF-8

COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

VOLUME ["/data", "/fp_img", "/notebooks"]
