FROM tensorflow/tensorflow:1.15.2-gpu-jupyter
MAINTAINER Euisoon Ahn <euisoon.ahn@gmail.com>

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get -y install python3.6
RUN ln -s --force $(which python3.6) /usr/local/bin/python
RUN ln -s --force $(which python3.6) /usr/local/bin/python3

ENV LANG C.UTF-8
ENV LANGUAGE C.UTF-8
ENV LC_ALL C.UTF-8

RUN mkdir -p /usr/share/fonts/opentype/noto/
RUN curl https://raw.githubusercontent.com/googlefonts/noto-cjk/master/NotoSansKR-Regular.otf > /usr/share/fonts/opentype/noto/NotoSansKR-Regular.otf
RUN curl https://raw.githubusercontent.com/googlefonts/noto-cjk/master/NotoSansKR-Bold.otf > /usr/share/fonts/opentype/noto/NotoSansKR-Bold.otf

COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

RUN jupyter nbextension install https://github.com/drillan/jupyter-black/archive/master.zip
RUN jupyter nbextension enable jupyter-black-master/jupyter-black

VOLUME ["/data", "/fp_img", "/notebooks"]
