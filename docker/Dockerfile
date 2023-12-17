FROM ubuntu:22.04

ARG DATASETS_PATH
ARG TXT_PATH

WORKDIR /home/

ENV TZ=America/Sao_Paulo
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ >/etc/timezone

RUN apt-get -y update && \
    	apt-get -y upgrade && \
	apt-get -y install git vim wget curl python3 build-essential unzip

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 100

RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py
RUN pip install tqdm numpy matplotlib scikit-learn pyYAML easydict
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

RUN git clone https://github.com/LucasFernando-aes/UnkE-OVANet.git && cd UnkE-OVANet

COPY $DATASETS_PATH ./UnkE-OVANet/data/
COPY $TXT_PATH ./UnkE-OVANet/txt/