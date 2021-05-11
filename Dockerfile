# FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 
FROM ubuntu:18.04

RUN apt-get update

RUN apt-get install -y git \
    software-properties-common \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install python3.6 -y && \
    apt install python3-distutils -y && \
    apt install python3.6-dev -y && \
    apt install build-essential -y && \
    apt-get install python3-pip -y && \
    apt update && apt install -y libsm6 libxext6 && \
    apt-get install -y libxrender-dev && \ 
    apt install libgl1-mesa-glx -y

RUN apt install pkg-config
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt-get update
RUN apt-get install -y libopencv-dev 

# make darknet
# make PCN
ADD libs libs

RUN cd libs/yolo_darknet && \
	sed -i "s/OPENCV=0/OPENCV=1/" Makefile &&\
	# sed -i "s/GPU=0/GPU=1/" Makefile &&\
	# sed -i "s/CUDNN=0/CUDNN=1/" Makefile &&\
	# sed -i "s/CUDNN_HALF=0/CUDNN_HALF=1/" Makefile &&\
    sed -i "s/AVX=0/AVX=1/" Makefile &&\
    sed -i "s/OPENMP=0/OPENMP=1/" Makefile &&\
	sed -i "s/LIBSO=0/LIBSO=1/" Makefile &&\
	make &&\
    cd ../..

# install lib
ADD requirements.txt .
RUN python3 -m pip install -U pip &&\
    # fix bug can not install skbuild
    python3 -m pip install -U setuptools &&\
    pip3 install -r requirements.txt 

WORKDIR /FPT_hackathon
COPY . .

# CMD ["python3", "vehicles_detection_api.py"]