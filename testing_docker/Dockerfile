FROM ubuntu:22.04

RUN apt update && \
    apt install software-properties-common -y && \
    apt install curl -y && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip install grpcio grpcio-tools networkx