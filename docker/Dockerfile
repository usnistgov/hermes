FROM ubuntu:focal

ENV EXEC_DIR="/opt/executables"
ENV DATA_DIR="/data"

RUN mkdir ${EXEC_DIR} && \
    mkdir ${DATA_DIR} && \
    mkdir /hermes_src

RUN apt update && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install python3 python3-dev build-essential python-is-python3 curl -y && \
    apt autoremove -y && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    # rm -rf /var/lib/apt/lists/*

    # RUN ln -sf /usr/bin/python3.9 /usr/bin/python3

    WORKDIR ${EXEC_DIR}


COPY . /hermes_src

RUN pip install /hermes_src[all] --no-cache-dir && \
    rm -rf /hermes_src