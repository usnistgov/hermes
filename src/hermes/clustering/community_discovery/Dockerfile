FROM ubuntu:22.04

RUN apt update && \
    apt install software-properties-common -y && \
    apt install curl -y && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip install grpcio grpcio-tools networkx cdlib leidenalg
COPY . /opt/community_discovery
RUN ls /opt/community_discovery
RUN pip install /opt/community_discovery

ENTRYPOINT ["python3", "/opt/cd/protobuf/server.py"]
