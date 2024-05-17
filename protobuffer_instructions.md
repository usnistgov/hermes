# Protobuffer files

### File Structure

The definition of objects and services is inside a `*.proto` file. Use `/src/hermes/clustering/rbpots.proto` as an example.

From this file, [two python scripts are generated](#file-generation). These scripts will live in two places (identical copies):
* In the same directory as the hermes python script that is using them (for example, in `src/hermes/clustering`)
* In the `protobuf` directory that will be accessed by the "server" running inside Docker (for example, in `src/hermes/clustering/community_discovery/protobuf`)

In hermes, import names of `*_pb2.py` and `*_pb2_grpc.py` need to be updated in each case for each of the scripts that will "make calls" to the docker containers.

In Docker, each `host.py` script inside `protobuf` needs to be updated with the correct import names.

## File Generation
In `src/hermes/clustering` there is a file called `generate_protobuf.py`. Copy this file to each location where you will need to generate the python scripts from a `.proto` file. This file must be in the same directory as `.proto` and the docker directory (for example, `community_discovery`) must also be in the same directory as `generate_protobuf.py`.

To use this script:
1. Create a new Python Virtual Environment
2. Install [grpcio-tools, typer]*
3. Run `python generate_protobuf.py -n "<name of .proto>" -d "<name of docker directory>

For example: `python generate_protobuf.py -n "rbpots.proto" -d "community_discovery"`

\* This must be in a fresh virtual environment since `grpcio-tools` depends on a `protobuffer` version that is incompatible with the version required by `tensorflow`. `typer` is just the python package that allows you to create simple scripts that can use command line arguments.

The script will:
1. Generate `_pb2.py` and `_pb2_grpc.py` from `.proto`
2. Copy these files to the docker directory specified (using ` -d`) inside `protobuf` directory
3. Log everything to `generate_protobuf.log`

# Docker

### Dockerfile
Use `src/hermes/clustering/community_discovery/Dockerfile` as reference.

Modify directory names (`*/commmunity_discovery`) as needed.

I am installing a lot of things here
```
RUN apt update && \
    apt install software-properties-common -y && \
    apt install curl -y && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/*
```
It could be that not all of these things are necessary and choosing more carefully what to install could decrease the installation time and size of containers.

### Building Containers

Python-on-whales takes care of building containers, running containers (first time after building), and starting containers (after running for first time).