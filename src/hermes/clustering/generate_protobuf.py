"""Script to generate protobuffer files."""

# pylint: disable=W1203
import logging
import shutil
import subprocess
from pathlib import Path

import typer

app = typer.Typer(help="Generate protobuffer files.")
logger = logging.getLogger("generate_protobuf")
fhandler = logging.FileHandler("generate_protobuf.log")
fformat = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
)
fhandler.setFormatter(fformat)
fhandler.setLevel("INFO")
logger.addHandler(fhandler)
logger.setLevel("INFO")


@app.command()
def main(
    name: str = typer.Option(
        None,
        "--name",
        "-n",
        help="Name of .proto file.",
    ),
    docker_dir: str = typer.Option(
        None,
        "--docker-dir",
        "-d",
        help="Directory to copy generated files to.",
    ),
) -> None:
    """Generate protobuffer files."""
    proto_command = (
        f"python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. {name}"
    )
    # Generate protobuffer files
    try:
        subprocess.run(
            proto_command.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error generating protobuffer files: {e.output}")
        raise e
    logger.info(f"Generated protobuffer files for {name}.")

    # Copy generated files to appropriate directory for docker
    # this will create a directory called "protobuf"
    # in the specified directory

    Path(__file__).with_name(docker_dir).joinpath("protobuf").mkdir(
        parents=True, exist_ok=True
    )
    name = name.split(".")[0]
    shutil.copyfile(
        Path(__file__).with_name(name + "_pb2.py"),
        f"{docker_dir}/protobuf/{name}_pb2.py",
    )
    logger.info(f"Copied {name}_pb2.py to {docker_dir}/protobuf.")
    shutil.copyfile(
        Path(__file__).with_name(name + "_pb2_grpc.py"),
        f"{docker_dir}/protobuf/{name}_pb2_grpc.py",
    )
    logger.info(f"Copied {name}_pb2_grpc.py to {docker_dir}/protobuf.")


if __name__ == "__main__":
    app()
# "python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. graph.proto"
