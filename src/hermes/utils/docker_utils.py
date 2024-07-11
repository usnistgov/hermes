"""Utilities for Docker, using python-on-whales."""

from python_on_whales import docker


def build_image(image_path: str, tag: str) -> None:
    """Builds a Docker image from the given path.

    Wrapper for docker.build

    Args:
        image_path: The path to the Dockerfile.
        tag: The tag to assign to the image.
    """
    docker.build(image_path, tags=tag)
