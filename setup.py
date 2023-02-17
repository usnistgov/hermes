"""hermes setup.py."""

from setuptools import find_packages, setup

package_data = ["hermes/*"]

setup(
    name="hermes",
    version="0.0.1",
    author="Austin McDannald, Brian DeCost, Zachary Trautt, Aaron Gilad Kusne, Camilo Velez",
    author_email="austin.mcdannald@nist.gov",
    description="Hermes",
    packages=find_packages(),
    package_data={"hermes": package_data},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
