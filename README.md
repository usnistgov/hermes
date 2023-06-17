# hermes

## Installation

We use `poetry` as our main package and dependency manager. We recommend using `poetry` to
install `hermes`. To do so, clone this repo, navigate to the root directory and run `poetry install`.
For instructions on how to install `poetry` see [here](https://python-poetry.org/docs/#installation)

Alternatively, you can run `pip install .` inside the root directory to install `hermes`. *If your machine is
macOS and ARM64 (M1, M2), this is the recommended method.*

### Installing without cloning
To install `hermes` without cloning this repository, run the following command:
``` bash
$ pip install git+ssh://git@github.com/cvelezrmc/hermes.git@scratchcv
```
or to run without SSH
``` bash
$ pip install git+https://<my_token>@github.com/cvelezrmc/hermes.git@scratchcv
```
where <my_token> is your personal access [GitHub Token](https://docs.github.com/en/enterprise-server@3.4/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token).

## goals

- Consistent active learning and modeling interface aimed at enabling nonstandard analysis and acquisition policy? But with batteries included for standard BayesOpt or whatever
- data acquisition and wrangling with no-work FAIR backend integration
- Possibly actual ML models and bag of materials/physics tricks lives in separate module?

## Base level tasks

- Instrument Communication:
-       Basic functions for importing data from instruments and setting them up for use in modeling
-       Instrument specific functions for reading data in, sending commands and the like.

- Intrinsic Data Analysis:
-       Analysis of the intrinsic properties of the data
-       Examples include: data pre-processing, domain-specific data manipulation,
-       clustering, dimesionallity reduction, distance measures.
-       All inputs are treated as features

- Relational Data Analysis:
-       Analysis of how the inputs are related to observations of the outputs.
-       Examples include: Regression, classification, physical models.

- Persistant Storage: 
-       Basic functions for data storage and database design/use.