# hermes



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