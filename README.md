# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To excute the script
python < scriptname.py >

## Export your active environment to a new file:
```
conda env export > env.yml
```

## Setting up the environment:
Use the terminal or an Anaconda Prompt for the following steps:

 1) Create the environment from the environment.yml file:

    command: conda env create -f environment.yml
    The first line of the yml file sets the new environment's name.

 2) Creating an environment file manually.

  - Activate the new environment: conda activate myenv

  - Verify that the new environment was installed correctly:

    Command: conda env list
    or
    Command: conda info --envs.

## VS Code IDE Configuration
User setting files contain
 - "workbench.editor.enablePreview": false,
 - "python.formatting.provider": "black",
 - "editor.formatOnSave": true


## Commands Used to format the script
black nonstandardcode.py
isort nonstandardcode.py
flake8 nonstandardcode.py


## Implimentation of below concepts:
 - Divided the code into three different scripts.
   1) ingest_data.py
   2) train.py
   3) score.py

 - ``argparse`` module to accept user inputs.
 - ``logging`` module to write console and log files.
 - ``pickle`` module to dump and reuse the model objects.
 - ``unittest`` module to write test cases
 - ``sphinx`` module to write the documentation.
 - ``packaging`` - Created own package and installed it.


## MLFlow Server
 - To start the MLFlow server, go to project root path in terminal and run the command:
`mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 127.0.0.1 --port 5000`
 - After this, we can go to a browser and open the server at url `http://localhost:5000/ ` and track the
progress of runs under various experiments.