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


#Setting up the environment:
Use the terminal or an Anaconda Prompt for the following steps:

 1) Create the environment from the environment.yml file:

  - command: conda env create -f environment.yml
  - The first line of the yml file sets the new environment's name. 
   
 2) Creating an environment file manually.

  - Activate the new environment: conda activate myenv

  - Verify that the new environment was installed correctly:

  - Command: conda env list
    or
  - Command: conda info --envs.




