# Housing value prediction

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - Prepare and cleand the data. Check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To excute the script
```
python < scriptname.py >
```

## Export your active environment to a new file:
```
conda env export > env.yml
```

## Setting up the environment:
Use the terminal or an Anaconda Prompt for the following steps:

 1) Create the environment from the environment.yml file:

    command: ```conda env create -f env.yml```

    The first line of the yml file sets the new environment's name.

 2) Creating an environment file manually.

  - Activate the new environment: conda activate myenv

  - Verify that the new environment was installed correctly:

    Command: ```conda env list```
    or
    Command: ```conda info --envs```

 3) Override the environment file manually.
    ```conda env export > env.yml```

## VS Code IDE Configuration
User setting files contain
 - "workbench.editor.enablePreview": false,
 - "python.formatting.provider": "black",
 - "editor.formatOnSave": true


## Commands Used to format the script
```
black nonstandardcode.py
```
```
isort nonstandardcode.py
```
```
flake8 nonstandardcode.py
```


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
```mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 127.0.0.1 --port 5000```
 - After this, we can go to a browser and open the server at url ```http://localhost:5000/``` and track the
progress of runs under various experiments.


## Install package .whl file
```
pip install pack_install-1.0-py3-none-any.whl
```

## Docker
Use Below steps to create a docker image and run it:

- Create a ``requirements.txt`` file to install all of your enviorment packages to docker
```
pip list --format=freeze > requirements.txt
```

- Install Docker to your system

- Create a ``Dockerfile`` according to your project requirements.

- Run the command to build the docker image
```
docker build -t mlflow-container-housing_price_prediction .
```

- Run the command to run the docker image in your system
```
docker run -p 5000:5000 mlflow-container-housing_price_prediction
```

### Docker Hub
- Create a docker hub id and run the below commands to push the dockers image to dockerhub reposetory.
1) To connect the dockerhub reposetory to your local system
```
docker login
```

2) To crate a tag for dockerhub repo with the version
```
docker tag mlflow-container-housing_price_prediction:latest protiger/mlflow-container-housing_price_prediction
```

3) To push the docker image to the docker hub repo.
```
 docker push protiger/mlflow-container-housing_price_prediction
```

4) To pull the image from dockehub repo to anyone's local system
```
docker pull protiger/mlflow-container-housing_price_prediction:latest
```

5) To run the downloaded image form docker hub
```
docker run -p 5000:5000 protiger/mlflow-container-housing_price_prediction:latest
```

6) To launch the mlflow from docker image
```
http://127.0.0.1:5000/
```

### Docker Volumes
Docker Volumes spaces we can only access with docker tools. We cannot access files in our local system until we have certain permissions.

Docker volums also we cannot create on our dezired location, it will only be created where docker is configured in our system. If we want to create volumes at a specific location then we need to change the configurations which we did while installing docker.

- Run the command to Create the docker volume name ```output``` in your system
```
docker volume create output
```

- Run the command to see the status of docker volume
```
docker volume inspect output
```

- Run the command to build the new docker image
```
docker build -t mlflow-container-housing_price_prediction_with_volumes .
```

- Run the command to mount docker image to volume and run the docker containor
This Command only run the docker image not mlflow server
```
docker run -d --name with_vol --mount source=output,destination=/TCE_2023/data/output mlflow-container-housing_price_prediction_with_volumes
```
This Command will run the docker image and mlflow server both
```
docker run -p 5000:5000 -d --name with_vol_and_mlflow --mount source=output,destination=/TCE_2023/data/output mlflow-container-housing_price_prediction_with_volumes
```
- Run these command to inspect the volume mounts
```
docker inspect with_vol | grep Mounts -A 10
```


- To launch the mlflow from docker image
```
http://127.0.0.1:5000/
```

### Docker Mounts
We use docker mounts to mount the docker image with local system like volumes but the difference is we can specify the location of our local system and also we don't need any docker app or enviorment to access the files we can access them by ourselves.

- Run the command to build the new docker image
```
docker build -t mlflow-container-housing_price_prediction_with_local_mounts .
```

- Run the command to mount docker image to volume and run the docker containor
This Command only run the docker image not mlflow server
```
docker run -v /mnt/c/Users/varsha.rajawat/Desktop/TCE_2023:/app mlflow-container-housing_price_prediction_with_local_mounts
```
This Command will run the docker image and mlflow server both
```
docker run -p 5000:5000 --name with_LocalMount_and_mlflow -v /mnt/c/Users/varsha.rajawat/Desktop/TCE_2023:/app mlflow-container-housing_price_prediction_with_local_mounts
```

- To launch the mlflow from docker image
```
http://127.0.0.1:5000/
```
