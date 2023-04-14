# Driver script to import the installed packages and call the functions accordingly
import mlflow
import mlflow.sklearn

import ingest_data
import score
import train
from log_config import configure_logger

# Configuring logger
log_file = False
Console = True
Log_level = "DEBUG"
log = configure_logger(log_file, Console, Log_level)

DOWNLOAD_ROOT = (
    "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
)

if __name__ == "__main__":
    # MLFlow server details
    remote_server_uri = "http://127.0.0.1:5000"  # set to your server URI

    mlflow.set_tracking_uri(remote_server_uri)  # or set the MLFLOW_TRACKING_URI in the env

    exp_name = "House-price-prediction"
    mlflow.set_experiment(exp_name)

    # start MLFlow parent process
    with mlflow.start_run(
        run_name="PARENT_RUN-driver",
        tags={"version": "v0.4", "priority": "P1"},
        description="parent",
    ) as parent_run:

        log.info("MLFlow parent process started!")
        mlflow.log_param("parent-driver", "yes")

        # start one MLFlow child process for each script

        with mlflow.start_run(
            run_name="CHILD_RUN-ingest_data",
            description="First Child - Fetching all the data",
            nested=True,
        ) as child_run:

            mlflow.log_param("child-ingest_data", "yes")

            log.info("MLFlow child process for ingest_data started!")

            ingest_data.ingest_data_main()

        with mlflow.start_run(
            run_name="CHILD_RUN-train",
            description="Second Child - Preprocessong, EDA and model training",
            nested=True,
        ) as child_run2:

            mlflow.log_param("child-train", "yes")
            
            log.info("MLFlow child process for train started!")

            train.train_main()

        with mlflow.start_run(
            run_name="CHILD_RUN-score",
            description="Third Child - Checking scores of all the models ",
            nested=True,
        ) as child_run3:

            mlflow.log_param("child-score", "yes")
            
            log.info("MLFlow child process for score started!")
            
            score.score_main()
