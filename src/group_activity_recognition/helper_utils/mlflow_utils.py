import mlflow
import os
from group_activity_recognition.configs import get_config


def start_run(tracking_uri=None, experiment_name=None, run_name=None):

    config = get_config()
    TRACKING_URI = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", config.TRACKING_URI)
    mlflow.set_tracking_uri(TRACKING_URI)

    experiment_name = experiment_name or  config.EXP_NAME
    mlflow.set_experiment(experiment_name=experiment_name)

    return mlflow.start_run(run_name=run_name)