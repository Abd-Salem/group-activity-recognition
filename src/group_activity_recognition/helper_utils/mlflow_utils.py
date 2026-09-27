import mlflow


def start_run(experiment_name, run_name=None):
    mlflow.set_experiment(experiment_name=experiment_name)
    return mlflow.start_run(run_name=run_name)