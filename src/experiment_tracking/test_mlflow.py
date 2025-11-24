import time

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from utils.logger import get_logger

_logs = get_logger(__name__)

mlflow.set_tracking_uri("http://localhost:5001")

if __name__ == "__main__":
    _logs.info('Starting ML flow test.')

### add check if experiment_name exists 

    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri("http://localhost:5001")
    client = MlflowClient()

    experiment_name = "mlflow_test_experiment2"

    # Search for the experiment
    experiments = client.search_experiments()
    exp_ids = [exp.experiment_id for exp in experiments if exp.name == experiment_name]

    if exp_ids:
        exp = [exp for exp in experiments if exp.name == experiment_name][0]
        if exp.lifecycle_stage == "deleted":
            client.restore_experiment(exp.experiment_id)
            _logs.info(f"Restored deleted experiment '{experiment_name}'.")
    else:
        # experiment does not exist; mlflow will create it when set_experiment is called
        _logs.info(f"Experiment '{experiment_name}' does not exist. It will be created.")

    mlflow.set_experiment(experiment_name)

    # mlflow.set_experiment("mlflow_test_experiment")
    with mlflow.start_run():
        try:
            X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
            y = np.array([0, 0, 1, 1, 1, 0])
            _logs.info(f'Created X: {X} and y: {y}')

            lr = LogisticRegression()
            lr.fit(X, y)
            
            _logs.info(f'Fitted model: {lr}')
            score = lr.score(X, y)
            
            _logs.info(f"Score: {score}")
            mlflow.log_metric("score", score)
            predictions = lr.predict(X)
            
            _logs.info(f'Predictions: {predictions}')
            signature = infer_signature(X, predictions)

            # time.sleep(10)  # wait 5 seconds for MinIO/bucket to be ready
            # _logs.info(f'wait for 5 seconds')

            _logs.info('Logging model.')
            mlflow.sklearn.log_model(lr, "model", signature=signature)
            
            _logs.info(f"Model saved in run {mlflow.active_run().info.run_uuid}")
        
        except Exception as e:
            _logs.error(f"Error during ML flow test: {e}")