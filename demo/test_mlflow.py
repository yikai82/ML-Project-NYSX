## For Demo Only, not logging artifact
## Current issue: hangging when running: mlflow.sklearn.log_model(lr, artifact_path, signature=signature)

import time

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow; print("MLflow version:", mlflow.__version__)
import mlflow.sklearn
from mlflow.models import infer_signature

from utils.logger import get_logger

_logs = get_logger(__name__)

mlflow.set_tracking_uri("http://localhost:5002")

if __name__ == "__main__":
    _logs.info('Starting ML flow test.')

    mlflow.set_experiment("mlflow_demo")
    with mlflow.start_run():
        try:
            X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
            y = np.array([0, 0, 1, 1, 1, 0])
            _logs.info(f'Created X: {X} and y: {y}')

            lr = LogisticRegression()
            lr.fit(X, y)

            try: 
                _logs.info(f'Fitted model: {lr}')
                score = lr.score(X, y)
                
                _logs.info(f"Score: {score}")
                mlflow.log_metric("score", score)
                predictions = lr.predict(X)
                
                _logs.info(f'Predictions: {predictions}')
                signature = infer_signature(X, predictions)

                # time.sleep(10)  # wait 5 seconds for MinIO/bucket to be ready
                # _logs.info(f'wait for 5 seconds')

                # # Use a local artifact path:
                # artifact_path = "./tmp_model"  # local folder for model artifacts
                # _logs.info(f"Logging model to local path {artifact_path} instead of S3")
                # mlflow.sklearn.log_model(lr, artifact_path, signature=signature)
                
                _logs.info(f"Model saved in run {mlflow.active_run().info.run_uuid}")
            
            except Exception as e:
                print("Failed to log model:", e)

        except Exception as e:
            _logs.error(f"Error during ML flow test: {e}")
