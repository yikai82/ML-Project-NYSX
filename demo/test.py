# test
import mlflow; print("MLflow version:", mlflow.__version__)

mlflow.set_tracking_uri("http://localhost:5002")

with mlflow.start_run():
    mlflow.log_param("demo_param", 123)
    mlflow.log_metric("demo_metric", 0.99)
