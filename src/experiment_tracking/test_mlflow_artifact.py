import mlflow
import os

mlflow.set_tracking_uri("http://localhost:5001")  # internal Docker hostname + port

with mlflow.start_run():
    mlflow.log_text("test", "test.txt")
    print("Logged a test artifact successfully!")
