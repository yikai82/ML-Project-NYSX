## For Demo Only 

# docker/minio/create-bucket.sh
#!/bin/sh
# Configure MinIO Client


# until (/usr/bin/mc config host add minio http://minio_demo:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}); do
until mc alias set minioserver http://minio:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}; do

  echo "Waiting for MinIO to be ready..."
  sleep 1
done

# Set alias again just to be sure
mc alias set minioserver http://minio:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}

# Create the MLFlow bucket (idempotent)
mc mb --ignore-existing minioserver/mlflow_demo ## <- default mlfow 
echo "Bucket mlflow_demo is ready."