<!-- <p align="center">
  <img src="[insert XXX IMAGE URL]" alt="Image Place" width="120">
</p> -->

<h1 align="center">
  Capstone Project<br>
  NYSX Stock Price Prediction: Docker and Experiment Tracking
  </h1>

<h4 align="center"> 
  <a href="/data/Readme.md">Data</a> •
  <a href="/experiments/Readme.md">Experiments Play Zone 🛝 </a> •
  <a href="/demo/Readme.md">Demo</a> •
  <a href="/production/Readme.md">Production 🏭 </a> •
  <a href="https://nysx-interactive-f35996.netlify.app/">Interactive Plots 🔍 📈 </a> • 
  <a href="/README.md">Main Page 🏠</a>
  </h4>
<br>




> [!IMPORTANT]  
> 
> Hello, 
>
> You are visiting the **Docker and MLflow Set Up** Zone. Here you will find detail instruction on how to set up Docker and MLflow for experiment tracking. If you have done it already, jump to [Demo](/demo/Readme.md) or [Production](/production/Readme.md)
> 
>

### Update Log: 
- 2025-11-30: Initial version


<!-- > [!WARNING]  
> 1. 
>  -->

---
## Content
- [1. Set up Docker]()

- [2. `docker-compose.yml`]()

- [3. Test MLflow]() 

- [4. Backup Docker and Artifact](#4-backup-docker-volume)



---
## 1. Set up Docker 

- **Docker Desktop** provides a straightforward GUI for user. For more information, you can go [here](https://docs.docker.com/desktop/). 
- Installation: Docker Desktop for [Mac](https://docs.docker.com/desktop/setup/install/mac-install/), [Windows](https://docs.docker.com/desktop/setup/install/windows-install/), and [Linux](https://docs.docker.com/desktop/setup/install/linux/)  
- For people prefer CLI (command line interface), you can install [Docker Engine](https://docs.docker.com/engine/install)  
- **Questions?**: Check out the [dockerdocs](https://docs.docker.com/)  

- ⚠️ By default, the Docker daemon in Linux runs with root privileges. This means that the Docker command, which interacts with the daemon, typically requires **sudo** to execute unless specific configurations are applied. See [here](#73-avoid-sudo-when-using-docker) for a solution. 

<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---
## 2. `docker-compose.yml`

The docker-compose.yml is recipe to tell docker how to create the **containers (metaphor: volume in disk)** for MLflow. The current setting contains the following containers:

- Postgres: Stores all the database data (MLflow experiment metadata) is stored here.
  - `volumes: - postgres_data:/var/lib/postgresql/data` --> a named Docker **volume**
  - Safe even if you stop or remove the container (docker compose down).
  - ⚠️ **Danger**: If you run `docker compose down -v`, this volume will be deleted, and all Postgres data will be lost.

- pgAdmin: Stores configuration inside the container itself.
  - No volume is defined for pgAdmin.
  - Not critical for your MLflow experiments; if lost, you can log in again.

- MinIO: Stores alll artifacts (MLflow models, files, logs) are stored here.
  - `volumes: - ./minio_data:/data` 
  - The ./minio_data folder on your host machine holds all MinIO data.
  - Safe even if you stop or remove the container.
  - ⚠️ **Danger**: If you delete the folder or use `docker compose down -v`, all artifacts will be lost.

- MLflow: Stores metadata in Postgres and artifacts in MinIO.
  - MLflow container itself doesn’t need to be committed unless you installed extra packages.
  - Shutting down is safe — data is in Postgres + MinIO volumes. 


<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---
## 3. Test MLflow

1. **`MLflow`** setup and quick test: Refer to UofT DSI [**production Repo**](https://github.com/UofT-DSI/production) and following the instruction from notebook [01_setup](https://github.com/UofT-DSI/production/blob/main/01_materials/labs/01_setup.ipynb) to test the MLflow in your docker. You can also test with the following code here


    ```bash
    # 1. clone the repo 
    cd path/to/clone/the/repo
    https://github.com/yikai82/ML3-Team-Project.git
    cd ML3-Team-Project$

    # 2. navigate to the cd/path/to//src/experiment tracking 
    cd src/experiment_tracking
    docker compose up -d  # linux user need to use sudo 
    # 3. if no error occurs, proceed to test the following two code
    python test_mlflow_artifact.py  
    python test_mlflow.py
    ```

2. If run successfully, you should see a terminal output as: `🧪 View experiment at: http://localhost:5001/#/experiments/#` 
   - `#` is the experiment number, the default is starting with `0`

3. To shut down docker, first **stop** then shut down

    ``` bash
    docker compose stop
    docker compose down  # this will perserve the status if you are going want resume experiment later
    docker compose down -v # "-v" = volumne; it is a nuclear option as it will shut down the containers AND **delete the attached named volumes**.
    ```   
    ⚠️ If you experience any issues, first to check the address and ports for the containers (Postgres, pgadmin, MinIO, and MLflow). See [here](#62-check-if-any-port-can-be-used-for-docker--mlflow) for additional information. If your issues are related to importing MLflow and suspect library conflicts, you might need to [reinstall MLflow](#63-re-installation-of-mlflow-copy-from-slack-message-from-dmytro). 

<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---
## 4 Backup Docker Volume and Artifacts: 

### 4.1 Backup Docker Volume

```bash
### Stop containers
docker compose stop

### Backup Postgres
# stop postgres if not done already
docker compose stop postgres

# create a local backup 
docker run --rm \  
  # `docker run` is a basic command to start a new container from an image. In this case, the image is alpine, which is a very small Linux distribution — lightweight and perfect for running temporary commands.
  # -- rm This tells Docker: “remove the container automatically when it exits.” Normally, when a container stops, it still exists in Docker’s container list (docker ps -a) and takes up space. This avoids leaving a stopped container behind.

-v experiment_tracking_postgres_data:/data \       
  # This mounts your Docker named volume experiment_tracking_postgres_data into the container at path /data.
  # Inside the container, /data now contains all your Postgres files.

-v $(pwd):/backup \                                  
  # Mounts the current host directory ($(pwd) = present work#ing directory) into the container at /backup.
  # This is where the backup .tar.gz file will be saved on your host.

alpine \  # The container image to use. Lightweight Linux with just enough tools to run
tar czf /backup/postgres_data_backup.tar.gz -C /data . #  compresses all data into a single .tar.gz file
  # tar is a Linux command to compress files into an archive.
  # c: create a new archive
  # z:  compress using gzip
  # f: /backup/postgres_data_backup.tar.gz --> write output to /backup/postgres_data_backup.tar.gz
  # -C /data → change directory to /data inside container before archiving
```

<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---
### 4.1 Backup MinIO artifacts
```bash
docker compose stop minio # run if not sopt already

tar czf minio_data_backup.tar.gz -C ./minio_data . # compresses all files in minio_data into a backup
```

<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---
## Reference
1. [Docker Docs: Manual](https://docs.docker.com/desktop/)
2. [MLflow: Getting Started](https://mlflow.org/docs/latest/ml/getting-started/)
3. [GitHub: UofT-DSI/Production, from Digital Science Institute at University of Toronto](https://github.com/UofT-DSI/production)