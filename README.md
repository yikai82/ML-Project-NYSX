<!-- insert image  -->
<!-- <p align="center">
  <img src="" alt="Image Plance" width="180">
</p> -->

<p>
<h1 align="center">
  Capstone Project: <br>
  The Next Day Stock Price Prediction 
  </h1>

> [!IMPORTANT]  
> <i><h4 align="center">Most of the business problems are not ML problems, and most of the ML problems are not business problems. Optimizing an ML model is not the same thing as optimizing a solution for a business problem </h4><p>  
> <h5 align="right"> - from Production Lecture Day 1 : 36:58</h5><p>
>
> <h4 align="center">A Machine Learning System is a system that can learn automatically to improve its performance</h4></i>  
> <br>  
>
> This capstone project showcases what I learned during a 16-week intensive AI/ML course offered by the University of Toronto’s Data Science Institute. I am not a financial professional, but I do invest in the market as a side pursuit, chasing the occasional moonshot 🌛 🏹.  

> [!WARNING]  
> 1. Your environment setup is as much as important as your data — you should always back up your environment if you don't know what are you doing and just blindly copy-pasting solution from internet as everyone's environment setup can be quite different from each other --> click [here](#71-backup-conda-environment) to back up your conda environment.
>  
> 2. MLflow keeps experiment metadata in the tracking server, such as SQLite, MySQL, PostgreSQL (here), etc. The default host UI is http://localhost:5001/#/experiments/0 (or localhost:5002 if run with demo). It is recommended not to delete any experiment you have been using for testing, as this will cause an error stating `experiment lifecycle_stage is "deleted"` or `Cannot set a deleted experiment(...)`, and you will not be able to start a test run.   
> **To fix**: create a new experiment with a different name. You can delete a run in an experiment
>
>
>
<br>

![DEMO-BLUE](https://img.shields.io/badge/DEMO-496C9C)  
👉 [Interactive actual vs predict for AAPL stock price with test data set.](https://nysx-lstm-aapl.netlify.app/) It’s best viewed on a desktop browser.

---

<p><h4 align="center">
Already got your docker and mlflow ready 💪💪 <br>
and just want to skip and ready to play 🕹️: 👉 Check out the <a href="/demo/Readme.md">Demo</a> and <a href="/production/Readme.md">Production</a><br>    
  
  
<p><h4 align="center">
** Continue Reading 👇**<br>
</p>

---
<p align="center">
   🧭 🗺️ Navigate to other place:
  <a href="/data/Readme.md">Data</a> •
  <a href="/demo/Readme.md">Demo</a> •
  <a href="/experiments/Readme.md">Experiment</a> •
  <a href="/production/Readme.md">Production</a><br></b>
</p>

## Content

* [1. Business Problem](#1-business-problem)    
* [2. Repo Layout](#2-repo-layout)
* [3. Environment Setup ](#3-environment-setup)
* [4. Setup MLflow with python Script for Experiment Tracking](#4-set-up-docker--mlflow-with-python-script-for-experiment-tracking)  
  - [Docker Setup](#41-set-up-docker) 
  - [Test MLflow](#42-test-mlflow)
* [5. Demo](#5-demo)
* [6. Results and Finding](#6-results-and-findings)
* [7. Troubleshooting](#7-️-troubleshooting)


---
## System

<div align="left">
  <div style="margin: 2px 0;">
    <img src="images/Linux2.svg" alt="Linux" width="50" style="vertical-align: middle; margin-right: 6px;">
    <span style="vertical-align: middle;"><b>Kubuntu-T2 24.04.2 LTS</b></span>
  </div>
  <div style="margin: 2px 0;">
    <img src="images/Noble.svg" alt="Noble" width="50" style="vertical-align: middle; margin-right: 6px;">
    <span style="vertical-align: middle;">Codename: Noble</span>
  </div>
</div>  

Release: 24.04  
Kernel Version: Linux 6.14.0-1-t2-noble  
Hardware: Intel® Core™ i9-9880H CPU @ 2.30GHz, 16 GM RAM 

---
**LSTM** = Long Short-Term Memory | **ML** = Machine Learning 
<br>


## 1. Business Problem: 

<img src="images/trump_tweet.jpg" width="350" align="right" style="margin-left: 40px; margin-bottom: 40px;">
Our client recently experienced market losses triggered by a high-profile tweet. 
To improve their ability to respond to sudden shifts, the client wants to explore 
LSTM-based stock prediction models. Instead of maintaining one model per stock, 
they’re interested in whether a sector-level model could learn shared patterns 
and then be applied to individual tickers. <p> 


**Constraint**:  
Maximum training epochs = 20 to enable fast iteration and leave room for future feature additions.

**Project goal**:  
Build simple LSTM models to forecast next-day stock prices, compare their performance, and evaluate whether a single sector-level model can match the performance of stock-specific models.


<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---
## 2. Repo Layout

```text
.
├── data
├── demo
├── experiments
├── html
├── images
├── LICENSE
├── ml_env_complete.txt
├── ml_env_complete.yml
├── models
├── production
├── README.md
├── requirements.txt
├── requirements.yml
├── results
└── src
```

- **data**: Contains both raw data and processed data. Click [here](/data/Readme.md) for more details.  

- **demo**: Contains instructions and files to perform a quick demo with Docker + MLflow 

- **experiments**: Contains all the notebooks for experiments performed here. Click [here](/experiments/Readme.md) for more details.

- **html**: Default path for interactive plots generated from notebooks:  
    `fig.write_html(f"../html/{Tick}_actual_vs_predicted.html)`

- **models**: Default path for saving models during production runs with MLflow:
    ` model_lstm.save(f"../../models/{Tick}_lstm_model.keras")`  
    - This default path ensures that models are first stored in a central location. After the production run completes, the models are moved to the specific run folder (e.g., `Run_3`) to avoid overwriting models from previous runs. Currently, Run_3 contains 56 models from the latest production run.

- **productions**: Default path for saving output *.csv and plots (*.pmg) during production runs with MLflow:


``` bash
# for example 
output_PATH = f"../../production/Run_X"
os.makedirs(f"{output_PATH}", exist_ok=True)
# some code here...

FILE_NAME = f"{Tick}_training_loss"
plt.savefig(f"{output_PATH}/{FILE_NAME}.png", dpi=300, bbox_inches='tight')

```

- **results**: Contains a high level summary of the finding from the work 

- **src**: Contains source code for experiment trackering (using MLflow), logs, and utilities (utils).

  
**src**: source code | **utils:** utilities 


<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---
## 3. Environment Setup 

⚠️ My system is Linux (Kubuntu24.04.02 LTS) with a MBP2019 [here](#system). For NVIDIA GPU users: Install CUDA toolkit and drivers appropriate for your GPU before creating the Conda environment.


### 3.1 Install `dsi_participant` environment with Miniconda

- Visit the [Miniconda nstallation Page](https://www.anaconda.com/docs/getting-started/miniconda/main) 

- Choose and download the appropriate installer based on your system architecture:

- Open a terminal in the folder where the installer was downloaded.

- Run the installer:

```bash 
bash Miniconda3-latest-Linux-<your_architecture>.sh
```

- Follow the on-screen instructions. When asked, say “yes” to initializing Conda.
- Confirm the installation:

```bash
conda --version
```

- You should see something like conda 23.x.x.

- Now following the following command one-by-one in terminal to install the required package.

```bash
conda create --name dsi_participant python=3.9 # create a new conda environment 

conda activate dsi_participant # activate the new conda enviroment 

conda install -c conda-forge numpy requests ipykernel pandas seaborn scikit-learn python-dotenv dask "pyarrow>=11.0.0" sacred sqlalchemy psycopg2 shap fancyimpute missingno tensorflow matplotlib plotly nbformat scikit-image opencv transformers yfinance pygam pybind11

conda list # this should return a list where you can save as txt for future reference. 
``` 

<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---
### 3.2 Set up environment with a clean envirment.yml or environment.txt

- I also exported my full Conda environment (`ml_env_complete.yml`) and included it in the repo as a fallback option in case the above setup method fails, you can also try [3.3](#33-set-up-minimum-environment-with-a-minimumyml) below.


```bash
## Option 1: For conda user
cd path/to/the/folder 
conda env create -f environment.yml # This will create a conda enviroment with the same name defined in the environmet.yml. 
                                    # In this case, the name will be "ml_env_complete" 

## Option 2: For python venv user  
python -m venv myenv # replace [myenv] to whatever name you like

# for Linux / Mac
source myenv/bin/activate

# for Windows
myenv\Scripts\activate

# install virtual environment 
pip install -r ml_env_complete.txt

```

<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---
### 3.3 Set up minimum environment with a requirement.yml or requirement.txt 

Same step as below just replace the `ml_env_complete` to `requirement` for either yml file or text file based on your setting. 

```bash
## Option 1: For conda user
cd path/to/the/folder 
conda env create -f requirements.yml # create a conda enviroment with the same name defined in the environmet.yml. In this case, the name will be "ml_env_complete" 


## Option 2: For python venv user  
python -m venv myenv # replace [myenv] to whatever name you like

# for Linux / Mac
source myenv/bin/activate

# for Windows
myenv\Scripts\activate

# install virtual environment 
pip install -r requirements.txt
```


<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---
## 4. Set up Docker + MLflow with python Script for Experiment Tracking

### 4.1 Set up Docker 

- **Docker Desktop** provides a straightforward GUI for user. For more information, you can go [here](https://docs.docker.com/desktop/). 
- Installation: Docker Desktop for [Mac](https://docs.docker.com/desktop/setup/install/mac-install/), [Windows](https://docs.docker.com/desktop/setup/install/windows-install/), and [Linux](https://docs.docker.com/desktop/setup/install/linux/)  
- For people prefer CLI (command line interface), you can install [Docker Engine](https://docs.docker.com/engine/install)  
- **Questions?**: Check out the [dockerdocs](https://docs.docker.com/)  

- ⚠️ By default, the Docker daemon in Linux runs with root privileges. This means that the docker command, which interacts with the daemon, typically requires **sudo** to execute unless specific configurations are applied. See [here](#73-avoid-sudo-when-using-docker) for solution. 

 ### 4.2 Test MLflow

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

 
- ⚠️ If you experience any issues, first to check the address and ports for the containers (Postgres, pgadmin, MinIO, and MLflow). See [here](#62-check-if-any-port-can-be-used-for-docker--mlflow) for additional information. If your issues are related to importing MLflow and suspect library conflicts, you might need to [reinstall MLflow](#63-re-installation-of-mlflow-copy-from-slack-message-from-dmytro). 


<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---

## 5 Demo 

```text
../demo/
.
├── images
├── logs
├── minio
├── minio_data_demo
├── mlflow
├── postgres
├── postgres_data
├── __pycache__
├── utils
├── demo_LSTM_v04.3.py
├── docker-compose-demo.yml
├── test_mlflow.py
└── test.py
```

- A simple setup to run a single stock price prediction (default: AAPL) using the NYSX dataset from Kaggle. For more information, check [here](/demo/Readme.md)

- ⚠️ Known issues: The current demo script cannot log artifacts, as doing so will cause MLflow to hang. Logging parameters or metrics works fine. There is no issue in production, since the demo and production Docker setups are  independently.  



<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---
## 6. Results and Findings 

### 6.1 



Coming soon


### 6.2



Coming soon...


## 7. ⚒️ Troubleshooting 

### 7.1 Backup Conda Environment 
1. Before doing more troubleshooting on your environment, make sure you create a backup so you can restore it if the solution does not work or makes it worse. If nothing works, sometimes it might be easier just press that reset bottom and start from the [scratch](#3-environment-setup)

```bash
## backup a conda environment
conda activate your_env_name
conda env export > bkup_your_env_name.yml  # change the name "bkup_your_env_name.yml" if you like

## restore a conda environment 
conda env create -f bkup_your_env_name.yml # this will create an envirment with the name set in the yml file. 
```

<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---

### 7.2 Check if any port can be used for Docker + MLflow 

1. The original [docker-compose_ver00.yml](/src/experiment_tracking/backup/docker-compose_ver00.yml) can be accessed in the src/experiment_tracking/backup/. The current working version can be access [here](/src/experiment_tracking/docker-compose.yml)

2. Below is the comparison the between ver00 and teh current version, which should help you to troubleshoot your issue.

| Component          | `<ver00>`                    | `<new>`                      | Effect                                  |
| ------------------ | ---------------------------- | ---------------------------- | --------------------------------------- |
| Postgres volume    | `./postgres_data` bind mount | `postgres_data` named volume | More portable, avoids permission issues |
| MinIO port 1       | `9000:9000`                  | `9100:9000`                  | Avoids port conflict                    |
| MinIO port 2       | `9001:9001`                  | `9101:9001`                  | Avoids port conflict                    |
| Named volume block | ❌ Not present                | ✅ Present                  | Required for named volume               |
| Other services     | Same                         | Same                         | No change                               |

3. To check whether a port is already in use on Linux — for example before starting MLflow so it doesn’t fight over the same port, there are a few options:

- Quick Check
```bash
sudo lsof -i :5000  # test port 5000, replace it to the port you want to test
```
Replace 5000 with your port. If you get output, something is using it. If it returns nothing, the port is opened can be used for mlflow or other container. 


- Using `ss` (modern replacement for netstat)
```bash
ss -tulpn | grep 5000
```
This shows the process, PID, and protocol bound to that port.


<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---

### 7.3 Avoid sudo when using docker 

⚠️ **Security Implications**: Adding users to the docker group effectively grants them root-level access to the host system through Docker. This is because a user with docker group privileges can mount host filesystems into containers with root permissions, potentially compromising the host. Therefore, careful consideration of security implications is necessary before granting such access.

Run the following commmands: 

1. Add your user to the Docker group,
```bash
sudo usermod -aG docker your_username  
  # usermod: modify a user
  # -aG docker: append the user to the docker group
  # your_username: replace with your actual Linux username
```
2. log out and log back in, or run the command below to apply the group change immediately.
```bash
newgrp docker
```

3. Test with: 
```bash
docker ps # normally you need sudo 
```

⚠️ If it fails, do the followings:
```bash
# check if docker is running (active)
sudo systemctl status docker
# if not, activate it 
sudo systemctl start docker
# restart the docker group again 
newgrp docker
```



<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub> 

---
### 7.4 Re-installation of MLflow (Copy from Slack message from Dmytro)

1. Install MLflow in your dsi environment.

```bash
conda install -c conda-forge mlflow   
```
2. Make sure that current working directory in your terminal is: ./05_src/experiment_tracking/. Then In your terminal:

```bash
docker compose up --build -d   
```
3. Check if everything is running:

```bash
docker ps
```
You should see: `postgres`, `pgadmin`, `minio`, `minio-setup`, `mlflow_server`

4. MinIO Bucket Setup (only once):

```bash
docker exec -it minio mc alias set minio http://minio:9000 minio HumanAfterAll  
docker exec -it minio mc mb minio/mlflow  
docker exec -it minio mc ls minio 
```
**Note**: These credentials are just for our local setup. In real projects we’d use environment variables or secret managers.


  
<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  


  
5. ⛔ **error**: `error during container init: error mounting "/home/....../production/05_src/experiment_tracking/postgres_data" to rootfs at "/var/lib/postgresql/data"...flags=0x44000: invalid argument: unknown` <br>  
  **To Fix**:  
  Inside `docker-compose.yml`, line 17 change:   
  `./postgres_data:/var/lib/postgresql/data`  --> `./postgres_data:/var/lib/postgresql` OR `postgres_data:/var/lib/postgresql/data` 


6. ⛔ **error**: `mlflow.exceptions.MlflowException: API request to endpoint /api/2.0/mlflow/logged-models failed with error code 404 != 200` <br>  
  **To Fix**:  
  `conda install -c conda-forge mlflow=2.22.0`


7. ⛔ **error**: `Error response from daemon: failed to set up container networking: driver failed programming.....failed to bind host port for 0.0.0.0:9000:172.18.0.3:9000/tcp: address already in use`<br>  
**To Fix**: Remap the port 


8. If you have ModuleNotFoundError: No module named 'utils'  during running test_mlflow.py<br>     
**Fix**: Add these lines at the top of test_mlflow.py so it can access logger.py   

```bash
import sys, os  
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```


9. Test the run again:

```bash
python test_mlflow.py  
# You should see logs, model training, and a run created in MLflow at http://localhost:5001
# You can also run the following with the productio repo
python -m credit.exp__logistic_simple  
```


<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---
## Reference
1. [How to Set up dsi_participant environment with Miniconda](https://github.com/yikai82/UofT_DSI_onboarding/blob/093064b03e664b48f3252efa3f7a238e98e3a0d4/environment_setup/tech_onboarding_linux.md#miniconda) 

2. [GitHub: UofT-DSI/Production, from Digital Science Institute at University of Toronto](https://github.com/UofT-DSI/production)

3. [Keras/LSTM](https://keras.io/api/layers/recurrent_layers/lstm/)