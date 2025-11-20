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
> <i><h4 align="center">Most of the business problems are not ML problems, and most of the ML problems are not business problems. Optimizing an ML model is not the same thing as optimizing a solution for a business problem </h3><p>  
> <h5 align="right"> - from Production Lecture Day 1 : 36:58</h5><p>
>
> <h4 align="center">A Machine Learning System is a system that can learn automatically to improve its performance</h5></i>    
> <br>  
>
> This capstone project showcases what I learned during a 16-week intensive AI/ML course offered by the University of Toronto’s Data Science Institute. I am not a financial professional, but I do invest in the market as a side pursuit, chasing the occasional moonshot 🌛 🏹.

---

<p align="center">
   ➡️ Navigate to other place:
  <a href="/data/Readme.md"><b>Data</a> •
  <a href="/experiments/Readme.md">Experiment</a> •
  <a href="/production/Readme.md">Production</a><br></b>
  <br>
  **Continue Reading 👇**<br>
</p>

---
### Content

* [1. Business Problem](#1-business-problem)    
* [2. Repo Layout](#2-repo-layout)
* [3. Environment Setup ](#3-environment-setup)
* [4. Setup MLflow with python Script for Experiment Tracking ](#4-setup-mlflow-with-python-script-for-experiment-tracking)


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

<img src="images/trump_tweet.jpg" width="350" align="right" style="margin-left: 40px; margin-bottom: 10px;">
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
├── experiments
├── html
├── images
├── models
├── production
├── results   
├── src
├── AAPL_example.html
├── LICENSE
├── ml_env_complete.txt
├── ml_env_complete.yml
├── README.md
├── requirements.txt
└── requirements.yml


```
- data: Contains both raw data and processed data. Click [here](/data/Readme.md) for more details.  

- experiments: Contains all the notebooks for experiments performed here. Click [here](/experiments/Readme.md) for more details.

- html: Default path for interactive plots generated from notebooks:  
    `fig.write_html(f"../html/{Tick}_actual_vs_predicted.html)`

- models: Default path for saving models during production runs with MLflow:
    ` model_lstm.save(f"../../models/{Tick}_lstm_model.keras")`  
    - This default path ensures that models are first stored in a central location. After the production run completes, the models are moved to the specific run folder (e.g., `Run_3`) to avoid overwriting models from previous runs. Currently, Run_3 contains 56 models from the latest production run.

- productions: Default path for saving output *.csv and plots (*.pmg) during production runs with MLflow:


``` bash
# for example 
output_PATH = f"../../production/Run_X"
os.makedirs(f"{output_PATH}", exist_ok=True)
# some code here...

FILE_NAME = f"{Tick}_training_loss"
plt.savefig(f"{output_PATH}/{FILE_NAME}.png", dpi=300, bbox_inches='tight')

```

- results: Contains a high level summary of the finding from the work 

- src: Contains source code for experiment trackering (using MLflow), logs, and utilities (utils).

  
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

- Now following the following command one-by0one in terminal to install the required package.

```bash
conda create --name dsi_participant python=3.9 # create a new conda environment 

conda activate dsi_participant # activate the new conda enviroment 

conda install -c conda-forge numpy requests ipykernel pandas seaborn scikit-learn python-dotenv dask "pyarrow>=11.0.0" sacred sqlalchemy psycopg2 shap fancyimpute missingno tensorflow matplotlib plotly nbformat scikit-image opencv transformers yfinance pygam pybind11

conda list # this should return a list where you can save as txt for future reference. 
``` 

<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---
### 3.2 Set up environment with a clean envirmental.yml

- I exported my conda environment(`ml_env_complete.yml`) and added to the repo here, which should be a cleaning a guarantee work around


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
### 3.3 Set up minimum environment with a minimum.yml

Same step as below just replace the `ml_env_complete` to `requirement` for either yml file or text file based on your setting. 

```bash

## Option 1: For conda user
cd path/to/the/folder 
conda env create -f requirements.yml # This will create a conda enviroment with the same name defined in the environmet.yml. 
                                    # In this case, the name will be "ml_env_complete" 

## Option 2: For python venv user  
python -m venv myenv # replace [myenv] to whatever name you like

# for Linux / Mac
source myenv/bin/activate

# for Windows
myenv\Scripts\activate

# install virtual environment 
pip install -r requirements.txt
```


--
## 4. Setup MLflow with python Script for Experiment Tracking







## Reference
1. [How to Set up dsi_participant environment with Miniconda](https://github.com/yikai82/UofT_DSI_onboarding/blob/093064b03e664b48f3252efa3f7a238e98e3a0d4/environment_setup/tech_onboarding_linux.md#miniconda) 

2. [GitHub: UofTDSI/Production, from Digital Science Institute at University of Toronto](https://github.com/UofT-DSI/production)

3. [Keras/LSTM]()