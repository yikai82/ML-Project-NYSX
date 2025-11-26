<!-- <p align="center">
  <img src="[insert XXX IMAGE URL]" alt="Image Place" width="120">
</p> -->

<h1 align="center">
  Yi-Kai's AI/ML New York Stock Exchange <br>Producion Zone ⛑️ 🏭 <br>
  </h1>

<h4 align="center"> 
  <a href="/data/Readme.md">Data</a> •
  <a href="/experiments/Readme.md">Experiments Play Zone 🛝 </a> •
  <a href="/demo/Readme.md">Demo</a> •
  <a href="/results/results.md">Results and Findings 📊 </a> •
  <a href="/README.md">Main Page 🏠</a>
  </h4>
<br>

Hello, 

Welcome to the **production** zone! Here, you will find the [Error/Challenges Logs](#1-errorchallenges-log-) where I document all the mistakes I made or challenges I faced, and how I solved them (if I was able to ...😅). You will also find the instructions on how to run the production code using **MLflow** (See [here](#2)). Documenting what I have done helps me to improve, and I hope it helps you too!

**Highly recommended**: Read the Error/Challenges Log first before the production run so you don't make the same mistake I did. 


## Update Log:  
- 2025-11-24: Added instruction for running production code
- 2025-11-18: initial version


## Content  
* [1. Production Challenges](#1-production-errorchallenges-log-)  

* [2. Docker + MLflow for production]()

* [3. Cheat sheet for post-production work](#3-cheat-sheet-for-post-production-work)

* [Resources](#resources) 



---

## 1. Error/Challenges Log 📕

### 1.1 Inconsistency of the target column for the Input X or Y: 

The exercise here is to use past price as input X and to predict future price, whether we choose open price, close price, or the high price. It is important to write the code as robustly as possible. During the initial base model evaluation, the `x_input` only contains one variable — the past `close` price, so the column index is 0. However, after feature engineering, the input now contains 10 features: open, high, low, close, volume, vol_change(%), 7-day returns, streak up, streak down, and range ratio. Thus, the `close` column has been shifted to column index = 3. The issue has been addressed by implementing the following code:

<br>

```python
# 1. print all the column and its index for the data frame we are intersted in 
for idx, col in enumerate(df.columns):
    print(idx, col)

# 2. assign target_col to the target we want, in this case `close```
target_col = df.columns.get_loc('close')
print(f"Target: close; Column index = {target_col}")
```

<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---

### 1.2 Small, yet details always matters
    
When running experiments in **MLflow**, it is important to have a strategy to ensure all the experiment variables and paths are set correctly so the output goes to the correct destination. For example, if the experiment ID is `Run_2`, obviously, we don't want the outcome from `Run_2` to end up in the `Run_1` folder. In the worst case, **overwrite** what is in Run_1. 

**Solution:**   
(a) Implement a small comment section at the top of code as a checklist and test with a **test** script    

```python
##### Evaluation of LSTM with Multiple Stocks ver04.4

#### ⚠️ Checklist before commit your code to run ⚠️ ####
## 1. Check for Run Number : Run_X: 3 places, RunX: 2 places
## 2. Check the Batch Number : batch/Batch: {2 places}
## 3. Confirm target_col = [correct target] before commit the run 
``` 

<br>
(b) Create a test script and test_folder, testing everything, then replace the test_folder with the actual intended folder. 


<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---

### 1.3 The mystery of duplication runs in the **MLflow** run  

During the testing phase, when only one ticker was selected to run in MLflow, I got two run results in MLflow: the first one only lasted for 7-10s while the second one is the actual run. After debugging, this was a very common **indentation problem**. When doing multiple MLflow runs, it is important to have a proper indentation.  

```python
### Example ### 
for Tick in Batch_1: 

# Some code here.....
    with mlflow.start_run(run_name=RUN_NAME, experiment_id=exp_id, nested = False):

        # log ticker as a tag instead of using run_name....
        mlflow.set_tag("ticker", Tick)
        mlflow.set_tag("version", "LSTM_ver04.3")

        # ...some code here.... 
        
        def add_stock_features(df):
            # ... some code here....
            return df
            # =========================================================================== #
        
        ##### >>> 3 Train LSTM : 3 different seeds o see if the model stable >> consistency result 
        from tqdm import tqdm
        metrics=[]
        histories=[]
        predictions=[]
        seeds=[111,222,333]

        for i in tqdm(range(3)):   ## <<< correct indent
            random.seed(seeds[i])
            np.random.seed(seeds[i])
            tf.random.set_seed(seeds[i])
            ### ... more code here ...
``` 


<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub> 

---

## 2. Docker + MLflow for production

The step is very similar to what was already described in [Demo](/demo/Readme.md). We will only need to replace the last script `demo_LSTM_v04.3.py` we run with in demo with the production script `run1_batch1_LSTM_v04.4.py`.  

⚠️ All the code you need for production run is under /src/experiment_tracking: 

```text.
.
├── backup
├── minio
├── mlflow
├── mlruns
├── postgres
├── docker-compose.yml
├── run1_batch1_LSTM_v04.4.py
├── run_TEST_LSTM_v04.3.py
├── test_mlflow_artifact.py
└── test_mlflow.py
```

1. Make sure you have [Docker and the environment](/README.md) set up properly. I will use the `dsi_participant` environment as an example here. Feel free to use any appropriate environment.

2. Start Docker and activate your environment in the terminal:   

    ```bash
    cd /path/to/ML3-Team-Project/demo
    conda activate dsi_participant  # activate environment 
    docker compose up -d # compose up docker 
    docker ps # should show the list of running containers.
    ```  
3. If everything passes, you should be able to see the terminal output similar to [this](/demo/images/demo_docker.png).

4. Run the following to test Docker + MLflow

    ```bash
    python test.py # this test mlflow.log_param and mlflow.log_metric functions 
    python test_mlflow.py # test mlflow with a simple logistic regression 
    ```

    If both are passed, you should be able to open link (http://localhost:5001/#/experiments/0) and (http://localhost:5001/#/experiments/1) and see something like this: 

<br>   
<img src="images/demo_mlflow.png" width="1200" align="left" style="margin-left: 20px; margin-bottom: 40px;">

<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---
5. Inspect `run_TEST_LSTM_v04.3.py` and run it

    - Open `run_TEST_LSTM_v04.3.py` in any command line editor or VScode, and review the paths if you want to modify anything. The following are just a few examples of editable paths: 
        - `exp_name = "NYSX_LSTM_TEST_predict_close"`
        - `output_PATH = "../../production/TEST"`
        - `output_PATH = "../../production/TEST"` (inside the for loop)
        - `model_PATH = "../../models/TEST"`
        - `RUN_NAME = f"TEST_{Tick}_LSTM_ver04.4"`
        - `FILE_PATH = "/../data/raw"`
        - `fig.write_html(f"{output_PATH}/{Tick}_filename.html")`
        - `fig.write_image(f"{output_PATH}/{Tick}_filename.html")`
        - `model_lstm.save(f"{output_PATH}/{Tick}_lstm_model.keras")`

    👉 Highly recommended: use **Ctrl + F** and search for keywords such as "save" or "write", etc.

    - After that, run the script in the terminal:  

    ```bash
    python run_TEST_LSTM_v04.3.py
    ```

    - You should see it run two tickers (AAPL and ADBE) successfully in the terminal, and you can access the result at: http://localhost:5001/#/experiments/2 


<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---

6. Check the output folder (production/TEST/) to ensure it contains all the expected output files (a total of 16 files). You should also see a file named `all_stocks_metrics.csv` under production/TEST. 

    ```text
    .
    ├── AAPL
    │   ├── AAPL_LSTM_actual_vs_predicted.html
    │   ├── AAPL_LSTM_actual_vs_predicted.png
    │   ├── AAPL_LSTM_Pred_vs_Actual_history.html
    │   ├── AAPL_LSTM_Pred_vs_Actual_history.png
    │   ├── AAPL_residuals_hist_seed111.png
    │   ├── AAPL_residuals_hist_seed222.png
    │   ├── AAPL_residuals_hist_seed333.png
    │   ├── AAPL_residuals_time_seed111.png
    │   ├── AAPL_residuals_time_seed222.png
    │   ├── AAPL_residuals_time_seed333.png
    │   ├── AAPL_training_loss.png
    │   ├── all_stocks_metrics.csv
    │   ├── df_AAPL.csv
    │   ├── df_result_metrics_AAPL.csv
    │   ├── overlay_AAPL_predicted_vs_actual.png
    │   └── prediction_history.csv
    ```

<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---

7. If everything looks correct, you can inspect `run1_batch1_LSTM_v04.4.py` and repeat the same process described in **step 5**, then run it. The script contains 56 [**tickers**](/production/ticker.csv) from the Information Technology sector. 

    **Highly recommended**: Run in batches (18-19-19) the first time to ensure everything is set up properly. Since the current base LSTM model is very light, it will take only 20 seconds per stock.  

    ⚠️ Double-check the output path, filename, and naming convention before running the script.


8. Stop and Shutdown Docker

    ```bash
    docker compose stop # Stop the container
    docker compose down # Shutdown the docker, aitifacts will be save preperly. or use flag -v for nuclear option to wipe out all the containers and data.  

    ⚠️ `docker compose down -v` will delete all container volumes and reset everything to a clean state. ONLY Use this if there is no real production or important data you would like to keep.


<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub> 

---

## 3. Cheat sheet for post-production work  

### Bash

1. **Copy-Past `*.csv` file with specific name into the same destination**  
  **Background**: Since I have created each stock having its own folder (just want to keep it neat and avoid data contamination), it is very common that I'll need to extract and consolidate some information there. The example below shows how to copy all the `df_result_metrics_[tikcer].csv` into a single folder in terminal 

```
    ├── AAPL
    │   ├── AAPL_LSTM_actual_vs_predicted.html
    │   ├── AAPL_LSTM_actual_vs_predicted.png
    │   ├── AAPL_LSTM_Pred_vs_Actual_history.html
    │   ├── AAPL_LSTM_Pred_vs_Actual_history.png
    │   ├── AAPL_residuals_hist_seed111.png
    │   ├── AAPL_residuals_hist_seed222.png
    │   ├── AAPL_residuals_hist_seed333.png
    │   ├── AAPL_residuals_time_seed111.png
    │   ├── AAPL_residuals_time_seed222.png
    │   ├── AAPL_residuals_time_seed333.png
    │   ├── AAPL_training_loss.png
    │   ├── df_AAPL.csv
    │   ├── df_result_metrics_AAPL.csv
    │   ├── overlay_AAPL_predicted_vs_actual.png
    │   └── prediction_history.csv
```

```bash
cd /path/to/the/parent/folder  # just one level up of the Ticker folder 
find -type f -name "*result*.csv" > csv_file.txt # look for any .csv containg 'result' and pipe into a text to make sure
find -type f -name "*result*.csv" -exec cp {} ../../data/TEMP/ # copy those into a target folder called 'TEMP'
``` 


<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  
 
---

2. **Copy-Paste all the `*.csv` file does **NOT** contain specific name into the same destination**  
**Background**: I want to put all the processed data that does **NOT** contain word `result` or `prediction` into a single folder. Basically, I want to collect all the df_[Ticker}.csv into a single folder 

```
    ├── AAPL
    │   ├── AAPL_LSTM_actual_vs_predicted.html
    │   ├── AAPL_LSTM_actual_vs_predicted.png
    │   ├── AAPL_LSTM_Pred_vs_Actual_history.html
    │   ├── AAPL_LSTM_Pred_vs_Actual_history.png
    │   ├── AAPL_residuals_hist_seed111.png
    │   ├── AAPL_residuals_hist_seed222.png
    │   ├── AAPL_residuals_hist_seed333.png
    │   ├── AAPL_residuals_time_seed111.png
    │   ├── AAPL_residuals_time_seed222.png
    │   ├── AAPL_residuals_time_seed333.png
    │   ├── AAPL_training_loss.png
    │   ├── df_AAPL.csv   # <==== move into a single folder  
    │   ├── df_result_metrics_AAPL.csv
    │   ├── overlay_AAPL_predicted_vs_actual.png
    │   └── prediction_history.csv
```

```bash
cd /path/to/the/parent/folder  # just one level up of the Ticker folder 

find -type f -name "*.csv" \
! -name "*prediction*" \
! -name "*result*" > csv_file.txt  # pipe into text file to check 


find -type f -name ".csv"\
! -name "*prediction*" \
! -name "*result*" \
-exec cp {} ../to/target/folder/ \; # copy to the target folder  
``` 


<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---
### Python
1. Find all the *.csv file and convert it to a single data frame

```Python
## make sure create a jupyternotebook in the working directory
import pandas as pd
import glob
import os

# Path to your folder of CSV files
# path = "path/to/your/folder"

# Get all CSV files
# files = glob.glob(os.path.join(path, "*.csv"))

files = sorted(glob.glob("*.csv")) # only look for *.csv in the current folder 

df_list = []

for f in files:
    print("Loading:", os.path.basename(f))
    df = pd.read_csv(f)
    
    # Drop unwanted index column if it exists
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    
    df_list.append(df)

# Combine into one DataFrame (keeps all columns from all files)
combined_df = pd.concat(df_list, ignore_index=True, sort=False)
combined_df.head()

# save as a new csv file
combined_df.to_csv("all_stocks_metrics_run3.csv", index=False)
```


<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub> 

<!-- ---
## 2. ⚔️🛡️ PLAY BOOK


<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub> 

<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub> 

 -->

---
## Resources
1. [W3School Pandas - DataFrame Reference](https://www.w3schools.com/python/pandas/pandas_ref_dataframe.asp)

2. [GitHub: UofTDSI/**Production**, from *Digital Science Institute* at University of Toronto](https://github.com/UofT-DSI/production)

3. [Numpy: Learn/Numpy tutorials](https://numpy.org/numpy-tutorials/)