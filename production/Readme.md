<!-- <p align="center">
  <img src="[insert XXX IMAGE URL]" alt="Image Place" width="120">
</p> -->

<h1 align="center">
  Yi-Kai's AI/ML New York Stock Exchange <br>Producion Zone ⛑️ 🏭 <br>
  </h1>

<p align="center"> 
  <a href="/experiments/Readme.md">Experiments Play Zone 🛝 </a> •
  <!-- <a href="URL">Enter Text Here</a> • -->
  <a href="/README.md">Main Page 🏠
</a><br> 
</p>


Hello, 

Welcome to the **production** ⛑️ 🏭 zone, where you can find the instructions on how to run the production code using **MLflow**. I also spent time documenting all the mistakes I made or challenges I was facing, and how I solved them (If I was able to ...😅). It helps me get better, and I hope it helps you too!

## Update log:  
- 2025-11-18: initial version


## Content  
* [1. Production Challenges](#production-challenges-log-)   

* [2. Cheat sheet for post-production work](#cheat-sheet-for-post-production-work)

* [Resources](#resources) 



---

## 1. Production Error/Challenges Log 📕
- Inconsistency of the target column for the Input X or Y: 

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

- Small, yet details always matters
    
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

3. The mystery of duplication in the **MLflow** run  

    During the testing phase, when only one ticker was selected to run in MLflow, I got two run results in MLflow: the first one only lasted for 7-10s while the second one is the actual run. After debugging, this was a very common indentation problem. When doing multiple MLflow runs, it is important to have a proper indentation.  

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

    ``` 


<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub> 

---

## 2. Cheat sheet for post-production work  

### Bash

1. Copy-Past *.csv file with specific name into the same destination 
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

2. Copy-Paste all the *.csv file does **NOT** contain specific name into the same destination  
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
    │   ├── df_AAPL.csv
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