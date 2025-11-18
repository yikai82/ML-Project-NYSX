<!-- <p align="center">
  <img src="[insert XXX IMAGE URL]" alt="Image Place" width="120">
</p> -->

<h1 align="center">
  Yi-Kai's AI/ML New York Stock Exchange <b>Producion</b> Zone 🏭⚙️ <br>
  </h1>

## Content  
* [Experiment Note](#experiment-note)   
* [1. 🏛️ Libraries 🏛️]()  
<!-- * [2. ⚔️🛡️ PLAY BOOK](#) -->
* [3. Resources](#resources) 



## Recent Update:  



## Production Challenges Log 📑
I just write down all the mistakes I made or challenges I was facing, and how I solve it. It helps me to get better and I hope it helps you too!

1. Target Column for the Input


2. Small but yet details 




## Coding Helper after production is done 

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


---

<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub> 

---
## 2. ⚔️🛡️ PLAY BOOK


<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub> 


---
### 2.1

```python
# Enter code here

```

<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub> 


---
## 3. Function List  



<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  


---
## Resources
1. [W3School Pandas - DataFrame Reference](https://www.w3schools.com/python/pandas/pandas_ref_dataframe.asp)
2. [GitHub: UofTDSI/**Production**, from *Digital Science Institute* at University of Toronto](https://github.com/UofT-DSI/production)