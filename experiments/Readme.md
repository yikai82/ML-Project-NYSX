<!-- <p align="center">
  <img src="[insert XXX IMAGE URL]" alt="Image Place" width="120">
</p> -->

<h1 align="center">
  Yi-Kai's AI/ML New York Stock Exchange Play Zone 🛝 <br>
  </h1>


<p align="center"> 
  <a href="/production/Readme.md">Production ⛑️ 🏭 </a> •
  <!-- <a href="URL">Enter Text Here</a> • -->
  <a href="/README.md">Main Page 🏠
</a><br> 
</p>

<!-- 
> [!WARNING]  
> 1. 
> -->

> [!IMPORTANT]  
>   
> Hello, 
>
> Here is the experiment zone where you can see some notebooks where I conduct the experiment and document its finding. Obviously, there is more can be done. Check [**here**](#3-️--experiment-note) for a quick overview. 
> 
>   


### Update log: 
- 2025-11-17: Fixed a bug that assigns a wrong target column (`target_col =0`) to predict the `close` price. The correct code should be `target_col = 3`, a more robust code `target_col = df.columns.get_loc('close')` is used for address this error to ensure correct column index for `close` was used. 
- 2025-11-15: Updated test 11 to test 15
- 2025-11-13: Updated test 10: PCA and UMAP with fundamentals
- 2025-11-11: Updated and cleaned up test 1 to 9.   
- 2025-11-09: Uploaded test 4 to 9, and experiments/Readme.md 
- 2025-11-06: Uploaded test 1 to test 3


## Content  
* [1. Hypothesis and Rationale](#-hypothesis-and-rationale-)

* [2. Current Roadmap](#2--️-current-roadmap)

* [3. Experiment Note](#3-️--experiment-note)   

* [4. 🏛️ Libraries 🏛️](#4-️-libraries-️)  

* [5. ⚔️🛡️ PLAY BOOK](#5-️-️-play-book)

* [Resources](#resources) 

---

### 1. 🧠 Hypothesis and Rationale : 

**Question:** `**Can we predict the next day price by using LSTM algorithms with a minimum feature engineering?**` 

- **Hypothesis**: The traditional stock price prediction requires extensive amount of feature engineering. Most stock prediction required 20 or more featuring engineering. The present work here adopted a lean approach and assume the following limitation:

  1. Maximum epoch: 20

  2. Maximum five additional engineered features beside the open, high, low, close and volume (10 features in total)     
  
  This is to allow room for future engineering.  


- **Rationale**: 

  - The goal is to develop a Neural Network model that is quick and stable (good reproducibility)

  - ML system: Learn automatically and gain from the experience (*the experience: in plain English, more data, more observation*); however, more data or experience does not necessarily guarantee better performance.

  - **The 80/20 rule**: Roughly 80% of outcomes result from 20% of causes or inputs. Therefore, to have the model predict with a reasonable confidence level, we only need to capture the 20% the most important information or patterns.   


<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---  
### 2. 🧭 🗺️ Current Roadmap:

This idea here is focuses on isolating the core predictive patterns rather than modeling every variable. The goal is to extract essential signals using available and ready to use algorithms by adopting the 80/20 rules — to keep the model both interpretable and computationally efficient.

  1. Quick test of available common stock models (ARIMA, LSTM, etc.) with representive stocks (AAPL, INTC, MSFT) using only close price as input (Test 1 to 4). ✅ 

  2. Assess the baseline performance with random seed, regression, residual analysis to ensure the similar performance is achieved -> model stability and robustness study (Test 5 to Test 6). ✅  
  
  3. Tuned the LSTM by featuring engineering (`close_price`, `return_7d`,`streak_up`, `streak_down`,`range_ratio`) and confirm the adding new features stabilized the current model for reproducibility (Test 7, 8, 9). ✅  
  
  4. Applying PCA and UMAP for the fundamentals (Account Payable, Revenue, etc.) analysis to sector to identify possible sector and subsector for grouping (Test 10, 11). ✅  

  5. Applying PCA and UMP to the Utility Sector (Test 14) ✅ 

  6. Initial test of feeding three tickers at the same time for two separate runs:
    - Run 1 (Test 12): AAPL, INTC, MSFT ✅    
    - Run 2 (Test 13): Information Technology: AMAT + KLAC + LRCX (same subsector:semiconductor equipment) ✅ 
    - Run 3 (Test 15): Utilities: AWK + LNT + WEC ✅ 



  
<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---  
### 3. ⚗️ 🧪 Experiment Note:
  
1. Test 1 to Test 3: Conducted quick test of ARIMA and LSTM using AAPL, INTC, and MSFT.
 
2. Test 4: Deep-dived into ARIMA, no good result ❎.
 
3. **Test 5**: Deep-dived into **LSTM**, interesting results. Testing model randomness robustness (**same random seed x 3 runs**), testing the concept of prediction of the next day closing price using the last 20-day closing price.  
 
4. **Test 6**: Deep-dived into LSTM, interesting results. Testing model randomness robustness (**three different random seed x 3 runs**), testing the concept of prediction of the next day closing price using the last 20-day closing price. 
 
5. **[Test 7](NYSX_test07_AAPL_LSTM_v03.ipynb), [Test 8](NYSX_test08_INTC_LSTM_v03.ipynb), [Test 9](NYSX_test09_MSFT_LSTM_v03.ipynb)**: Added extra **feature engineering** (10 features total) to increase the model stability and testing the robustness with APPL, INTC, and MSTF.

7. [**Test 10**](NYSX_test10_sectors_PCA_UMAP.ipynb): Performed PCA and UMAP using data from *fundamentals.csv* and *security.csv* to characterize sector-level features patterns. Identified the top 10 contributing features for PC1 and PC2.   


8. [**Test 11**](NYSX_test11_subsector_tech_PCA.ipynb): Performed PCA and UMAP using data from *fundamentals.csv* and *security.csv* to characterize subsector patterns within the **Information Technology** sector. Identified the top 10 contributing features for PC1 and PC2.   

9. [Test 12](NYSX_test12_LSTM_multi_stock_v03.ipynb): Tested to feed AAPL, INTC, MSFT together and evaluation its performance. Not great.  

10. [Test 13](NYSX_test13_LSTM_multi_stock_semiconductor.ipynb): Based on the PCA, and UMAP, three tightly clustered stocks from Information technology subsectors (AMAT, KLAC, LRCX) were used to train an LSTM model jointly to evaluate the LSTM predictive performance. The results still remained suboptimal.

11. [**Test 14**](NYSX_test14_subsector_utils_PCA.ipynb): Performed PCA and UMAP using data from *fundamentals.csv* and *security.csv* to characterize subsector patterns within the **`Utilities`** sector. Identified the top 10 contributing features for PC1 and PC2.   

12. [Test 15](NYSX_test15_LSTM_multi_stock_utilitis.ipynb): Based on the PCA, and UMAP, three tightly clustered stocks Utility Sector (WEC, LNT, AWK) were used to train an LSTM model jointly to evaluate the LSTM predictive performance. The results still showed limited improvement and overall performance remained suboptimal.

13. **Training time** from different LTSM modeling tests:

    | Test | Training Time (s) | Comment                                   |
    |------|-------------------|-------------------------------------------|
    | 7    | 39.24             | AAPL                                      |
    | 8    | 23.61             | INTC                                      |
    | 9    | 28.41             | MSFT                                      |
    | 12   | 132.42            | AAPL + INTC + MSFT, poor performance      |
    | 13   | 129.64            | AMAT + KLAC + LRCX, poor performance      |
    | 15   | 141.55            | AWK + LNT + WEC, poor performance         |


<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---
## 4. 🏛️ Libraries 🏛️

```python

# utilities
import time 
import os, sys
import webbrowser

# dataframe, randome
import pandas as pd
import numpy as np
import random

# plot 
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

# Machince Learning 
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

```

<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub> 

---
## 5. ⚔️ 🛡️ PLAY BOOK

### 5.1 Concept 

1. Initial LSTM base model with only one input 

    ```text
                ┌────────────────────────────────────┐
                │           Input Layer              │
                │     (20 timesteps × 1 feature)     │
                │      Input Shape: 1393 x 20 x 1    │
                └──────────────────┬─────────────────┘
                                  │
                                  ▼
                ┌────────────────────────────────────┐
                │            LSTM Layer              │
                │  50 hidden units  (~10,400 params) │
                │                                    │
                │       Total params: 31,355         │
                │       Trainable params: 10,451     │
                │       Non-trainable params: 0      │
                │       Optimizer params: 20,904     │
                │                                    │
                └──────────────────┬─────────────────┘
                                  │
                                  ▼
                ┌────────────────────────────────────┐
                │           Dense Output             │
                │         (1 neuron output)          │
                │   Predicts next-day price value    │
                └────────────────────────────────────┘
    ```  


2. Improved LSTM model with 10 features 

    ```Text
                ┌─────────────────────────────────────────┐
                │               Input Layer               │
                │         (20 timesteps × 10 features)    │
                │          Input Shape: 1388 x 20 x 10    │
                └────────────────────┬────────────────────┘
                                    │
                                    ▼
                ┌─────────────────────────────────────────┐
                │                LSTM Layer               │
                │              50 hidden units            │
                │      Learns temporal dependencies       │
                │                                         │
                │           Total params: 36,755          │
                │           Trainable params: 12,251      │
                │           Non-trainable params: 0       │
                │           Optimizer params: 24,504      │
                └─────────────────────┬───────────────────┘
                                      │
                                      ▼
                ┌─────────────────────────────────────────┐
                │               Dense Output              │
                │                 (1 neuron)              │
                │           Predicts next-day price       │
                └─────────────────────────────────────────┘
    ```

<br>

<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub> 


<!-- ---
### 5.2  

```python
# Enter code here

```

<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub> 


---
### 5.3 



<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>   -->

---
## Resources
1. [W3School Pandas - DataFrame Reference](https://www.w3schools.com/python/pandas/pandas_ref_dataframe.asp)
2. [Forecasting: Principles and Practice](https://otexts.com/fpp3/)