Update Date: 2025-11-17

> [!NOTE]  
>
> The `data` folder contains the followings:
>   - Proccessed data:any file that is considered processed data. This includes:
>       - Raw data that has been processed (e.g. after feature engineering).
>       - Consolidated data generated from modeling work.
> 
>   - Raw data: the New York Stock Exchange (NYSX) data set download from Kaggle.You can also access them from [here](https://www.kaggle.com/datasets/dgawlik/nyse?select=securities.csv) 



---

1. Processed data: 

    - `df_[ticker]`: A file containing an individual stock price data (date, open, high, low, close, volume) and engineered features (vol_change, return_7d, streak_up, streak_down, range_ratio). 

    - `all_stocks_metrics`: Contains all evaluation metrics (MSE,RMSE, MAE, MAPE, DirAcc) from both runs   

2. Raw data: 
    - **fundametnals.csv**:  
    Contains company-level financial indicators (revenue, earnings, assets, liabilities, ratios, etc.). These features form the backbone of all fundamental analysis and help uncover deeper structure in sector and subsector patterns.  

    - **price-split-adjusted.csv**:  
    Provides historical daily prices adjusted for stock splits. The columns include Date, Symbol, Open, High, Low, Close, and Volume. Uses for time-series modeling. 

    - price.csv  
    The unadjusted historical daily price data, only useful when you want the raw, unaltered market behavior—including gaps, splits, etc. 

    - securities.csv:  
    Includes metadata for each ticker: sector, subsector, company name, IPO date, and other classification details. Essential for grouping, filtering, and building tPCA/UMAP clusters.


--- 
Acronyms:  

- MSE: Mean Squared Error
- RMSE: Root Mean Squared Error  
- MAE: Mean Absolute Error  
- MAPE: Mean Absolute Percentage Error (%)  
- DirAcc: Directional Accuracy (%)
 