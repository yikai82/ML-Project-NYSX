
<h4 align="center">
  <a href="/experiments/Readme.md">Experiment</a> •
  <a href="/demo/Readme.md">Demo</a> •
  <a href="/production/Readme.md">Production 🏭 </a> •
  <a href="/results/results.md">Results and Findings 📊 </a> •
  <a href="/README.md">Main Page 🏠 </a><br> 
  <br></b>
</h4>

---

Update Date: 2025-11-17

> [!NOTE]  
>
> The `data` folder contains the followings:
>   - Proccessed data: any file that is considered processed data. This includes:
>       - Raw data that has been processed (e.g. after feature engineering).
>       - Consolidated data generated from modeling work.
> 
>   - Raw data: the New York Stock Exchange (NYSX) data set download from Kaggle. You can also access them from [here](https://www.kaggle.com/datasets/dgawlik/nyse) 

---

1. Processed data: 

    - `df_[ticker]`: A file containing an individual stock price data (date, open, high, low, close, volume) and engineered features (vol_change, return_7d, streak_up, streak_down, range_ratio). 

    - `all_stocks_metrics`: Contains all evaluation metrics (MSE,RMSE, MAE, MAPE, DirAcc) from both runs   

2. Raw data: 
    - **fundametnals.csv**:  
    Contains company-level financial indicators (revenue, earnings, assets, liabilities, ratios, etc.). These features form the backbone of all fundamental analysis and help uncover deeper structure in sector and subsector patterns.  
      - ***Identifier***: 'Ticker Symbol', 'Period Ending', 'For Year, 'Estimated Shares Outstanding'.
      - ***Income Statements***: 'Total Revenue', 'Cost of Revenue', 'Gross Profit', 'Operating Income', 'Operating Margin', 'Earnings Before Interest and Tax', 'Earnings Before Tax', 'Net Income', 'Net Income Adjustments', 'Net Income Applicable to Common Shareholders', 'Net Income-Cont. Operations', 'Income Tax', 'Research and Development', 'Sales, General and Admin.', 'Non-Recurring Items', 'Other Operating Items', 'Other Operating Activities', 'Other Investing Activities', 'Other Financing Activities', "Add'l income/expense items", 'Interest Expense'.
      - ***Balance Sheet Items***:  
          (a) `Asset`: 'Total Assets', 'Current Assets', 'Other Current Assets', 'Cash and Cash Equivalents', 'Net Receivables', 'Inventory', 'Changes in Inventories', 'Fixed Assets', 'Intangible Assets', 'Goodwill', 'Investments', 'Long-Term Investments', 'Other Assets'

          (b) `Liabilities`: 'Total Liabilities', 'Current Liabilities', 'Other Current Liabilities', 'Long-Term Debt', 'Short-Term Debt / Current Portion of Long-Term Debt', 'Other Liabilities', 'Deferred Liability Charges'.  

          (c) `Equity`: 'Total Equity', 'Retained Earnings', 'Capital Surplus', 'Common Stocks', 'Misc. Stocks', 'Treasury Stock', 'Other Equity', 'Equity Earnings/Loss Unconsolidated Subsidiary'.

      - ***Cash Flow Items***: 'Net Cash Flow', 'Net Cash Flow-Operating', 'Net Cash Flows-Investing', 'Net Cash Flows-Financing', 'Sale and Purchase of Stock', 'Net Borrowings', 'Depreciation'

      - ***Financial Ratio & Metrics***: 'Profit Margin', 'Gross Margin', 'Operating Margin', 'Quick Ratio', 'Current Ratio', 'Cash Ratio', 'Pre-Tax Margin', 'Pre-Tax ROE', 'After Tax ROE', 'Earnings Per Share'

      - ***Miscellaneous***: 'Effect of Exchange Rate', 'Other Operating Activities', 'Other Investing Activities', 'Other Financing Activities', 'Minority Interest'  
      
    ---

    - **price-split-adjusted.csv**:  
    Provides historical daily prices adjusted for stock splits. The columns include Date, Symbol, Open, High, Low, Close, and Volume. Uses for time-series modeling. 

    ---

    - price.csv  
    The unadjusted historical daily price data, only useful when you want the raw, unaltered market behavior—including gaps, splits, etc. 

    ---

    - **securities.csv**:  
    Includes metadata for each ticker: sector, subsector, company name, IPO date, and other classification details. Essential for grouping, filtering, and building tPCA/UMAP clusters.


--- 
Acronyms:  

- MSE: Mean Squared Error
- RMSE: Root Mean Squared Error  
- MAE: Mean Absolute Error  
- MAPE: Mean Absolute Percentage Error (%)  
- DirAcc: Directional Accuracy (%)
 
 <sub>[Return Main Page 🏠](/README.md) </sub> 