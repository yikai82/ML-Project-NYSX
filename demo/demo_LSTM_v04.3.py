##### Demo: LSTM with single stocks ver04.3 

#### >>> This is a DEMO Script <<<< #### 

## Known issue: Unable to log artifact ## 

## Date: 2025-11-22
## Test three randome seeds(111,222,333) to assess model with different random weight initializations
## Feature engineering: vol_change, return_7d, streak_up, streak_down, range_ratio
## intergration with mlflow


#### ⚠️ Thing to do before commit your code to run ⚠️ ####
## 1. Make Sure to check for Run Nummber and Batch Number 
## 2. Please confirm target_col = [correct target ] 

#### Run 2 : IT sector, Run 1 Repeat  <<<<<
#### Single Batch 


##### >>> 1: Setup and Load Data
### >> 1.1 Load Libraries
import time

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# os.makedirs("plots", exist_ok=True)
# os.makedirs("../html", exist_ok=True)

## >> Set up logger from utils.logger 
from utils.logger import get_logger
_logs = get_logger(__name__)


## >> mlflow << 
import mlflow; print("MLflow version:", mlflow.__version__)
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

mlflow.set_tracking_uri("http://localhost:5002")

## >> setup new mlfow experiment 
exp_name = "NYSX_LSTM_demo"
experiment = mlflow.get_experiment_by_name(exp_name)

if experiment is None:
    exp_id = mlflow.create_experiment(exp_name)
else:
    exp_id = experiment.experiment_id
client = MlflowClient()
# ===================== # 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random


import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler


### >> 1.2 Load the price data, AAPL 
output_PATH = "./output"
os.makedirs(f"{output_PATH}", exist_ok=True)


Batch_1 = ['AAPL']

df_metrics_all =[]

for Tick in Batch_1: 
    output_PATH = "./output"
    model_PATH ="./output"
    os.makedirs(f"{output_PATH}", exist_ok=True)
    os.makedirs(f"{model_PATH}", exist_ok=True)
    
    RUN_NAME = f"TEST_{Tick}_LSTM_ver04.3"
    FILE_PATH = "../data/raw"
    RAW_FILE = os.path.join(FILE_PATH, "prices-split-adjusted.csv")
    
    with mlflow.start_run(run_name=RUN_NAME, experiment_id=exp_id, nested = False):

        # log ticker as a tag instead of using run_name
        mlflow.set_tag("ticker", Tick)
        mlflow.set_tag("version", "LSTM_ver04.3")

        mlflow.log_param("ticker", Tick)
        mlflow.log_param("look_back", 20)
        mlflow.log_param("epochs", 20)
        mlflow.log_param("batch_size", 16)
        mlflow.log_param("LSTM Layer 1", 50)
        # mlflow.log_param("LSTM Layer 2", 32)
        # mlflow.log_param("Dropout-in LSTM Layer 1", "dropout=0.05", "recurrent_dropout=0.05")
        # mlflow.log_param("Dropout", 0.2)

        df = pd.read_csv(RAW_FILE, parse_dates=["date"])
        df = df.groupby('symbol').filter(lambda x: len(x) > 1500) # only select more data with more than 1500 records 
        # check for NaN or Null 
        print(df.isnull().sum())
        print(f"Any NA: {df.isna().any().any()}")  # False

        print(f"Ticker: {Tick}")
        print(f"Data Exist: {Tick in df['symbol'].unique() }")

        df = df[df["symbol"] == Tick].sort_values("date")  # Example: AAPL
        df = df[["date", "open", "high", "low", "close", "volume"]].set_index("date")

        ### ======================================= ###
        ### >>      1.3 Feature Engineering      << ###
        ### ======================================= ###
        ### >> Help Function: Feature engineering
        def add_stock_features(df):
            """
            Add engineered features to a stock price DataFrame.
            
            Parameters
            ----------
            df : pd.DataFrame
                Stock data with columns: ['open','high','low','close','volume'] and date as index.
            
            Returns
            -------
            pd.DataFrame
                Original DataFrame with new features:
                - vol_change: % change in volume
                - return_7d: 7-day return %
                - streak_up: consecutive up days
                - streak_down: consecutive down days
                - range_ratio: normalized daily volatility
            """
            df = df.copy()
            
            # 1. Volume change (%)
            if 'volume' in df.columns:
                df['vol_change'] = df['volume'].pct_change()
            else:
                df['vol_change'] = np.nan

            # 2. 7-day return (%)
            df['return_7d'] = df['close'].pct_change(periods=7)

            # 3. Streak up (consecutive up days)
            streak_up = []
            count = 0
            for i in range(len(df)):
                if i == 0:
                    count = 0
                elif df['close'].iloc[i] > df['close'].iloc[i - 1]:
                    count += 1
                else:
                    count = 0
                streak_up.append(count)
            df['streak_up'] = streak_up

            # 4. Streak down (consecutive down days)
            streak_down = []
            count = 0
            for i in range(len(df)):
                if i == 0:
                    count = 0
                elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
                    count += 1
                else:
                    count = 0
                streak_down.append(count)
            df['streak_down'] = streak_down

            # 5. Range ratio (normalized daily volatility)
            if {'high', 'low'}.issubset(df.columns):
                df['range_ratio'] = (df['high'] - df['low']) / df['close']
            else:
                df['range_ratio'] = np.nan

            # Drop initial NaN from pct_change calculations
            df = df.dropna()

            return df
            # =========================================================================== #
        
        # featuring engineering
        df = add_stock_features(df)
        df.to_csv(f"{output_PATH}/df_{Tick}.csv")
        # mlflow.log_artifact(f"{output_PATH}/df_{Tick}.csv")


        # Show summary
        print(f"Number of record: {len(df)}")
        df.tail()
        target_col = df.columns.get_loc('close')
        print(f"Target: close; Column index = {target_col}")

        # =========================================================================== #
        ##### >>> 2. LSTM Model Construction 
        # Scale data
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)


        ### >> 2.1 Define a helper fucntion to create a training data as input sequence (the sliding window and look_back = 20)
        # with a multiple input columns, each y should usually predict one target (e.g., next-day close), not the full feature vector -> a

        def create_dataset(series, look_back, target_col):   # add target_col
            X, y = [], []
            for i in range(len(series) - look_back):
                X.append(series[i:i+look_back, :]) # all features for look_back days
                y.append(series[i+look_back, target_col])
            return np.array(X), np.array(y)


        # create train, test data for both x and y
        ### >>> Please confirm target_col = [correct target ]
        X, y = create_dataset(scaled, 20, target_col=target_col)  ## set look_back = 20 days, target_column = 3 
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # split for validation
        split_val = int(len(X_train)*0.2) # 20% of last training data for validation
        X_train_sub, X_val = X_train[:-split_val], X_train[-split_val:]
        y_train_sub, y_val = y_train[:-split_val], y_train[-split_val:]

        # print shape
        print(X_train.shape)
        print(y_train.shape)

        ### >> 2.2 Create a helper function

        def create_lstm_model(input_shape):
            model = Sequential([
                tf.keras.Input(shape=input_shape),
                LSTM(50, activation='tanh'), # 50 memory unit, common values are :32, 50. 64
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            return model



        # =========================================================================== #
        ##### >>> 3 Train LSTM : 3 different seeds o see if the model stable >> consistency result 
        from tqdm import tqdm
        metrics=[]
        histories=[]
        predictions=[]
        seeds=[111,222,333]

        for i in tqdm(range(3)): 
            random.seed(seeds[i])
            np.random.seed(seeds[i])
            tf.random.set_seed(seeds[i])
            ## >> Train
            n_steps = X_train.shape[1]; n_features = X_train.shape[2]
            model_lstm = create_lstm_model((n_steps,n_features))
            
            history = model_lstm.fit(X_train_sub, y_train_sub, validation_data=(X_val, y_val), 
                                    shuffle=False, epochs=20, batch_size=16, verbose=0) # time series, need to set shuffer = False
            histories.append(history)

            ## >> Predict
            y_pred = model_lstm.predict(X_test)
            n_features = scaled.shape[1]  # number of columns scaler was fit on
            # Prepare y_pred for inverse transform
            y_pred_pad = np.zeros((len(y_pred), n_features))
            y_pred_pad[:, target_col] = y_pred.flatten()   # put predicted close into the first column
            y_pred_inv = scaler.inverse_transform(y_pred_pad)[:, target_col]  # take only the close column

            # Same for y_test
            y_test_pad = np.zeros((len(y_test), n_features))
            y_test_pad[:, target_col] = y_test.flatten()
            y_test_inv = scaler.inverse_transform(y_test_pad)[:, target_col]

            residuals = y_test_inv - y_pred_inv
            
            ## >> Save individual run results
            predictions.append({
                "Run": i+1,
                "Seed": seeds[i],
                "y_test": y_test_inv.copy(),
                "y_pred": y_pred_inv.copy()
            })

            # Metrics calculation
            mse = mean_squared_error(y_test_inv, y_pred_inv)
            mae = mean_absolute_error(y_test_inv, y_pred_inv)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs(residuals / y_test_inv)) * 100
            # Compute directional accuracy
            # np.diff calculates day-to-day changes
            # np.sign gets the direction (+1 up, -1 down)
            direction_match = np.sign(np.diff(y_pred_inv, axis=0)) == np.sign(np.diff(y_test_inv, axis=0))
            direction_accuracy = np.mean(direction_match) * 100  # convert to percentage

            # --- Log metrics to MLflow (add this)
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("MAPE", mape)
            mlflow.log_metric("DirAcc", direction_accuracy)

            # --- Save results for comparison ---
            metrics.append({                            # matric is a list with dictinoary
                "Run ID": RUN_NAME,
                "Ticker": Tick,
                "Run": i+1,
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "MAPE": mape,
                "DirAcc (%)": direction_accuracy
            })
            
        df_results_metrics = pd.DataFrame(metrics)
        df_results_metrics.to_csv(f"{output_PATH}/df_result_metrics_{Tick}.csv")
        df_metrics_all.append(df_results_metrics)
        
        print(df_results_metrics.to_string(index=False, float_format="%.4f"))
        
        model_lstm.summary()
        
        # log model 
        # example_input = X_train_sub[:1]

        # mlflow.keras.log_model(
        #     model_lstm,
        #     artifact_path = "lstm_model",
        #     signature = infer_signature(X_train, model_lstm.predict(X_train))
        #     )

        # log the artifact 
        # mlflow.log_artifact(f"{output_PATH}/df_result_metrics_{Tick}.csv")
        ## ================================================================++++= ##


        ### >> [Plot] 3.2 training plot
        plt.figure(figsize=(10,6))
        for run in range(3):
            history = histories[run]
            plt.plot(history.history['loss'], label=f'Run {run+1} Train Loss')
            plt.plot(history.history['val_loss'], '--', label=f'Run {run+1} Val Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title(f'{Tick}: LSTM Training and Validation Loss Across 3 Runs')
        plt.legend()
        plt.grid(True)


        FILE_NAME = f"{Tick}_training_loss"
        plt.savefig(f"{output_PATH}/{FILE_NAME}.png", dpi=300, bbox_inches='tight')
        # mlflow.log_artifact(output_PATH)


        # =========================================================================== #
        ##### >>> 4. Model Robustness Performance Analysis
        ### >> [Plot] 4.1 Plot: Residual Plot and Histrogram >> Loop through all predictions (3 seeds)

        # Create a folder to store plots locally

        for i, pred in enumerate(predictions):
            residuals = pred["y_test"] - pred["y_pred"]
            
            # Residual over time
            plt.figure(figsize=(10,4))
            plt.plot(residuals, color="#1f77b4")
            plt.title(f"{Tick}: Residuals Over Time - Seed {pred['Seed']}")
            plt.xlabel("Time Index")
            plt.ylabel("Residual")
            plt.grid(True, alpha=0.3)

            # Save figure
            FILE_NAME = f"{Tick}_residuals_time_seed{pred['Seed']}"
            plt.savefig(f"{output_PATH}/{FILE_NAME}.png", dpi=300, bbox_inches="tight")
            plt.close()  # Close figure to free memory
            
            # Log to MLflow
            # mlflow.log_artifact(f"{output_PATH}/{FILE_NAME}.png")
            
            # Residual histogram
            plt.figure(figsize=(6,4))
            plt.hist(residuals, bins=50, alpha=0.7) #color="#ff7f0e", edgecolor="#ff7f0e"
            plt.title(f"{Tick}: Residual Distribution - Seed {pred['Seed']}")
            plt.xlabel("Residual")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)

            # Save figure
            FILE_NAME = f"{Tick}_residuals_hist_seed{pred['Seed']}"
            plt.savefig(f"{output_PATH}/{FILE_NAME}.png", dpi=300, bbox_inches="tight")
            plt.close()
            
            # Log to MLflow
            # mlflow.log_artifact(f"{output_PATH}/{FILE_NAME}.png")


        ### >> 4.2 Plot Predicted vs Actual
        ## [Plot] Overlay
        plt.figure(figsize=(6,6))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        for i, pred in enumerate(predictions):
            plt.scatter(
                pred["y_test"],
                pred["y_pred"],
                alpha=0.5,
                s=20,
                color=colors[i],
                label=f"Seed {pred['Seed']}"
            )

        # perfect prediction line
        min_val = min([pred["y_test"].min() for pred in predictions])
        max_val = max([pred["y_test"].max() for pred in predictions])
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="Perfect Fit")

        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title(f"{Tick}: Predicted vs Actual Price Scatter (All Seeds)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save to file
        FILE_NAME = f"overlay_{Tick}_predicted_vs_actual"
        plt.savefig(f"{output_PATH}/{FILE_NAME}.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Log to MLflow
        # mlflow.log_artifact(f"{output_PATH}/{FILE_NAME}.png")

        # =========================================================================== #
        ##### 5. Final Price - Date Plot and Prediction over history
        ## >> Melt date and prepare long format for plot
        all_dates = df.index

        # dates corresponding to y
        look_back = 20
        dates_y = all_dates[look_back:]  # start from day 21

        # split dates into train and test
        dates_train = dates_y[:split]
        dates_test = dates_y[split:]

        results_all = pd.DataFrame({"Date": dates_test, "Actual": predictions[0]["y_test"]})

        # Add all prediction columns
        for i, pred in enumerate(predictions):
            results_all[f"Pred{i+1}"] = pred["y_pred"]

        # Convert to long format
        results_long = results_all.melt(
            id_vars='Date',
            value_vars=['Actual', 'Pred1', 'Pred2', 'Pred3'],
            var_name='Type',
            value_name='Price'
        )

        ### >> [Interactive Plot]: Actual vs Predicted Price with Date

        ## calculate mean and std
        results_all["Pred_Mean"] = results_all[["Pred1","Pred2","Pred3"]].mean(axis=1)
        results_all["Pred_Std"] = results_all[["Pred1","Pred2","Pred3"]].std(axis=1)

        ### >> [Plot] Plotly: Actual vs Predicted Price with Date
        import plotly.graph_objects as go
        import plotly.io as pio
        # Show in browser
        pio.renderers.default = "browser"

        fig = go.Figure()

        # Actual line (gray)
        fig.add_trace(go.Scatter(x=results_all["Date"], y=results_all["Actual"],
                                mode='lines', name='Actual',line=dict(color='#727272', width=2)))

        # Mean predicted line (blue)
        fig.add_trace(go.Scatter(x=results_all["Date"], y=results_all["Pred_Mean"],
                                mode='lines', name='Pred_Mean', line=dict(color='#1f77b4', width=2)))

        # ±1 SD shaded band
        fig.add_trace(go.Scatter(x=pd.concat([results_all["Date"], results_all["Date"][::-1]]),
                                y=pd.concat([results_all["Pred_Mean"] + results_all["Pred_Std"],
                                            (results_all["Pred_Mean"] - results_all["Pred_Std"])[::-1]]),
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.5)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='±1 SD'
        ))

        # Layout with stronger grid
        fig.update_layout(width=1200,height=600,
                        title_font_size=20, xaxis_title_font_size=16,yaxis_title_font_size=16,legend_title_font_size=14,
                            #   plot_bgcolor=plot_bgcolor,        # plot area background
                            #   paper_bgcolor=background_color,   # figure background
                        margin=dict(l=50, r=50, t=60, b=50),

                        title=f"{Tick}:Actual vs Mean Predicted Prices (3 Random Seeds)",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        template="plotly_white",
                        legend_title="Series",
                        font=dict(size=13),
                        hovermode="x unified",
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='lightgray',
                            gridwidth=1.5  # increase grid thickness
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor='lightgray',
                            gridwidth=1.5  # increase grid thickness
                        )
                    )

        fig.update_xaxes(tickfont=dict(size=14))
        fig.update_yaxes(tickfont=dict(size=14))
        # fig.show()

        ## save your Plotly figure as a standalone, interactive HTML file (zoomable, hoverable, no server required).

        fig.write_html(f"{output_PATH}/{Tick}_LSTM_actual_vs_predicted.html")
        fig.write_image(f"{output_PATH}/{Tick}_LSTM_actual_vs_predicted.png", 
                        width=1200, height=600, scale=3)   

        # Log into mlflow
        # mlflow.log_artifact(f"{output_PATH}/{Tick}_LSTM_actual_vs_predicted.html")
        # mlflow.log_artifact(f"{output_PATH}/{Tick}_LSTM_actual_vs_predicted.png")



        ### >>> 5.2. Prediction Function
        ### >> create predition over the history
        def predict_over_history(df, model, scaler, ticker, look_back=20):
            """
            Generate next-day predictions over the full history of the dataset.

            Returns a DataFrame with:
            - actual_close: the actual close price
            - predicted_close: model's predicted next-day close
            - residual: difference between actual and predicted
            """
            n_features = df.shape[1]
            df_scaled = scaler.transform(df)

            predicted = []
            actual = []
            dates = []

            # slide over history
            for i in tqdm(range(look_back, len(df_scaled))):
                input_scaled = df_scaled[i-look_back:i, :].reshape(1, look_back, n_features)
                pred_scaled = model.predict(input_scaled, verbose=0)

                # inverse transform predicted close
                dummy = np.zeros((1, n_features))
                dummy[0, target_col] = pred_scaled[0, 0]
                pred_close = scaler.inverse_transform(dummy)[0, target_col]

                predicted.append(pred_close)
                actual.append(df['close'].values[i])
                dates.append(df.index[i])

            # create DataFrame
            pred_df = pd.DataFrame({
                'ticker': ticker,
                'actual_close': actual,
                'predicted_close': predicted,
                'residual': np.array(actual) - np.array(predicted)
            }, index=dates)

            return pred_df


        ### >> Test with look_back = 20
        predictions_table = predict_over_history(df, model_lstm, scaler, Tick, look_back=20)
        print(predictions_table.tail(10))  # last 10 days with predictions


        ### >> [Plor] Actual vs Predicted Price Over the History
        # Create figure
        fig = go.Figure()

        # Actual close
        fig.add_trace(go.Scatter(
            x=predictions_table.index,
            y=predictions_table['actual_close'],
            mode='lines',
            name='Actual Close',
            line=dict(color='steelblue', width=2)
        ))

        # Predicted close
        fig.add_trace(go.Scatter(
            x=predictions_table.index,
            y=predictions_table['predicted_close'],
            mode='lines',
            name='Predicted Close',
            line=dict(color="tomato", width=2.5, dash='dash')
            # dash options: 'solid', 'dash', 'dot', 'dashdot', or specific patterns (e.g., '5px 5px' for a custom dash-dot pattern).
        ))

        predictions_table.to_csv(f"{output_PATH}/prediction_history.csv")
        # mlflow.log_artifact(f"{output_PATH}/prediction_history.csv")

        # Layout with stronger grid
        fig.update_layout(width=1200,height=600,
                        title_font_size=20, xaxis_title_font_size=16,yaxis_title_font_size=16,legend_title_font_size=14,
                        # plot_bgcolor=plot_bgcolor,        # plot area background
                        # paper_bgcolor=background_color,   # figure background
                        margin=dict(l=50, r=50, t=60, b=50),

                        title=f"{Tick}: Actual vs Mean Predicted Prices (3 Random Seeds)",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        template="plotly_white",
                        legend_title="Series",
                        font=dict(size=13),
                        hovermode="x unified",
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='lightgray',
                            gridwidth=1.5  # increase grid thickness
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor='lightgray',
                            gridwidth=1.5  # increase grid thickness
                        )
                    )

        fig.update_xaxes(tickfont=dict(size=16))
        fig.update_yaxes(tickfont=dict(size=16))

        ## save your Plotly figure as a standalone, interactive HTML file (zoomable, hoverable, no server required).
        fig.write_html(f"{output_PATH}/{Tick}_LSTM_Pred_vs_Actual_history.html")
        fig.write_image(f"{output_PATH}/{Tick}_LSTM_Pred_vs_Actual_history.png", 
                        width=1200, height=600, scale=3)   
        
        # Log into mlflow
        # mlflow.log_artifact(f"{output_PATH}/{Tick}_LSTM_Pred_vs_Actual_history.html")
        # mlflow.log_artifact(f"{output_PATH}/{Tick}_LSTM_Pred_vs_Actual_history.png")


        ##### >>> Save Model
        model_lstm.save(f"{output_PATH}/{Tick}_lstm_model.keras")
        mlflow.end_run()


## Combine all the runs 
df_metrics_all = pd.concat(df_metrics_all, ignore_index=True)
df_metrics_all.to_csv(f"{output_PATH}/all_stocks_metrics.csv", index=False)
