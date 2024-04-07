import numpy as np 
import pandas as pd 
import logging 
from os import path, getcwd
import os
from analysis import AnalysisManager
# The main driver for the lstm-sde-stock-forecasting project.

# This study explores the integration of stochastic modeling and 
# advanced machine learning techniques, focusing specifically on 
# recurrent neural networks (RNNs) to forecast stock prices. 
# This will use a combination of models, including 
# (1) Geometric Brownian Motion
# (2) Long Short-Term Memory (LSTM) (subclass of RNN) 
# (3) LSTM model using stochastic differential equations (SDEs)

# We aim to compare performance using metrics such as R-CWCE, RMSE, R2, and BIC.  
# This project performs univariate, stochastic, time-dependent modeling principles and 
# evaluates performance of regressors across variable-length time horizons. 
# The study conducts experiments using a dataset comprised of 10 stocks within the technology 
# industry from 2009 to 2024, evaluating performance over 5-day, 1-month, 6-month, 1-year, and 5-year periods. 
# This data is to be read from multiple CSVs 

# Make an analysis class for each stock This class contains the dataset and the models
# This analysis should have descriptive analytics and predictive analytics functionality.

# Create analysis class with models, and dataset
# Create a class for each stock


def preprocessing_callback(df):
    # There are likely certain things that we don't want to include in the dataset...
    # TODO: Implement this function, this is a placeholder

    # Preprocess the dataset
    # Remove any missing values
    df = df.dropna()
    return df

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    print("Running main.py")
    # Create an analysis object for each stock
    # Create a list of stock names
    stock_names = ["AAPL", "AMD", "AMZN", "EA", "GOOG", "INTC", "MSFT", "NFLX", "NVDA"]
    # Create a list of stock dataframes
    stock_df_dict = {}
    raw_dir = path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'input', '00_raw')
    output_dir = path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'output')

    for stock in stock_names:
        try: 
            stock_df_dict[stock] = pd.read_csv(path.join(raw_dir, stock, f'{stock}.csv'))
        except FileNotFoundError as e:
            print(f"File not found for {stock}")
            continue
    # Create a list of analysis objects
    analysis_objects = []
    analysis = AnalysisManager(raw_dir, output_dir)
    analysis.set_preprocessing_callback(preprocessing_callback)
    analysis.add_analysis_objs(stock_df_dict)

    analysis.preprocess_datasets()
    analysis.run_analysis(run_descriptive=True, run_predictive=False)
    # Print the stock names