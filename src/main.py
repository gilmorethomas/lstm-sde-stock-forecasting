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
    for col in df.columns:
        if df[col].dtype == 'object':
            logging.info(f'Skipping normalization for {col=}, as dtype is object') 
            continue
        else:
    # Normalize the dataset, giving a max of 1 and min of 0
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    # Impute data using data from the previous day
    df = df.fillna(method='ffill')
    # Set the index to the date
    #df = df.set_index('Date')
    return df

def create_models_dict(gbm=True, lstm=True, lstm_sde=True):
    """Creates the dictionary of predictive models to use

    Args:
        gbm (bool, optional): Whether or not to use the GBM models. Defaults to True.
        lstm (bool, optional): Whether or not to use the LSTM models. Defaults to True.
        lstm_sde (bool, optional): Whether or not to use the LSTM SDE models. Defaults to True.

    Returns:
        dict:   Dictionary of model parameters. Primary key is the model type. 
                Secondary key is the model name, with the value being the model parameters.
    """    
    models_dict = {}
    if gbm: 
        models_dict["GBM"] = None
        logging.info("Creating placeholder for GBM")
    if lstm_sde:
        models_dict["LSTM_SDE"] = {'lstm_1' : {'units': 50, 'activation': 'relu', 'input_shape': (1, 1), 'return_sequences': True},}
    if lstm:
        logging.info('Creating placeholder for lstm model using stochastic differential equations (SDE)')
        # TODO allow test train split to also be a callback to a function, so we can filter specifically on the date
        # See valid model arguments documentation https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
        models_dict["LSTM"] = {
            'lstm_5node_1' : {
                'units' : 1,
                'library_hyperparameters' : {
                    'activation' : 'relu',
                    'recurrent_activation' : 'sigmoid',
                }
            },
        }
    return models_dict

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    print("Running main.py")
    # Create an analysis object for each stock
    # Create a list of stock names
    # stock_names = ["AAPL", "AMD", "AMZN", "EA", "GOOG", "INTC", "MSFT", "NFLX", "NVDA"]
    stock_names = ["AAPL"]

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
    kwargs = {'plotting' : 
                  {'x_vars' : 'Date', 
                   'y_vars' : None}}
    analysis = AnalysisManager(raw_dir, output_dir, x_vars_to_plot = ['Date'], y_vars_to_plot=None, plotting = {'x_vars': 'Date', 'y_vars': None}, foo='bar')
    analysis.set_preprocessing_callback(preprocessing_callback)
    analysis.add_analysis_objs(stock_df_dict)

    

    analysis.preprocess_datasets()
    #analysis.validate_datasets()
    models_dict = create_models_dict(gbm=False, lstm=True, lstm_sde=False)
    analysis.set_models_for_analysis_objs(models_dict=models_dict)
    analysis.run_analysis(run_descriptive=False, run_predictive=True)
    # Print the stock names