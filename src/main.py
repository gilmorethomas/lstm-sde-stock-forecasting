import numpy as np 
import pandas as pd 
from lstm_logger import logger as logging
from os import path
import os
from analysis_manager import AnalysisManager # The main driver for the lstm-sde-stock-forecasting project.
from model_parameters import create_models_dict

def preprocessing_callback(df):
    # There are likely certain things that we don't want to include in the dataset...
    # TODO: Implement this function, this is a placeholder

    # Preprocess the dataset
    # Remove any missing values
    df = df.dropna()
    # for col in df.columns:
    #     if df[col].dtype == 'object':
    #         logging.info(f'Skipping normalization for {col=}, as dtype is object') 
    #         continue
    #     else:
    # # Normalize the dataset, giving a max of 1 and min of 0
    #         df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    # Where there is not an entry for a given day, use the data from the previous day
    # convert date to datetime
    df['Date_string'] = df['Date']
    df['Date'] = pd.to_datetime(df['Date'])
    # Reindex the dataframe to include all dates
    idx = pd.date_range(df['Date'].min(), df['Date'].max())
    idx.name = 'Date'
    # Set missing dates to NaN for detection, to later impute
    df = df.set_index('Date').reindex(idx, fill_value=np.nan)
    # s.index = pd.DatetimeIndex(s.index)
    logging.info("NEED TO PULL INFORMATION ABOUT HOW MUCH DATA IS MISSING, NEED TO ADD A COLUMN THAT IT WAS IMPUTED ")
    # Add indicators for columns that are imputed
    for col in df.columns: 
        if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            df[f'{col}_imputed_indicator'] = df[col].isnull().astype(int)

    # Impute data using data from the previous day
    df = df.fillna(method='ffill')
    
    # Add the date column back in by resetting index
    df = df.reset_index(drop=False)

    # Calculate a days since start column, returning it as an integer
    df['Days_since_start'] = (df['Date'] - df['Date'].min()).dt.days
    return df


if __name__ == "__main__":
    # OPTIONS ARE INFO, ERROR, and CRITICAL. 
    # INFO will print out a ton of stuff, but is useful for debugging
    # Set the logger 
    #Options are DEBUG, INFO, WARNING, ERROR, and CRITICAL. These are increasing order and will change what gets printed out
    logging.setLevel("INFO")
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
        except FileNotFoundError:
            print(f"File not found for {stock}")
            continue

    analysis_objects = []
    kwargs = {'plotting' : 
                  {'x_vars' : 'Date', 
                   'y_vars' : None}}
    analysis = AnalysisManager(raw_dir, 
        output_dir, 
        x_vars_to_plot = ['Date'], 
        y_vars_to_plot=None, 
        save_png=False,
        save_html=True,
        plotting = {'x_vars': 'Date', 'y_vars': None}, 
        overwrite_out_dir=False,
        load_previous_results=False # Load previous results, rather than refitting a model
        )
    analysis.set_preprocessing_callback(preprocessing_callback)
    # Add the analysis objects to the analysis manager 
    analysis.add_analysis_objs(analysis_dict=stock_df_dict, x_vars=['Date', 'Days_since_start'], y_vars=['Close'])
    
    analysis.preprocess_datasets()
    #analysis.validate_datasets()
    models_dict = create_models_dict(gbm=False, lstm=True, lstm_sde=False)
    analysis.set_models_for_analysis_objs(models_dict=models_dict)
    analysis.run_analysis(run_descriptive=False, run_predictive=True)
    # Print the stock names