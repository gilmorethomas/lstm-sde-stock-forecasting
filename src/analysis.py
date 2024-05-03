import os
from os import path, makedirs
import re
import pandas as pd
from numpy.random import RandomState
import tensorflow as tf
import torch

from lstm_logger import logger as logging
from project_globals import DataNames as DN
from plotting import plot_all_x_y_combinations
from lstm import LSTM 
from geometricbrownianmotion import GeometricBrownianMotion
from lstm_sde import LSTMSDE_to_train as LSTMSDE
            
class Analysis(): 
    def __init__(self, dataset_name, dataset_df, x_vars, y_vars, output_directory, seed, preprocessing_callback=None, save_html=True, save_png=True, load_previous_results=False):
        self.dataset_name = dataset_name
        self._raw_dataset_df = dataset_df
        self.preprocessing_callback = preprocessing_callback
        self.output_directory = output_directory
        self.x_vars = x_vars
        self.y_vars = y_vars
        self.save_html = save_html
        self.save_png = save_png
        self._rand_state_mgr = seed
        self.load_previous_results = load_previous_results
        if not(path.exists(output_directory)):
            logging.info(f"Creating output directory {output_directory}")
            makedirs(output_directory)
        if self.load_previous_results:
            self.load_from_previous_output()
    def preprocess_dataset(self):
        if self.load_previous_results: 
            return
        # preprocess the dataset, using the preprocessing_callback if provided
        logging.info(f"Preprocessing dataset for {self.dataset_name} using {self.preprocessing_callback}")
        if self.preprocessing_callback is not None: 
            self.dataset_df = self.preprocessing_callback(self._raw_dataset_df)
        # Get rid of columns that are not x_vars or y_vars and say which columns we are removing 
        logging.info(f"Removing columns {self.dataset_df.columns.difference(self.x_vars + self.y_vars)}")
        self.dataset_df = self.dataset_df[self.x_vars + self.y_vars]
        logging.info(f"Columns remaining {self.dataset_df.columns}")
        # Float 64 columns not supported by some models, so convert to float32
        float64_cols = list(self.dataset_df.select_dtypes(include=['float64']).columns)
        if float64_cols is not None and len(float64_cols) > 0:
            logging.info(f"Converting columns {float64_cols} to float32")
            self.dataset_df[float64_cols] = self.dataset_df[float64_cols].astype('float32')
        self.dataset_df.to_csv(path.join(self.output_directory, f'{DN.all_data}.csv'))
        # Normalize data using minmax scaling
        # self.dataset_df = (self.dataset_df - self.dataset_df.min()) / (self.dataset_df.max() - self.dataset_df.min())


    def run_analysis(self, run_descriptive=True, run_predictive=True): 

        logging.info(f"Running analysis for {self.dataset_name}")
        if run_descriptive: 
            self.run_descriptive()
        if run_predictive: 
            self.run_predictive()
    def run_descriptive(self):
        # run descriptive analytics
        self.report_stats() 
        self.run_plots(plot_types=["line", "scatter"])
        # TODO Perform seasonal and trend analysis like in this link: https://www.geeksforgeeks.org/what-is-a-trend-in-time-series/


    def report_stats(self):
        # report stats for each analysis object
        # Report mean, standard deviation, median, min, max, and number of observations
        # build dataframe for each column's stats
        stats_df = self.dataset_df.describe().T
        # stats_df["skew"] = self.dataset_df.skew()
        # stats_df["kurtosis"] = self.dataset_df.kurtosis()
        # stats_df["missing_values"] = self.dataset_df.isnull().sum()
        # stats_df["unique_values"] = self.dataset_df.nunique()
        # stats_df["dtype"] = self.dataset_df.dtypes
        # stats_df["range"] = stats_df["max"] - stats_df["min"]
        # stats_df["mean_absolute_deviation"] = self.dataset_df.mad()
        # stats_df["variance"] = self.dataset_df.var()
        # stats_df["standard_deviation"] = self.dataset_df.std()
        # stats_df["coefficient_of_variation"] = stats_df["standard_deviation"] / stats_df["mean"]
        # stats_df["interquartile_range"] = self.dataset_df.quantile(0.75) - self.dataset_df.quantile(0.25)
        # stats_df["outliers"] = self.dataset_df[(self.dataset_df < (self.dataset_df.quantile(0.25) - 1.5 * stats_df["interquartile_range"])) | (self.dataset_df > (self.dataset_df.quantile(0.75) + 1.5 * stats_df["interquartile_range"]))].count()
        # stats_df["outlier_percentage"] = stats_df["outliers"] / stats_df["count"]
        # stats_df["z_score"] = (self.dataset_df - self.dataset_df.mean()) / self.dataset_df.std()
        output_file = path.join(self.output_directory, f'{self.dataset_name}_stats.csv')
        logging.info(f"Stats for {self.dataset_name} being written to {output_file}")

        stats_df.to_csv(output_file)

    def run_plots(self, plot_types):
        # run plots for each analysis object
        self.plot_dataset(plot_types)
        # also create an all-in-one plot with all the datasets, adding an extra variable that is 
        # self._run_all_in_one_plot()
        # Run descriptive time series analysis 

    def run_predictive(self):
        # run predictive analytics
        # Validate the models dictionary 
        self._validate_models()
        # Create the test and train datasets based on the 
        # For each model in the models dict, call the requisite model class's init method
        # For each model in the models dict, call the requisite model class's train method
        # For each model in the models dict, call the requisite model class's test method
        # For each model in the models dict, call the requisite model class's predict method
        # For each model in the models dict, call the requisite model class's save method

        for model_type, all_models_for_type in self.models_dict.items():
            if model_type.lower() == 'lstm' and all_models_for_type is not None and len(all_models_for_type) > 0: 
                logging.info("Creating LSTM models")
                for model_name, model_dict in all_models_for_type.items():
                    logging.info(f"Creating LSTM model {model_name}")
                    if not self.load_previous_results: 
                        model = LSTM(data=self.dataset_df, 
                            model_hyperparameters=model_dict['library_hyperparameters'],
                            units = model_dict['units'],
                            save_dir=path.join(self.output_directory, 'lstm', model_name), 
                            model_name=model_name,
                            x_vars=self.x_vars,
                            y_vars=self.y_vars,
                            seed=self._rand_state_mgr,
                            test_split_filter=model_dict['test_split_filter'],
                            train_split_filter=model_dict['train_split_filter'],
                            evaluation_filters=model_dict['evaluation_filters'], 
                            save_png=self.save_png,
                            save_html=self.save_html)
                        self._call_model_funcs(model)
                    else: 
                        model = LSTM.load_from_previous_output(LSTM, path.join(self.output_directory, 'lstm', model_name), model_name)
                    # Save the model off in the models_dict
                    self.models_dict[model_type][model_name]['model_object'] = model

            elif model_type.lower() == 'gbm':
                logging.info("Creating GBM models")
                for model_name, model_dict in all_models_for_type.items():
                    # Create a GBM model
                    logging.info(f"Creating GBM model {model_name}")
                    if not self.load_previous_results: 
                        model = GeometricBrownianMotion(data=self.dataset_df,
                            model_hyperparameters=model_dict[DN.params], 
                            save_dir=path.join(self.output_directory, 'gbm', model_name), 
                            model_name=model_name,
                            x_vars=self.x_vars,
                            y_vars=self.y_vars,
                            seed=self._rand_state_mgr,
                            test_split_filter=model_dict['test_split_filter'],
                            train_split_filter=model_dict['train_split_filter'],
                            evaluation_filters=model_dict['evaluation_filters'], 
                            save_png=self.save_png,
                            save_html=self.save_html)
                        self._call_model_funcs(model)
                    else: 
                        model = GeometricBrownianMotion.load_from_previous_output(GeometricBrownianMotion, path.join(self.output_directory, 'lstm', model_name), model_name)
                    # Save the model off in the models_dict
                    self.models_dict[model_type][model_name]['model_object'] = model

            elif model_type.lower() == 'lstm_sde':
                
                logging.info("Creating LSTM SDE models")
                for model_name, model_dict in all_models_for_type.items():
                    # Create a LSTM SDE model
                    logging.info(f"Creating LSTM SDE model {model_name}")
                    model_args = {'data' : self.dataset_df, 
                            'model_hyperparameters' : model_dict[DN.params], 
                            'save_dir' : path.join(self.output_directory, 'lstm_sde', model_name), 
                            'model_name' : model_name,
                            'x_vars' : self.x_vars,
                            'y_vars' : self.y_vars,
                            'seed' : self._rand_state_mgr,
                            'test_split_filter' : model_dict['test_split_filter'],
                            'train_split_filter' : model_dict['train_split_filter'],
                            'evaluation_filters' : model_dict['evaluation_filters'], 
                            'save_png' : self.save_png,
                            'save_html' : self.save_html
                        }
                    
                    if not self.load_previous_results: 
                        model = LSTMSDE(**model_args) 
                        self._call_model_funcs(model)
                    else: 
                        model = LSTMSDE.load_from_previous_output(model_args)
                    # Save the model off in the models_dict
                    self.models_dict[model_type][model_name]['model_object'] = model

            else:   
                logging.error(f"Model {model_type} not implemented yet")
    
    def _call_model_funcs(self, model):
        """Helper function to call all model functions
        Args:
            model (_type_): _description_
        """        
        model.split_data()
        model.fit()
        model.save()
        model.plot()
        model.report()
    def _validate_models(self):
        # Validate the models dictionary. Make sure that the specified models are models 
        # that exist in pytorch, sklearn, or other libraries that we are using
        # If the models are not valid, raise an exception
        logging.debug("Validating models")

    def _set_models(self, models_dict):
        # set models for the analysis object
        # For all models in the models_dict, create a model object and store it in the models_dict

        # Assuming model_dict is a dictionary where the first level keys are the model types, and the second level keys are the model names

        for model_type, model_type_dict in models_dict.items():
            for model_name in model_type_dict: 
                models_dict[model_type][model_name]['model_object'] = None

        self.models_dict = models_dict



    def _get_models(self):
        # get models for the analysis object
        return self.models_dict

    def plot_dataset(self, plot_types):
        # Use the plotting class to plot the entirety of the dataset (all columns as options)
        for plot_type in plot_types:
            plot_all_x_y_combinations(self.dataset_df, 
                x_cols=None, 
                y_cols=None, 
                plot_type=plot_type, 
                output_dir=self.output_directory, 
                output_name='Stock Market Data', 
                save_png=self.save_png, 
                save_html=self.save_html)
            

    def load_from_previous_output(self):
        """Loads datasets and models from previous output

        Args:
        """        
        self.dataset_df = pd.read_csv(path.join(self.output_directory, DN.all_data + '.csv'))