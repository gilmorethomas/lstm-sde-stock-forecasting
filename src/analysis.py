import os
from os import path, makedirs
import re
import copy 

import pandas as pd
from numpy.random import RandomState
import tensorflow as tf
import torch

from lstm_logger import logger as logging
from project_globals import DataNames as DN
from project_globals import ModelTypes as MT
from project_globals import ModelStructure as MS

from plotting import plot_all_x_y_combinations, plot_multiple_y_cols, finalize_plot
from model import Model
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
            separator = "*" * 80
            logging.warning(separator)
            logging.warning("Loading models and datasets from previous results. If you have changed the hyperparameters or the dataset, you should set load_previous_results to False")
            logging.warning(separator)
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


    def run_analysis(self, run_descriptive=True, run_predictive=True, run_cross_model=True): 

        logging.info(f"Running analysis for {self.dataset_name}")
        if run_descriptive: 
            self.run_descriptive()
        if run_predictive: 
            self.run_predictive()
        if run_cross_model: 
            self.run_cross_model()
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
                    model_args = {'data' : self.dataset_df,
                            'model_hyperparameters' : model_dict['library_hyperparameters'],
                            'save_dir' : path.join(self.output_directory, 'LSTM', model_name),
                            'units' :  model_dict['units'],
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
                        model = LSTM(**model_args)
                        self._call_model_funcs(model)
                    else: 
                        model = LSTM.load_from_previous_output(model_args)
                    # Save the model off in the models_dict
                    self.models_dict[model_type][model_name]['model_object'] = model

            elif model_type.lower() == 'gbm':
                logging.info("Creating GBM models")
                for model_name, model_dict in all_models_for_type.items():
                    # Create a GBM model
                    logging.info(f"Creating GBM model {model_name}")
                    model_args = {'data' : self.dataset_df, 
                            'model_hyperparameters' : model_dict[DN.params], 
                            'save_dir' : path.join(self.output_directory, 'LSTMSDE', model_name), 
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
                        model = GeometricBrownianMotion(**model_args) 
                        self._call_model_funcs(model)
                    else: 
                        model = GeometricBrownianMotion.load_from_previous_output(model_args)
                    # Save the model off in the models_dict
                    self.models_dict[model_type][model_name]['model_object'] = model

            elif model_type.lower() == 'lstmsde':
                
                logging.info("Creating LSTM SDE models")
                for model_name, model_dict in all_models_for_type.items():
                    # Create a LSTM SDE model
                    logging.info(f"Creating LSTM SDE model {model_name}")
                    model_args = {'data' : self.dataset_df, 
                            'model_hyperparameters' : model_dict[DN.params], 
                            'save_dir' : path.join(self.output_directory, 'LSTMSDE', model_name), 
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
    def run_cross_model(self):
        # Create the cross-model directory 
        cross_model_dir = path.join(self.output_directory, 'cross_model')
        # Make all the submodel types. Pull only the types that we've defined from the modeule. This ignores special variables like __name__ and __doc__
        model_types = [v for k, v in vars(MT).items() if isinstance(v, str) and not v.startswith('__') and not callable(v) and v is not None and not v.startswith('<') and not k.startswith('__')]
        
        # Create the cross model directory
        #model_dict = Model._get_empty_datadict_struct()
        #models_dict = {model_type: copy.deepcopy(model_dict) for model_type in model_types}


        model_type_dirs = [path.join(cross_model_dir, mt) for mt in model_types + ['all']]
        [makedirs(mt) for mt in model_type_dirs if not path.exists(mt)]
        # For each model type, combine the results of the models in that type and write a report
        for model_type_dir, model_type in zip(model_type_dirs, model_types):
            # Get the models for the model type
            model_type_models = self.models_dict.get(model_type, {})
            if model_type_models is None or len(model_type_models) == 0:
                logging.info(f"No models for model type {model_type}")
                continue
            # Get the model objects for the model type
            model_name_with_model_obj  = {model_name: model_dict['model_object'] for model_name, model_dict in model_type_models.items()}
            #model_name_with_hyperparams  = {model_name: model_dict['model_hyperparameters'] for model_name, model_dict in model_type_models.items()}
                                       
            if model_name_with_model_obj is None or len(model_name_with_model_obj) == 0:
                logging.info(f"No model objects for model type {model_type}")
                continue
            # Combine the results of the models in that type
            # For now, just plot the results
            self._plot_cross_model_results(model_name_with_model_obj, model_type_dir, model_type)
            # Write a report
            self._write_cross_model_report(model_name_with_model_obj, model_type_dir, model_type)
        # Combine the results of all models and write a report
        all_models = {}
        for model_type, model_type_dict in self.models_dict.items():
            all_models.update((model_name, model_dict['model_object']) for model_name, model_dict in model_type_dict.items())
        self._plot_cross_model_results(all_models, path.join(cross_model_dir, 'all'), 'all')
        self._write_cross_model_report(all_models, path.join(cross_model_dir, 'all'), 'all')
    
    def _plot_cross_model_results(self, model_dict, output_dir, model_type):
        """Plots the results multiple models, which are contained in the models dictionary
        Creates one plot for each datatype in the data dictionary.
        TODO (break this out for multiple y vars and multiple x vars)

        Args:
            model_dict (_type_): _description_
            output_dir (_type_): _description_
            model_type (_type_): _description_
        """        
        # Assume that the first model objects types in data dict match all the others 
        first_model = list(model_dict.keys())[0]
        # For the plots for all models, only pull the first processed responses, since that will be a lot of traces otherwise
        proc_responses = list(model_dict[first_model].model_responses['proc']) if model_type != 'all' else [model_dict[first_model].model_responses['proc'][0]]
        eval_names = list(model_dict[first_model].evaluation_data_names)
        x_vars = list(model_dict[first_model].x_vars)
        y_vars = list(model_dict[first_model].y_vars)
        all_vars = x_vars + y_vars + proc_responses 
        data_dict_struct = model_dict[first_model].get_empty_datadict_struct()

        data_keys = list(data_dict_struct[DN.not_normalized].keys())
        figs = {dk : None for dk in data_keys} 

        for k in data_dict_struct[DN.not_normalized]:
            # Create a dictionary to store the plots for each datatype
            if k == 'all_data':
                continue
            for modelkey, modelvals in model_dict.items():
                this_data = modelvals.data_dict[DN.not_normalized][k]
                if this_data is None or this_data.empty:
                    logging.warning(f"No data for {k} in model {modelkey}")
                    continue
                if k not in figs:
                    figs[k] = None

                # Only plot the actual response onse
                y_cols_to_plot = y_vars + proc_responses if modelkey == first_model else proc_responses
                # First pass figs[k] is going to be null, but plot can handle that
                figs[k] = plot_multiple_y_cols(
                    df=this_data, 
                    x_col=x_vars[0], 
                    y_cols=y_cols_to_plot,
                    trace_name_to_prepend=modelkey, 
                    plot_type='line', 
                    title=None,
                    fig=figs[k],
                    output_dir=None, 
                    output_name=None, 
                    save_png=False, 
                    save_html=False,
                    finalize=False)
                
        # Finalize the plot for each plot in figs 
        for k, fig in figs.items():
            if fig is None: 
                logging.warning(f'No figure for {k}')
                continue
            finalize_plot(fig,
                          title = f'{k} for all {model_type} models',
                          output_dir=output_dir,
                          filename=k,
                          save_png=self.save_png,
                          save_html=self.save_html)


            # Now plot the data 




    def _write_cross_model_report(self, model_dict, output_dir, model_type):
        # Assume that the first model objects types in data dict match all the others 
        first_model = list(model_dict.keys())[0]
        # For the plots for all models, only pull the first processed responses, since that will be a lot of traces otherwise
        if model_type == 'all':
            first_model_type = first_model
            first_model_obj = model_dict[first_model]
            proc_responses = first_model_obj.model_responses['proc'][0]
            if not isinstance(proc_responses, list):
                proc_responses = [proc_responses]
            #eval_names = list(model_dict[first_model_type][first_model].evaluation_data_names)
            eval_names = first_model_obj.evaluation_data_names

        else: 
            first_model_obj = model_dict[first_model]
            proc_responses = first_model_obj.model_responses['proc'] 
            if not isinstance(proc_responses, list):
                proc_responses = [proc_responses]
            eval_names = model_dict[first_model].evaluation_data_names
        
        # Create a dictionary to store the performance metrics for each model

        performance_struct = {k: None for k in first_model_obj.model_performance.keys()}
        x_vars = list(first_model_obj.x_vars)
        y_vars = list(first_model_obj.y_vars)
        all_vars = x_vars + y_vars + proc_responses 

        for k in performance_struct:
            # Create a dictionary to store the plots for each datatype
            if k == 'all_data':
                continue
            for modelkey, modelvals in model_dict.items():
                this_data = modelvals.model_performance[k]
                # Make a multiindex dataframe where the first index is the modelname and the second is the existing index 
                this_data['model_name'] = modelkey

                this_data.set_index(['model_name'], append=True, inplace=True)
                if performance_struct[k] is None:
                    performance_struct[k] = this_data
                else: 
                    performance_struct[k] = pd.concat([performance_struct[k], this_data])

        # Finalize the plot for each plot in figs 
        for k, df in performance_struct.items():
            if df is None: 
                logging.warning(f'No performance for {k}')
                continue
            output_file = path.join(output_dir, MS.perf, f'{k}.csv')
            if not path.exists(path.join(output_dir, MS.perf)):
                makedirs(path.join(output_dir, MS.perf))
            logging.info(f"Writing performance metrics for {k} to {output_file}")
            df.to_csv(output_file)

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