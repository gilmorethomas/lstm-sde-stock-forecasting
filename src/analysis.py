import logging 
from os import path, makedirs
# import plotting module 
from plotting import Plotting
import sklearn
import torch
# import tensorflow as tf
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

class AnalysisManager(): 
    # Create an analysis manager whose job is to manage the analysis objects
    # It should store a list of analysis objects and call the methods of the analysis objects
    # It should also take a raw directory and an output directory
    # Raw directory is where the raw data is stored
    # Output directory is where the output data is stored


    def __init__(self, raw_dir, output_dir): 
        """
        Creates the analysis maanger 
        Args:
            raw_dir (string, path-like): _description_
            output_dir (string, path-like)): _description_
        """
        self.analysis_objects_dict = {}
        self.raw_dir = raw_dir
        self.output_dir = output_dir

    def set_preprocessing_callback(self, preprocessing_callback):
        """ 
        Sets a preprocessing callback to preprocess data 

        Args:
            preprocessing_callback (function handle): function handle for preprocessing
        """        
        self.preprocessing_callback = preprocessing_callback
        # set the preprocessing callback for the analysis objects

    
    def add_analysis_objs(self, analysis_dict): 
        # add a dictionary of analysis objects, with key being the name of the analysis object
        # and the value being the dataset
        for dataset_name, dataset_df in analysis_dict.items(): 
            logging.info(f"Creating analysis object for {dataset_name}")
            analysis = Analysis(dataset_name, dataset_df, path.join(self.output_dir, dataset_name), preprocessing_callback=self.preprocessing_callback)
            self.analysis_objects_dict[dataset_name] = analysis
    
    def preprocess_datasets(self):
        """
        Preprocesses datasets for each analysis object
        """        
        # preprocess the datasets for each analysis object
        logging.info("Preprocessing datasets")
        for analysis_name, analysis_obj in self.analysis_objects_dict.items(): 
            analysis_obj.preprocess_dataset()
    
    def get_analysis_objs(self):
        """Gets the analysis objects belonging to the analysis manager

        Returns:
            dict : analysis objects, with key being the name of the analysis object and the value being the analaysis object 
        """
        return self.analysis_objects_dict
        
    def run_analysis(self, run_descriptive=True, run_predictive=True):
        """Runs analysis 

        Args:
            run_descriptive (bool, optional): Whether or not to run descriptive anlaysis. Defaults to True.
            run_predictive (bool, optional): Whether or not to run predictive analysis. Defaults to True.
        """        
        # run analysis on each object
        for analysis_name, analysis_obj in self.analysis_objects_dict.items(): 
            logging.info(f"Running analysis for {analysis_name}")
            analysis_obj.run_analysis(run_descriptive, run_predictive)
    def set_models_for_analysis_objs(self, models_dict, analysis_objs_to_use=None): 
        # set models for each analysis object
        # if analysis_objs_to_use is None, set models for all analysis objects,
        # otherwise, set models for the analysis objects in the list
        if analysis_objs_to_use is None: 
            analysis_objs_to_use = self.analysis_objects_dict
        for a_obj in analysis_objs_to_use.values():
            a_obj._set_models(models_dict)


    def _run_all_in_one_plot(self):
        # run an all-in-one plot with all the datasets
        raise NotImplementedError("This method is not implemented yet")
            
class Analysis(): 
    def __init__(self, dataset_name, dataset_df, output_directory, preprocessing_callback=None):
         
        self.dataset_name = dataset_name
        self._raw_dataset_df = dataset_df
        self.preprocessing_callback = preprocessing_callback
        self.output_directory = output_directory
        if not(path.exists(output_directory)):
            logging.info(f"Creating output directory {output_directory}")
            makedirs(output_directory)
    def preprocess_dataset(self):
        # preprocess the dataset, using the preprocessing_callback if provided
        logging.info(f"Preprocessing dataset for {self.dataset_name} using {self.preprocessing_callback}")
        if self.preprocessing_callback is not None: 
            self.dataset_df = self.preprocessing_callback(self._raw_dataset_df)
            

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
        ...        

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
            
            if model_type.lower() == 'lstm':
                logging.info("Creating LSTM models")
                for model_name, model_hyperparameters in all_models_for_type.items():
                    logging.info(f"Creating LSTM model {model_name}")
                    model = LSTM(model_hyperparameters, self.output_directory, model_name)
                    model.split_data()
                    model.train()
                    model.test()
                    model.predict()
                    model.save()
                    model.plot()
                    model.report()
            else:   
                logging.error(f"Model {model_type} not implemented yet")
        raise NotImplementedError("This method is not implemented yet")
    def _validate_models(self):
        # Validate the models dictionary. Make sure that the specified models are models 
        # that exist in pytorch, sklearn, or other libraries that we are using
        # If the models are not valid, raise an exception
        logging.info("Validating models")

    def _set_models(self, models_dict):
        # set models for the analysis object
        self.models_dict = models_dict

    def _get_models(self):
        # get models for the analysis object
        return self.models_dict

    def plot_dataset(self, plot_types):
        # Use the plotting class to plot the entirety of the dataset (all columns as options)
        for plot_type in plot_types:
            plotter = Plotting(self.dataset_df, plot_type)
            plotter.plot()

class Model(): 
    # Define a model class that takes a test train split and model hyperparameters
    # The model class should have a train method that trains the model
    # The model class should have a test method that tests the model
    # The model class should have a predict method that predicts the model
    # The model class should have a save method that saves the model
    # The model class should have a load method that loads the model
    # The model class should have a plot method that plots the model
    # The model class should have a report method that reports the model's performance

    def __init__(self, model_hyperparameters, save_dir, model_name):
        self.model_hyperparameters = model_hyperparameters
        self.save_dir = save_dir
        self.model_name = model_name
        self.model = None
    
    def split_data(self):
        # Split the data into test and train using sklearn.model_selection.train_test_split
        raise NotImplementedError("This should be implemented by the child class")
    def train(self):
        # train the model
        raise NotImplementedError("This should be implemented by the child class")
    
    def test(self):
        # test the model
        raise NotImplementedError("This should be implemented by the child class")
    
    def predict(self):
        # predict the model
        raise NotImplementedError("This should be implemented by the child class")
    
    def save(self):
        # Save the model to a file using .pkl serialization or some other method, which is dependent on the library
        raise NotImplementedError("This should be implemented by the child class")
        if self.model is None:
            logging.error("No model exists yet")
        with open(path.join(self.save_dir, self.model_name + '.pkl'), 'wb') as file:  
            logging.info(f"Saving model to {path.join(self.save_dir, self.model_name + '.pkl')}")
            pickle.dump(model, file)

    def load(self):
        # Load the model from pkl if it exists
        if path.exists(path.join(self.save_dir, self.model_name + '.pkl')):
            with open(path.join(self.save_dir, self.model_name + '.pkl'), 'rb') as file:  
                logging.info(f"Loading model from {path.join(self.save_dir, self.model_name + '.pkl')}")
                self.model = pickle.load(file)
        else:
            logging.status(f"No model exists at {path.join(self.save_dir, self.model_name + '.pkl')}")
    
    def plot(self):
        # Plot the model
        raise NotImplementedError("This should be implemented by the child class")
    
    def report(self):
        # Report the model's performance
        raise NotImplementedError("This should be implemented by the child class")
    
    def _validate_hyperparameters(self):
        # Validate the hyperparameters of the model
        raise NotImplementedError("This should be implemented by the child class")
    

class LSTM(Model):
    # Define an LSTM class that inherits from the model class, implemented using pytorch as similar to this link: 
    # https://www.datacamp.com/tutorial/lstm-python-stock-market

    def __init__(self, model_hyperparameters, save_dir, model_name):
        logging.info("Creating lstm model")
        super().__init__(model_hyperparameters, save_dir, model_name)



        # self.model = LSTM(**model_hyperparameters)

# tf.keras.layers.LSTM(
#     units,
#     activation='tanh',
#     recurrent_activation='sigmoid',
#     use_bias=True,
#     kernel_initializer='glorot_uniform',
#     recurrent_initializer='orthogonal',
#     bias_initializer='zeros',
#     unit_forget_bias=True,
#     kernel_regularizer=None,
#     recurrent_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     recurrent_constraint=None,
#     bias_constraint=None,
#     dropout=0.0,
#     recurrent_dropout=0.0,
#     seed=None,
#     return_sequences=False,
#     return_state=False,
#     go_backwards=False,
#     stateful=False,
#     unroll=False,
#     use_cudnn='auto',
#     **kwargs
# )
    def train(self):
        # Train the model using the test train split
        logging.info("Need to implement")
    