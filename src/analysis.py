import logging 
from os import path, makedirs
# import plotting module 
from plotting import plot_all_x_y_combinations
# import tensorflow as tf
import re
import os
from lstm import LSTM 
from geometricbrownianmotion import GeometricBrownianMotion
from numpy.random import RandomState

class AnalysisManager(): 
    # Create an analysis manager whose job is to manage the analysis objects
    # It should store a list of analysis objects and call the methods of the analysis objects
    # It should also take a raw directory and an output directory
    # Raw directory is where the raw data is stored
    # Output directory is where the output data is stored
    def __init__(self, 
        raw_dir, 
        output_dir,
        master_seed = 0, 
        save_png = True, 
        save_html=True, 
        overwrite_out_dir=True,
        **kwargs): 
        """
        Creates the analysis maanger 
        Args:
            raw_dir (string, path-like): _description_
            output_dir (string, path-like)): _description_
        """
        self.analysis_objects_dict = {}
        self.raw_dir = raw_dir
        # Create a directory called at output_dir/output 
        self.output_dir = self._override_output_dir(output_dir, overwrite_out_dir=overwrite_out_dir)
        self._random_state_mgr = RandomState(master_seed)
        self.master_seed = master_seed
        self.save_png = save_png
        self.save_html = save_html
    def _override_output_dir(self, output_dir, overwrite_out_dir=True):
        outdir = path.join(output_dir, 'output_v0')
        if path.exists(outdir) and overwrite_out_dir:
            # If the output directory exists, roll the output directory to output_v{version_number}, 
            # where version number is the next available version number

            # Get the version number
            version_number = 0
            for f in os.listdir(output_dir):
                if re.match(r'output_v\d+', f):
                    version_number += 1
            # Create the new output directory and override outdir
            outdir = path.join(output_dir, f'output_v{version_number}')
        logging.info(f"Creating output directory {outdir}")
        if not path.exists(outdir):
            makedirs(outdir)
        return outdir

    def set_preprocessing_callback(self, preprocessing_callback):
        """ 
        Sets a preprocessing callback to preprocess data 

        Args:
            preprocessing_callback (function handle): function handle for preprocessing
        """        
        self.preprocessing_callback = preprocessing_callback
        # set the preprocessing callback for the analysis objects

    
    def add_analysis_objs(self, analysis_dict, x_vars, y_vars): 
        # add a dictionary of analysis objects, with key being the name of the analysis object
        # and the value being the dataset
        for dataset_name, dataset_df in analysis_dict.items(): 
            logging.info(f"Creating analysis object for {dataset_name}")
            analysis = Analysis(dataset_name, 
                dataset_df, 
                x_vars, 
                y_vars, 
                path.join(self.output_dir, dataset_name), 
                preprocessing_callback=self.preprocessing_callback,
                seed=self._random_state_mgr,
                save_html=self.save_html,
                save_png=self.save_png
            )
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
    def __init__(self, dataset_name, dataset_df, x_vars, y_vars, output_directory, seed, preprocessing_callback=None, save_html=True, save_png=True):
         
        self.dataset_name = dataset_name
        self._raw_dataset_df = dataset_df
        self.preprocessing_callback = preprocessing_callback
        self.output_directory = output_directory
        self.x_vars = x_vars
        self.y_vars = y_vars
        self.save_html = save_html
        self.save_png = save_png
        self._rand_state_mgr = seed
        if not(path.exists(output_directory)):
            logging.info(f"Creating output directory {output_directory}")
            makedirs(output_directory)
        
    def preprocess_dataset(self):
        # preprocess the dataset, using the preprocessing_callback if provided
        logging.info(f"Preprocessing dataset for {self.dataset_name} using {self.preprocessing_callback}")
        if self.preprocessing_callback is not None: 
            self.dataset_df = self.preprocessing_callback(self._raw_dataset_df)
        # Get rid of columns that are not x_vars or y_vars and say which columns we are removing 
        logging.info(f"Removing columns {self.dataset_df.columns.difference(self.x_vars + self.y_vars)}")
        self.dataset_df = self.dataset_df[self.x_vars + self.y_vars]
        logging.info(f"Columns remaining {self.dataset_df.columns}")
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
    def _import_tensorflow(self):
        """Imports tensorflow. Used as a utility function to avoid import time bogdown if not using tf
        """        
        import tensorflow as tf
        tf.random.set_seed(master_seed)


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
                self._import_tensorflow()
                logging.info("Creating LSTM models")
                for model_name, model_dict in all_models_for_type.items():
                    logging.info(f"Creating LSTM model {model_name}")
                    model = LSTM(data=self.dataset_df, 
                        model_hyperparameters=model_dict['library_hyperparameters'],
                        units = model_dict['units'],
                        save_dir=path.join(self.output_directory, model_name), 
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
            elif model_type.lower() == 'gbm':
                logging.info("Creating GBM models")
                for model_name, model_dict in all_models_for_type.items():
                    # Create a GBM model
                    logging.info(f"Creating GBM model {model_name}")
                    model = GeometricBrownianMotion(data=self.dataset_df,
                        model_hyperparameters=model_dict['model_hyperparameters'], 
                        save_dir=path.join(self.output_directory, model_name), 
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
            elif model_type.lower() == 'lstm_sde':
                logging.info("Creating LSTM SDE models")
                for model_name, model_hyperparameters in all_models_for_type.items():
                    # Create a LSTM SDE model
                    logging.info(f"Creating LSTM SDE model {model_name}")
                    raise NotImplementedError("LSTM SDE model not implemented yet")
            else:   
                logging.error(f"Model {model_type} not implemented yet")
    
    def _call_model_funcs(self, model):
        """Helper function to call all model functions
        Args:
            model (_type_): _description_
        """        
        model.split_data()
        model.train()
        model.test()
        model.predict()
        model.save()
        model.plot()
        model.report()
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
            plot_all_x_y_combinations(self.dataset_df, 
                x_cols=None, 
                y_cols=None, 
                plot_type=plot_type, 
                output_dir=self.output_dir, 
                output_name='Stock Market Data', 
                save_png=self.save_png, 
                save_html=self.save_html)