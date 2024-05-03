from lstm_logger import logger as logging
from analysis import Analysis
from os import path, makedirs
# import plotting module 
from plotting import plot_all_x_y_combinations
# import tensorflow as tf
import re
import os
from lstm import LSTM 
from geometricbrownianmotion import GeometricBrownianMotion
from lstm_sde import LSTMSDE_to_train as LSTMSDE
from numpy.random import RandomState
import tensorflow as tf
import torch
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
        load_previous_results=False,
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
        self.load_pervious_results = load_previous_results
        self.output_dir = self._override_output_dir(output_dir, overwrite_out_dir=overwrite_out_dir)
        self._random_state_mgr = RandomState(master_seed)
        self.master_seed = master_seed
        tf.random.set_seed(master_seed)
        torch.manual_seed(master_seed)
        self.save_png = save_png
        self.save_html = save_html
    def _override_output_dir(self, output_dir, overwrite_out_dir=True, load_previous_results=False):
        outdir = path.join(output_dir, 'output_v0')
        if (path.exists(outdir) and overwrite_out_dir) or (path.exists(outdir) and load_previous_results):
            # If the output directory exists, roll the output directory to output_v{version_number}, 
            # where version number is the next available version number

            # Get the version number
            version_number = 0
            for f in os.listdir(output_dir):
                if re.match(r'output_v\d+', f):
                    version_number += 1
            # Create the new output directory and override outdir
            outdir = path.join(output_dir, f'output_v{version_number}')
        logging.debug(f"Creating output directory {outdir}")
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
                save_png=self.save_png,
                load_previous_results=self.load_pervious_results
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
        # Run cross-model analysis 
        #self.run_cross_model_analysis()
    def run_cross_model_analysis(self):
        # Run cross model analysis. this should ultimately 
        # 1.) create plots for the different model types to look at the differences between them
        # 2.) create summary statistics for the different model types to look at the differences between them, simmilar to the model.py report method

        raise NotImplementedError("This method is not implemented yet")
    
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