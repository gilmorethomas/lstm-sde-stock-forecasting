import os
from os import path, makedirs
import re
import copy 

from numpy.random import RandomState
import tensorflow as tf
import torch
import pandas as pd 
from lstm_logger import logger as logging
from analysis import Analysis
from project_globals import ModelStructure as MS
from plotting import plot, finalize_plot
import plotly.express as px
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
        self.output_dir = self._override_output_dir(output_dir, overwrite_out_dir=overwrite_out_dir, load_previous_results=load_previous_results)
        self._random_state_mgr = RandomState(master_seed)
        self.master_seed = master_seed
        tf.random.set_seed(master_seed)
        torch.manual_seed(master_seed)
        self.save_png = save_png
        self.save_html = save_html
    def _override_output_dir(self, output_dir, overwrite_out_dir=True, load_previous_results=False):
        outdir = path.join(output_dir, 'output_v0')

        if load_previous_results:
            version_number = -1
            for f in os.listdir(output_dir):
                if re.match(r'output_v\d+', f):
                    version_number += 1

        elif (path.exists(outdir) and overwrite_out_dir):
            # If the output directory exists, roll the output directory to output_v{version_number}, 
            # where version number is the next available version number

            # Get the version number
            version_number = 0
            for f in os.listdir(output_dir):
                if re.match(r'output_v\d+', f):
                    version_number += 1
        outdir = path.join(output_dir, f'output_v{version_number}')
            
            # Create the new output directory and override outdir
            
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

            if not path.exists(path.join(self.output_dir, analysis_name)):
                makedirs(path.join(self.output_dir, analysis_name))
            #logging.basicConfig(filename = path.join(self.output_dir, analysis_name, 'lstm_sde_stock_forecasting.log'), level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            #try:
            analysis_obj.run_analysis(run_descriptive, run_predictive)
            #except Exception as e:
            #    logging.error(f"Error occurred during analysis for {analysis_name}: {e}")
        # Run cross-model analysis 
        self.run_cross_model_analysis()
    def run_cross_model_analysis(self):
        # Run cross model analysis. this should ultimately 
        # 1.) create plots for the different model types to look at the differences between them
        # 2.) create summary statistics for the different model types to look at the differences between them, simmilar to the model.py report method
                # Assume that the first model objects types in data dict match all the others 
        first_aobj = list(self.analysis_objects_dict.keys())[0]
        first_model_type = list(self.analysis_objects_dict[first_aobj].models_dict.keys())[0]
        first_model_name = list(self.analysis_objects_dict[first_aobj].models_dict[first_model_type].keys())[0]
        
        first_model_obj = self.analysis_objects_dict[first_aobj].models_dict[first_model_type][first_model_name]['model_object']

        responses = first_model_obj.model_responses['proc']
        # For the plots for all models, only pull the first processed responses, since that will be a lot of traces otherwise
        eval_names = first_model_obj.evaluation_data_names
        
        # Create a dictionary to store the performance metrics for each model

        performance_struct = {k: None for k in first_model_obj.model_performance.keys()}
        performance_struct_avg = {k: None for k in first_model_obj.model_performance.keys()}
        x_vars = list(first_model_obj.x_vars)
        y_vars = list(first_model_obj.y_vars)
        all_vars = x_vars + y_vars + responses 
        for a_obj_name, a_obj in self.analysis_objects_dict.items():
            model_dict = a_obj.models_dict
            for k in performance_struct:
                # Create a dictionary to store the plots for each datatype
                if k == 'train' or k == 'train_data' or k == 'test' or k == 'test_data':
                    #import pdb; pdb.set_trace()
                    ...
                for model_type in model_dict.keys():
                    if model_type == 'LSTMSDE' and k == 'test':
                        ...
                        #import pdb; pdb.set_trace()
                    for modelkey, modelvals in model_dict[model_type].items():
                        this_data = modelvals['model_object'].model_performance[k].copy(deep=True)
                        this_data = this_data[responses] 
                        this_data['stock'] = a_obj_name
                        # Make a multiindex dataframe where the first index is the modelname and the second is the existing index 
                        this_data['model_type'] = modelvals['model_object'].__class__.__name__  
                        this_data['model_name'] = modelkey
                        #this_data_avgd_across_stocks = this_data.copy(deep=True) 
                        #this_data_avgd_across_stocks = this_data_avgd_across_stocks.drop(columns=['stock'])
                        #this_data_avgd_across_stocks = this_data_avgd_across_stocks.reset_index().pivot(index=['model_type', 'model_name'], columns='index', values=responses)

                        #this_data_avgd_across_stocks = this_data.groupby(['model_type', 'model_name'])[responses].mean()
                        this_data2 = this_data.reset_index().pivot(index=['model_type', 'model_name', 'stock'], columns='index', values=responses)
                        #this_data_avgd_across_stocks = this_data_avgd_across_stocks.reset_index().pivot(index=['model_type', 'model_name'], columns='index', values=responses)
                        if performance_struct[k] is None:
                            performance_struct[k] = this_data2
                            #performance_struct_avg[k] = this_data_avgd_across_stocks
                        else: 
                            performance_struct[k] = pd.concat([performance_struct[k], this_data2])
                            #performance_struct_avg[k] = pd.concat([performance_struct_avg[k], this_data_avgd_across_stocks])
        performance_struct_avg = copy.deepcopy(performance_struct)
        # Iterate over the performance struct and create the averaged data 
        if not path.exists(path.join(self.output_dir, 'all_stocks', MS.perf)):
            makedirs(path.join(self.output_dir, 'all_stocks', MS.perf))
        for k, df in performance_struct_avg.items():
            if df is None: 
                continue
            df = performance_struct_avg[k].copy(deep=True)
            #import pdb; pdb.set_trace()
            ## Rename the values in the model name column to remove 'gbm', 'lstm', and 'lstm_sde' from the name
            
            #df.to_csv(path.join(self.output_dir, 'all_stocks', MS.perf, f'{k}_avg.csv'), float_format='%.2e')
            # Make a bar plot for each of the performance metrics

            df = df.reset_index()
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
            df = df.rename(columns={'model_name_': 'model_name', 'model_type_': 'model_type', 'stock_': 'stock'})
            #df['model_name'] = df['model_name'].str.replace('gbm_', '')
            #df['model_name'] = df['model_name'].str.replace('lstm_sde_', '')
            #df['model_name'] = df['model_name'].str.replace('lstm_', '')
            #df['model_name'] = df['model_name'].str.replace('_', ' ')
            df_to_avg = df.copy(deep=True)
            df_to_avg = df_to_avg.drop(columns=['stock'])
            # Coerce the values to floats
            for col in df_to_avg.columns:
                if 'Close' in col:
                    df_to_avg[col] = df_to_avg[col].astype(float)
            # groupby model type and model name and average the values, converting values to floats
            df2 = df_to_avg.groupby(['model_type', 'model_name']).mean()
            df2 = df2.reset_index()
            figMAE = plot(df2, x_col='model_name', 
                y_col='Close_mean_MAE', 
                trace_name='Model',
                plot_type='bar',
                groupbyCols=['model_type'])
            
                 #title=f'{k} averaged across stocks')
            figR2 = plot(df2, x_col='model_name', 
                y_col='Close_mean_R2', 
                trace_name='Model',
                plot_type='bar',
                groupbyCols=['model_type'])

            
                 #title=f'{k} averaged across stocks')      
            figRMSE = plot(df2, x_col='model_name', 
                y_col='Close_mean_RMSE', 
                trace_name='Model',
                plot_type='bar',
                groupbyCols=['model_type'])

            
                 #title=f'{k} averaged across stocks')
            finalize_plot(figMAE, 
                          title=f'{k} MAE averaged across stocks', 
                          filename = f'{k}_avg_MAE',
                          output_dir=path.join(self.output_dir, 'all_stocks', MS.perf),
                          save_html=self.save_html,
                          save_png=self.save_png,
                          plot_type='bar',
                          xaxis_title='Model',
                          yaxis_title='Mean Absolute Error')
                          #margin=dict(l=100, r=100, t=100, b=100),
                          #tickangle=45)

            finalize_plot(figR2,  
                          title=f'{k} R2 averaged across stocks', 
                          filename = f'{k}_avg_R2',
                          output_dir=path.join(self.output_dir, 'all_stocks', MS.perf),
                          save_html=self.save_html,
                          save_png=self.save_png,
                          plot_type='bar',
                          xaxis_title='Model',
                          yaxis_title='R2')
            finalize_plot(figRMSE,
                          title=f'{k} RMSE averaged across stocks', 
                          filename = f'{k}_avg_RMSE',
                          output_dir=path.join(self.output_dir, 'all_stocks', MS.perf),
                          save_html=self.save_html,
                          save_png=self.save_png, 
                          plot_type='bar',
                          xaxis_title='Model',
                          yaxis_title='Root Mean Squared Error')
            output_file = path.join(self.output_dir, 'all_stocks', MS.perf, f'{k}_mean.csv')
            df2.to_csv(output_file, float_format='%.2e')

        # Create the plots for each datatype

        # Finalize the plot for each plot in figs 
        for k, df in performance_struct.items():
            if df is None: 
                logging.warning(f'No performance for {k}')
                continue
            output_file = path.join(self.output_dir, 'all_stocks', MS.perf, f'{k}.csv')
            logging.info(f"Writing performance metrics for {k} to {output_file}")
            # Reindex the csv so the model name is the first column, and R2, MAE, and RMSE are columns 
            df.to_csv(output_file, float_format='%.2e')
    
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