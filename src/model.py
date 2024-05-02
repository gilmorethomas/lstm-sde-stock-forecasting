from lstm_logger import logger as logging
from plotting import plot_all_x_y_combinations, plot_multiple_dfs
from project_globals import DataNames as DN

from os import path, makedirs
import pickle
import pandas as pd 
import numpy as np
import copy
from scaler import Scaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Model(): 
    # Define a model class that takes a test train split and model hyperparameters
    # The model class should have a train method that trains the model
    # The model class should have a test method that tests the model
    # The model class should have a predict method that predicts the model
    # The model class should have a save method that saves the model
    # The model class should have a load method that loads the model
    # The model class should have a plot method that plots the model
    # The model class should have a report method that reports the model's performance

    def __init__(self, 
        data, 
        model_hyperparameters, 
        save_dir, 
        model_name,
        x_vars:list,
        y_vars:list,
        seed=np.random.RandomState,
        test_split_filter=None, 
        train_split_filter=None, 
        evaluation_filters:dict={},
        save_html=True,
        save_png=True
    ):
        self.data = data
        self.model_hyperparameters = model_hyperparameters
        self.test_split_filter = test_split_filter
        self.train_split_filter = train_split_filter
        self.evaluation_filters = evaluation_filters
        self.evaluation_data_names = list(evaluation_filters.keys())
        self.save_dir = save_dir
        if not path.exists(save_dir):
            logging.info(f"Creating save directory {save_dir}")
            makedirs(save_dir)
        self.model_name = model_name
        self.model_objs = []  # List of model objects
        self.train_params = {}
        self.x_vars = x_vars
        self.y_vars = y_vars
        self.model_responses = {DN.raw: [],
                                DN.proc: [],}
        self._rand_state_mgr = seed
        self.save_html = save_html
        self.save_png = save_png
        self.seed = seed
        logging.debug('Using the first y var as the base column for scaling')
        # Assign all the dataframes that are expected to be filled later 
        self.model_performance = {}

        # The below is if you want to base the scalings off of some other dataframe or column. This is kinda weird though...
        # Have the scalings operated independently (that is, each column is scaled independently of the others and independently of the other dataframes)
        self.scaler = MinMaxScaler()
        self.scaler.fit(data[self.y_vars])

        self.data_scaled = self.data.copy(deep=True)
        self.data_scaled[self.y_vars] = self.scaler.transform(data[self.y_vars])
        self._build_model_response_dict()
        # Save scaled and unscaled data
        # Add the data to the scaler
        
        # If you want to scale the data by default by default the data is scaled

    def _unpack_model_params(self, model_params):
        """Assign model parameters to the class 

        Args:
            model_params (dict): 
        """        
        # Assigns the model parameters to the class
        for k, v in model_params.items():
            setattr(self, k, v)
        
    def fit(self, data_dict):
        """Helper method that overrides self.data with the data dict

        Args:
            data_dict (dict): dictionary of dataframes
        """        
        # TODO (check that all the expected things are in the data dict for validation check). 
        # This is because each model may train differently, using different data (scaled or unscaled)
        self.data_dict = data_dict
        # train the model
        assert self.data_dict[DN.not_normalized][DN.train_data] is not None, "Train data fit cannot be None, subclass must provide fit train data to the Model train method"

        # Add rollup data for all dataframes    

        self.model_responses[DN.proc], self.data_dict[DN.not_normalized][DN.train_data] = self._add_rollup_data(data_dict[DN.not_normalized][DN.train_data])
        _, self.data_dict[DN.normalized][DN.train_data] = self._add_rollup_data(data_dict[DN.normalized][DN.train_data])
        _, self.data_dict[DN.not_normalized][DN.train_data] = self._add_rollup_data(data_dict[DN.not_normalized][DN.train_data])
        _, self.data_dict[DN.normalized][DN.train_data] = self._add_rollup_data(data_dict[DN.normalized][DN.train_data])
        for eval_name in self.evaluation_data_names:
            _, self.data_dict[DN.not_normalized][eval_name] = self._add_rollup_data(self.data_dict[DN.not_normalized][eval_name])
            _, self.data_dict[DN.normalized][eval_name] = self._add_rollup_data(self.data_dict[DN.normalized][eval_name])

        self._plot_fit(DN.train_data)
        self._plot_fit(DN.train_data)
        [self._plot_fit(eval_name) for eval_name in self.evaluation_data_names]
        #self._plot_fit('evaluation_data')
        self._print_train_end_message()
    def _build_model_response_dict(self):
        # Build model response raw list from y_vars with model name and number of sims appended
        self.model_responses[DN.raw] = [f'{y_var}_{self.model_name}_{i}' for y_var in self.y_vars for i in range(self.model_hyperparameters['num_sims'])]
    def _normalize_data_dict(self, data_dict, y_col):
        # Returns a normalized data dictionary, where data with columns in y_col are scaled
        # Unscale the data
        all_resps = [y_col] + [resp for resp in self.model_responses[DN.raw] if y_col in resp]
        # Now unscale the data
        for k, v in data_dict.items():
            # check if v is a dataframe 
            logging.error("need to fix the implementation of scaling")
            if isinstance(v, pd.DataFrame):
                data_dict[k][all_resps] = self.scaler.inverse_transform(data_dict[k][all_resps])
            # nested dictionary
            else:
                for k1, v1 in v.items():
                    data_dict[k][k1][all_resps] = self.scaler.inverse_transform(data_dict[k][k1][all_resps])
        return data_dict

    def _unnormalize_data_dict(self, data_dict):
        # Returns a normalized data dictionary, where data with columns in y_col are scaled
        for k, v in data_dict.items():
            # check if v is a dataframe 
            if k == DN.all_data :
                continue
            if isinstance(v, pd.DataFrame):
                cols = v.select_dtypes(include=['float64', 'float32']).columns
                data_dict[k][cols] = self.scaler.inverse_transform(data_dict[k][cols], df_name=k)
            # nested dictionary
            else:
                for k1, v1 in v.items():
                    cols = v1.select_dtypes(include=['float64', 'float32']).columns
                    data_dict[k][k1][cols] = self.scaler.inverse_transform(data_dict[k][k1][cols], df_name=k1)
        return data_dict
    
    def split_data(self):
        # Perform test, train, evaluation split using the filters in model hyperparameters. 
        # Set self.train_data equal to the data with the train split filter applied
        test_data = self.data[self.test_split_filter]
        # Scale the data and then apply the test split filter 
        test_data_scaled = self.data_scaled[self.test_split_filter]
        train_data = self.data[self.train_split_filter]
        train_data_scaled = self.data_scaled[self.train_split_filter]
        evaluation_data = {k: self.data[v] for k, v in self.evaluation_filters.items()}
        evaluation_data_scaled = {k: self.data_scaled[v] for k, v in self.evaluation_filters.items()}
        self.data_dict = {
            DN.not_normalized : 
            {
                DN.all_data : self.data, 
                DN.train_data: train_data, 
                DN.test_data: test_data, 
                #'evaluation_data' : evaluation_data
            },
            DN.normalized : 
            {
                DN.all_data : self.data_scaled, 
                DN.train_data: train_data_scaled, 
                DN.test_data: test_data_scaled, 
                #'evaluation_data' : evaluation_data_scaled
            }
        }
        self.data_dict[DN.not_normalized].update(evaluation_data)
        self.data_dict[DN.normalized].update(evaluation_data_scaled)
        # Get rid of the filters, as lambda functions are not serializable
        self.train_split_filter = None
        self.test_split_filter = None
        self.evaluation_filters = {k : None for k in self.evaluation_filters.keys()}
        # Add the evaluation data to the scaler
        self._plot_test_train_split()
    def _plot_test_train_split(self):
        """Plots the test train split data
        """        
        # Build a temporary dictionary of all the data splits, including train, test, and eval
        # This will be used to plot the data

        plot_multiple_dfs(self.data_dict[DN.not_normalized], 
            title='Train, Test, and Evaluation Data Split',
            x_cols=self.x_vars, 
            y_cols=self.y_vars, 
            plot_type='line', 
            output_dir=path.join(self.save_dir, 'test_train_split'), 
            output_name=self.model_name,
            save_png=self.save_png, 
            save_html=self.save_html,
            add_split_lines=True)
        
    def _train_starting_message(self):
        # Print a couple line separator of special characters to make the log more readable and the model name
        [logging.info(''.join(['-'] * 50)) for _ in range(1)]
        logging.info('Training starting')
        [logging.info(''.join(['-'] * 50)) for _ in range(1)]
        logging.info(f"Training {self.model_name} with hyperparameters {self.model_hyperparameters}")

    def _print_train_end_message(self):
        # Print a couple line separator of special characters to make the log more readable and the model name
        [logging.info(''.join(['-'] * 50)) for _ in range(1)]
        logging.info(f"Training {self.model_name} complete")
        [logging.info(''.join(['-'] * 50)) for _ in range(1)]
    
    def _add_rollup_data(self, response_data):
        """
        Calculates replicate information up to a y-var level. 
        This is useful for plotting the data
        This should be called after the model has been trained

        In the future, could make this a public method called from 
        the individual model classes if there is any other additional rollup data 
        that a user is intersted in for particular models
        
        Args:
            train_data_fit (pd.DataFrame): Raw dataframe, containing data for each simulation iteration

        Returns:
            _type_: _description_
        """

        # Add the mean and std of the model responses to the train data fit, calculated as mean from the list of raw responses from the model
        response_data = response_data.copy(deep=True)
        processed_repsonses = []
        for y_var in self.y_vars:
            # Calculate a new column as the mean of the model responses
            # Add the column to the list of processed responses
            
            processed_repsonses.append(f'{y_var}_mean')             # Add the mean of the model responses
            processed_repsonses.append(f'{y_var}_median')           # Add the median of the model responses
            processed_repsonses.append(f'{y_var}_min_mean_model')   # Add the model with the minimum mean
            processed_repsonses.append(f'{y_var}_max_mean_model')   # Add the model with the maximium mean
            processed_repsonses.append(f'{y_var}_min_max_mean_avg') # Add the average of the minimum and maximum mean

            response_data[f'{y_var}_mean'] = response_data[[f'{y_var}_{self.model_name}_{i}' for i in range(self.model_hyperparameters['num_sims'])]].mean(axis=1)
            response_data[f'{y_var}_median'] = response_data[[f'{y_var}_{self.model_name}_{i}' for i in range(self.model_hyperparameters['num_sims'])]].median(axis=1)
            # Calculate the model columns that have the minimum and maximum mean
            min_mean_col = response_data[[f'{y_var}_{self.model_name}_{i}' for i in range(self.model_hyperparameters['num_sims'])]].mean().idxmin()
            max_mean_col = response_data[[f'{y_var}_{self.model_name}_{i}' for i in range(self.model_hyperparameters['num_sims'])]].mean().idxmax()
            response_data[f'{y_var}_min_mean_model'] = response_data[min_mean_col]
            response_data[f'{y_var}_max_mean_model'] = response_data[max_mean_col]
            response_data[f'{y_var}_min_max_mean_avg'] = (response_data[f'{y_var}_min_mean_model'] + response_data[f'{y_var}_max_mean_model']) / 2
  
        return processed_repsonses, response_data
        #self.train_data_fit_scaled[model_cols_to_scale] = self.scaler.unscale_data(self.train_data_fit[model_cols_to_scale], model_cols_to_scale)

    def _plot_fit(self, data_type=DN.train_data, norm = DN.not_normalized):
        # Plots the train data fit for each model along with the train data 
        # This should be used to validate that the model is fitting the data correctly
        # Build a dictionary of the columns that are not x_vars
        logging.info(f'Plotting {data_type}')
        # Byuild a dictionary with dataframes with one column each 
        if data_type in [DN.train_data, DN.train_data] + self.evaluation_data_names:
            all_data = {col: self.data_dict[norm][data_type][self.x_vars + [col]] for col in self.y_vars + self.model_responses[DN.proc] + self.model_responses[DN.raw]}
        else: 
            raise NotImplementedError("Plotting overlaid datasets from different periods not enabled yet")
        # Rename the columns with {y_var}_{model_name}_{seed} to be {y_var}. This is so we can pass the 
        for key in all_data.keys():
            cols = {key: col.split('_')[0] for col in all_data[key].columns if col not in self.x_vars}
            all_data[key] = all_data[key].rename(columns = cols)

        # Build a dict for raw and proc based off of the all_data dict
        all_data_raw = {k: v for k, v in all_data.items() if k in self.model_responses[DN.raw] + self.y_vars}
        all_data_proc = {k: v for k, v in all_data.items() if k in self.model_responses[DN.proc] + self.y_vars}
        # Plot the raw data
        title = f'{data_type} Raw'
        plot_multiple_dfs(all_data_raw,
            title=title,
            x_cols=self.x_vars, 
            y_cols=self.y_vars,
            plot_type='line', 
            output_dir=path.join(self.save_dir, 'model_predictions'),
            output_name=f'{self.model_name}_{data_type}_all_models', 
            save_png=self.save_png, 
            save_html=self.save_html,
            add_split_lines=False)
        # Plot the processed data
        title = f'{data_type} Processed'
        plot_multiple_dfs(all_data_proc,
            title=title,
            x_cols=self.x_vars, 
            y_cols=self.y_vars,
            plot_type='line', 
            output_dir=path.join(self.save_dir, 'model_predictions'),
            output_name=f'{self.model_name}_{data_type}_all_models_processed', 
            save_png=self.save_png, 
            save_html=self.save_html,
            add_split_lines=False)
        title = f'{data_type} Raw and Processed'
        # plot raw and proc data
        plot_multiple_dfs(all_data,
            title=title,
            x_cols=self.x_vars, 
            y_cols=self.y_vars,
            plot_type='line', 
            output_dir=path.join(self.save_dir, 'model_predictions'),
            output_name=f'{self.model_name}_{data_type}_all_models_raw_and_processed', 
            save_png=self.save_png, 
            save_html=self.save_html,
            add_split_lines=False)
        
    def predict(self):
        # predict the model
        raise NotImplementedError("This should be implemented by the child class")
    
    def save(self):
        # Save the model to a file using .pkl serialization or some other method, which is dependent on the library
        raise NotImplementedError("This should be implemented by the child class")

    def load(self):
        # Load the model from pkl if it exists
        if path.exists(path.join(self.save_dir, self.model_name + '.pkl')):
            with open(path.join(self.save_dir, self.model_name + '.pkl'), 'rb') as file:  
                logging.info(f"Loading model from {path.join(self.save_dir, self.model_name + '.pkl')}")
                self.model = pickle.load(file)
        else:
            logging.status(f"No model exists at {path.join(self.save_dir, self.model_name + '.pkl')}")
    
    def plot(self, 
             plot_types = ['line']
             ):
        # Plot the model
        # Use the plotting class to plot the entirety of the dataset (all columns as options)
        # Create a list of columns from the train_data_fit that do not exist in the x_vars
        # This will be used to plot all x vs y combinations
        # Plot the train data 
        train_df = self.data_dict[DN.not_normalized][DN.train_data]
        if train_df is not None:
            y_cols = [col for col in train_df.columns if col not in self.x_vars]
            # Plot all x, y combinations
        else:
            logging.error('Train Data Fit Cannot be None')
            return
        assert all([col in train_df.columns for col in self.x_vars]), "All x_vars must be in the train_data_fit"
        # Plot each individual model
        for plot_type in plot_types:
            plot_all_x_y_combinations(train_df, 
                x_cols=self.x_vars, 
                y_cols=y_cols, 
                plot_type=plot_type, 
                output_dir=path.join(self.save_dir, DN.train_data, 'single_iterations'), 
                output_name=self.model_name, 
                save_png=self.save_png, 
                save_html=self.save_html)
        
        # Overlay all the models on the same plot

    def report(self):
        # Report the model's performance
        # This should include the model hyperparameters, the model's performance on the test data, and the model's performance on the evaluation data
        # This should be saved to a file in the save directory
        # 
        # The report should include the following:
        # - Model hyperparameters
        # - Model performance on test data
        # - Model performance on evaluation data
        # - Any other relevant information
        data_dict = self.data_dict[DN.not_normalized]
        if data_dict[DN.train_data] is not None:
            logging.debug('Calculating model performance for train data')
            train_performance = self._calculate_model_performance(data_dict[DN.train_data])
            self.model_performance['train'] = train_performance
        else:
            logging.error('Train Data Fit Cannot be None')
        if data_dict[DN.train_data] is not None:
            logging.debug('Calculating model performance for test data')
            test_performance = self._calculate_model_performance(data_dict[DN.train_data])
            self.model_performance['test'] = test_performance
        else:
            logging.error('Test Data Fit Cannot be None')
        for eval_name in self.evaluation_data_names:
            logging.debug(f'Calculating model performance for {eval_name}')
            eval_performance = self._calculate_model_performance(data_dict[eval_name])
            self.model_performance[eval_name] = eval_performance
        self._write_output_csvs()

    def _calculate_model_performance(self, df):
        """Calculates model performance for all models, given a dataframe
        The dataframe can be the train, test, or evaluation data

        Args:
            df (pd.DataFrame): Input dataframe. Should contain the raw and processed 

        Returns:
            _type_: _description_
        """        
        # Calculate the model performance on the test data and evaluation data
        # This should include the RMSE, MAE, and any other relevant performance metrics
        assert df is not None, "Dataframe cannot be None"
        assert df.shape[0] > 0, "Dataframe must have at least one row"

        metrics_df = pd.DataFrame()
        for y_var in self.y_vars:
            assert y_var in df.columns, f"Response dataframe must contain the y_var {y_var}"
            # The key for the metrics_df is the {y_var}_vs_{model_name}
            for model in self.model_responses[DN.proc] + self.model_responses[DN.raw]:
                # (TODO) Ultimately have a better way to map model repsonses back to y_vars
                if y_var not in model:
                    continue
                assert model in df.columns, f"Response dataframe must contain the model response {model}. Ensure the dataframe contains both raw and processed data."
                logging.debug(f'Calculating model performance for {model=} {y_var=}')
                perf = self.__calculate_model_performance_single(df[y_var], df[model])
                metrics_df[model] = pd.DataFrame(perf, index=[model]).T
                #metrics_df[f'{y_var}_vs_{model}'] = 
        return metrics_df
    
    def __calculate_model_performance_single(self, y_data, model_data):
        """Calculates model performance of single model

        Args:
            model (str): model_response
        Returns:
            dict: Dictionary of model performance metrics
        """    
        metrics = {}
        # Calculate the RMSE    
        metrics['RMSE'] = mean_squared_error(y_data, model_data)
        # Calculate the MAE
        metrics['MAE'] = mean_absolute_error(y_data, model_data)
        # Calculate the R2
        metrics['R2'] = r2_score(y_data, model_data)
        # Calculate the confidence-weighted-calibration error 
        metrics['CWCE'] = 0
        metrics['AIC'] = 0
        metrics['BIC'] = 0
        return metrics

    def _write_output_csvs(self):
        """Writes the output data to csvs
        """        
        # Write the train and model data to csv 
        report_dir = path.join(self.save_dir, 'report')
        if not path.exists(report_dir):
            makedirs(report_dir)
        logging.info(f'Writing report to {report_dir}')


        for k, v in self.data_dict.items():
            for k1, v1 in v.items():
                if isinstance(v1, pd.DataFrame):
                    pd.DataFrame(v1).to_csv(path.join(report_dir, f'{k}_{k1}.csv'))
                else: 
                    for k2, v2 in v1.items():
                        pd.DataFrame(v2).to_csv(path.join(report_dir, f'{k}_{k1}_{k2}.csv'))

        # Write the model performance to csv
        for k, v in self.model_performance.items():
            v.to_csv(path.join(report_dir, f'{k}_model_performance.csv'))
        # Write the model hyperparameters to csv
        try:
            pd.DataFrame(self.model_hyperparameters, index=[0]).to_csv(path.join(report_dir, 'model_hyperparameters.csv'))
        except Exception as e:
            try:
                logging.warning(f"Error writing model hyperparameters to csv: {e}, trying again")
                pd.DataFrame(self.model_hyperparameters).to_csv(path.join(report_dir, 'model_hyperparameters.csv'))
            except Exception as e2:
                logging.error(f"Error writing model hyperparameters to csv: {e2}")
    def _validate_hyperparameters(self):
        # Validate the hyperparameters of the model
        raise NotImplementedError("This should be implemented by the child class")

    @classmethod
    def load_from_previous_output(cls, save_dir, model_name):
        ...