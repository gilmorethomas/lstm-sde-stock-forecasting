from os import path, makedirs
import pickle
import logging
import pandas as pd 
import numpy as np
from plotting import plot_all_x_y_combinations, plot_multiple_dfs

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
        scaler=None, 
        save_html=True,
        save_png=True
    ):
        self.data = data
        self.model_hyperparameters = model_hyperparameters
        self.test_split_filter = test_split_filter
        self.train_split_filter = train_split_filter
        self.evaluation_filters = evaluation_filters
        self.save_dir = save_dir
        if not path.exists(save_dir):
            logging.info(f"Creating save directory {save_dir}")
            makedirs(save_dir)
        self.model_name = model_name
        self.model = None
        self.train_params = {}
        self.x_vars = x_vars
        self.y_vars = y_vars
        self._rand_state_mgr = seed
        self.scaler=scaler
        self.train_data_fit = None
        self.save_html = save_html
        self.save_png = save_png
        self.seed = seed
        
    
    def split_data(self):
        # Split the data into test and train using sklearn.model_selection.train_test_split
        # Perform test, train, evaluation split using the filters in model hyperparameters. Us
        # Set self.train_data equal to the data with the train split filter applied
        self.train_data = self.data[self.train_split_filter]
        # Set self.test_data equal to the data with the test split filter applied
        self.test_data = self.data[self.test_split_filter]
        # Set self.evaluation_data equal to the data with the evaluation filters applied
        self.evaluation_data = {k + ' Evaluation Data': self.data[v] for k, v in self.evaluation_filters.items()}
        self._plot_test_train_split()
    def _plot_test_train_split(self):
        # Build a temporary dictionary of all the data splits, including train, test, and eval
        # This will be used to plot the data
        all_data = {'train': self.train_data, 'test': self.test_data}
        all_data.update(self.evaluation_data)
        # Unscale the data
        all_data = {k: self.scaler.unscale_data(v, self.y_vars) for k, v in all_data.items()}
        # Plot the data
        plot_multiple_dfs(all_data, 
            title='Train, Test, and Evaluation Data Split',
            x_cols=self.x_vars, 
            y_cols=self.y_vars, 
            plot_type='line', 
            output_dir=path.join(self.save_dir, 'test_train_split'), 
            output_name=self.model_name, 
            save_png=self.save_png, 
            save_html=self.save_html,
            add_split_lines=True)

    def train(self):
        # train the model
        self.train_data = self.train_data.copy(deep=True)
        self.train_data_scaled = self.scaler.unscale_data(self.train_data, self.y_vars)
    
    def test(self):
        # test the model
        raise NotImplementedError("This should be implemented by the child class")
    
    def predict(self):
        # predict the model
        raise NotImplementedError("This should be implemented by the child class")
    
    def save(self):
        # Save the model to a file using .pkl serialization or some other method, which is dependent on the library
        if self.model is None:
            logging.info("No model exists yet")
        with open(path.join(self.save_dir, self.model_name + '.pkl'), 'wb') as file:  
            logging.info(f"Saving model to {path.join(self.save_dir, self.model_name + '.pkl')}")
            pickle.dump(self.model, file)

    def load(self):
        # Load the model from pkl if it exists
        if path.exists(path.join(self.save_dir, self.model_name + '.pkl')):
            with open(path.join(self.save_dir, self.model_name + '.pkl'), 'rb') as file:  
                logging.info(f"Loading model from {path.join(self.save_dir, self.model_name + '.pkl')}")
                self.model = pickle.load(file)
        else:
            logging.status(f"No model exists at {path.join(self.save_dir, self.model_name + '.pkl')}")
    
    def plot(self, plot_types = ['scatter', 'line', 'bar']):
        # Plot the model
        # Use the plotting class to plot the entirety of the dataset (all columns as options)
        # Create a list of columns from the train_data_fit that do not exist in the x_vars
        # This will be used to plot all x vs y combinations
        if self.train_data_fit is not None:
            y_cols = [col for col in self.train_data_fit.columns if col not in self.x_vars]
            # Plot all x, y combinations
        else:
            logging.error('Train Data Fit Cannot be None')
            return
        assert all([col in self.train_data_fit.columns for col in self.x_vars]), "All x_vars must be in the train_data_fit"
        for plot_type in plot_types:
            plot_all_x_y_combinations(self.train_data_fit, 
                x_cols=self.x_vars, 
                y_cols=y_cols, 
                plot_type=plot_type, 
                output_dir=path.join(self.save_dir, 'train_data_fit'), 
                output_name=self.model_name, 
                save_png=self.save_png, 
                save_html=self.save_html)

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

        # Write the train and model data to csv 
        logging.info(f'Writing report to {self.save_dir}')
        pd.DataFrame(self.train_data).to_csv(path.join(self.save_dir, 'train_data.csv'))
        pd.DataFrame(self.test_data).to_csv(path.join(self.save_dir, 'test_data.csv'))
        for k, v in self.evaluation_data.items():
            pd.DataFrame(v).to_csv(path.join(self.save_dir, f'evaluation_data_{k}.csv'))
        # Write the train fit data to csv
        if self.train_data_fit is not None:
            pd.DataFrame(self.train_data_fit).to_csv(path.join(self.save_dir, 'train_data_fit.csv'))
        else:
            logging.info('No training data found, bypassing writing to csv')
    def _validate_hyperparameters(self):
        # Validate the hyperparameters of the model
        raise NotImplementedError("This should be implemented by the child class")
