from os import path, makedirs
import pickle
import logging
import pandas as pd 
import numpy as np
from plotting import plot_all_x_y_combinations, plot_multiple_dfs
from scaler import Scaler

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
        self.save_dir = save_dir
        if not path.exists(save_dir):
            logging.info(f"Creating save directory {save_dir}")
            makedirs(save_dir)
        self.model_name = model_name
        self.model = None
        self.train_params = {}
        self.x_vars = x_vars
        self.y_vars = y_vars
        self.model_responses = {'raw': [],
                                'processed': [],}
        self._rand_state_mgr = seed
        self.train_data_fit = None
        self.save_html = save_html
        self.save_png = save_png
        self.seed = seed
        self.scaler = Scaler()

        # Save scaled and unscaled data
        # Add the data to the scaler
        self.scaler.fit(df=self.data, df_name='all_data', scaler_type='MinMaxScaler')
        # by default the data is scaled
        self.data = self.scaler.transform(df=self.data, df_name='all_data', columns = self.y_vars)
        self._build_model_response_dict()

    def _build_model_response_dict(self):
        # Build model response raw list from y_vars with model name and number of sims appended
        self.model_responses['raw'] = [f'{y_var}_{self.model_name}_{i}' for y_var in self.y_vars for i in range(self.model_hyperparameters['num_sims'])]
    
    def split_data(self):
        # Perform test, train, evaluation split using the filters in model hyperparameters. Us
        # Set self.train_data equal to the data with the train split filter applied
        data_unscaled = self.scaler.inverse_transform(df=self.data, df_name='all_data', columns=self.y_vars)                

        # Split the data into test and train using filters

        train_data_unscaled = data_unscaled[self.train_split_filter]
        
        # Set self.test_data equal to the data with the test split filter applied
        test_data_unscaled = data_unscaled[self.test_split_filter]
        # Set self.evaluation_data equal to the data with the evaluation filters applied
        evaluation_data_unscaled = {k + ' Evaluation Data': data_unscaled[v] for k, v in self.evaluation_filters.items()}

        # Add the data to the scaler. First need to unscale the data 
        self.train_data = self.scaler.transform(df=train_data_unscaled, df_name='train_data', columns=self.y_vars)

        self.test_data = self.scaler.transform(df=test_data_unscaled, df_name='test_data', columns=self.y_vars)
        # Add the evaluation data to the scaler
        self.evaluation_data = {k: self.scaler.transform(df=v, df_name=k, columns=self.y_vars) for k, v in evaluation_data_unscaled.items()}
        self._plot_test_train_split()
    def _plot_test_train_split(self):
        """Plots the test train split data
        """        
        # Build a temporary dictionary of all the data splits, including train, test, and eval
        # This will be used to plot the data
        all_data = {'train_data': self.train_data, 'test_data': self.test_data}
        all_data.update(self.evaluation_data)
        # Unscale the data
        #all_data = {k: self.scaler.inverse_transform(df_name =k, df=v, columns=self.y_vars) for k, v in all_data.items()}
        # Plot the data
        
        plot_multiple_dfs(all_data, 
            title='Train, Test, and Evaluation Data Split',
            x_cols=self.x_vars, 
            y_cols=self.y_vars, 
            plot_type='line', 
            output_dir=path.join(self.save_dir, 'test_train_split'), 
            output_name=self.model_name + '_scaled', 
            save_png=self.save_png, 
            save_html=self.save_html,
            add_split_lines=True)
        
        # generate unscaled data, using the scaler to perform inverse transform on each data split
        all_data = {k: self.scaler.inverse_transform(df=v, df_name=k, columns=self.y_vars) for k, v in all_data.items()}
        #all_data = {k: self.scaler.inverse_transform(df_name =k, df=v, columns=self.y_vars) for k, v in all_data.items()}
        # Plot the unscaled data
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
    def _train_starting_message(self):
        # Print a couple line separator of special characters to make the log more readable and the model name
        [logging.info(''.join(['-'] * 50)) for _ in range(2)]
        logging.info(f"Training {self.model_name} with hyperparameters {self.model_hyperparameters}")

    def _print_train_end_message(self):
        # Print a couple line separator of special characters to make the log more readable and the model name
        logging.info(f"Training {self.model_name} complete")
        [logging.info(''.join(['-'] * 50)) for _ in range(2)]

    def train(self):
        # train the model
        import pdb; pdb.set_trace()
        self.train_data = self.train_data.copy(deep=True)
        self.train_data_scaled = self.scaler.inverse_transform(df=self.train_data, df_name='train_data', columns=self.y_vars)
        # Unscale model fit data. Since the model data columns will have y_col_{model_name} we need to check for that
        import pdb; pdb.set_trace()
        self._add_rollup_data()
        # Scale back down 
        #self.train_data_fit = self.scaler.transform(df=self.train_data_fit, df_name='train_data_fit', columns=model_cols_to_scale)



        self.train_data_fit_scaled = self.train_data_fit.copy(deep=True)

    def _add_rollup_data(self):
        """Calculates replicate information up to a y-var level. 
        This is useful for plotting the data
        This should be called after the model has been trained

        In the future, could make this a public method called from 
        the individual model classes if there is any other additional rollup data 
        that a user is intersted in
        
        """        
        import pdb; pdb.set_trace() 
        # Add the mean and std of the model responses to the train data fit, calculated as mean from the list of raw responses from the model
        for y_var in self.y_vars:
            # Calculate the mean and std of the model responses
            self.train_data_fit_scaled[f'{y_var}_mean'] = self.train_data_fit_scaled[[col for col in self.train_data_fit_scaled.columns if y_var in col]].mean(axis=1)
            self.train_data_fit_scaled[f'{y_var}_std'] = self.train_data_fit_scaled[[col for col in self.train_data_fit_scaled.columns if y_var in col]].std(axis=1)


            # # Take the mean of the simulations and assign it to the train_data_fit dataframe
            # train_data_fit[f'{col}_{self.model_name}_mean'] = gbm_data.mean(axis=1)
            # # Take the median of the simulations and assign it to the train_data_fit dataframe
            # train_data_fit[f'{col}_{self.model_name}_median'] = np.median(gbm_data, axis=1)
            # # Take the max of the simulations and assign it to the train_data_fit dataframe
            # train_data_fit[f'{col}_{self.model_name}_max'] = gbm_data.max(axis=1)
            # # Take the min of the simulations and assign it to the train_data_fit dataframe
            # train_data_fit[f'{col}_{self.model_name}_min'] = gbm_data.min(axis=1)
            # train_data_fit[f'{col}_{self.model_name}_min_max_avg'] = .5 * train_data_fit[f'{col}_{self.model_name}_max'] + .5 * train_data_fit[f'{col}_{self.model_name}_min']



        # Calculate the RMSE
        #import pdb; pdb.set_trace()

        #for seed_num in range(self.model_hyperparameters['num_seeds']):
            
            # Prepend train data with np.nan equal to the number of time steps in hyperparameters
            #nan_array = np.array([[np.nan] * self.model_hyperparameters['time_steps']]).T
            #train_data_fit = np.concatenate((nan_array, train_data_fit), axis=0)
            #model_rmse{f'{self.model_name}_{seed_num}'} = math.sqrt(mean_squared_error(y_train, train_data_fit))
            # Add this data to the train fit array
            #self.train_data_fit[f'{self.y_vars[0]}_{self.model_name}_{seed_num}'] = train_data_fit

        #self.train_rmse = model_rmse
        #import pdb; pdb.set_trace()
        # Unscale the data
        #import pdb; pdb.set_trace()
        self.train_data_fit_scaled[model_cols_to_scale] = self.scaler.unscale_data(self.train_data_fit[model_cols_to_scale], model_cols_to_scale)
        self._plot_train_data()
        self._print_train_end_message()
    def _plot_train_data(self):
        # Plots the train data fit for each model along with the train data
        # This should be used to validate that the model is fitting the data correctly
        #y_cols=[col for col in self.train_data_fit_scaled if col not in self.x_vars]
        # Build a dictionary of the columns that are not x_vars
        logging.info('Plotting train data')
        keys = [col for col in self.train_data_fit_scaled.columns if col not in self.x_vars]
        # Add 
        all_data = {key: self.train_data_fit_scaled[self.x_vars + [key]] for key in keys}
        # Rename the columns with {y_var}_{model_name}_{seed} to be {y_var}. This is so we can pass the 
        # same column to the plotting function across multiple dataframes
        for key in all_data.keys():
            cols = {key: col.split('_')[0] for col in all_data[key].columns if col not in self.x_vars}
            all_data[key] = all_data[key].rename(columns = cols)

        # Plot the data
        plot_multiple_dfs(all_data, 
            title='Model Train Data',
            x_cols=self.x_vars, 
            y_cols=self.y_vars, 
            plot_type='line', 
            output_dir=path.join(self.save_dir, 'train_data', 'all_iterations'),
            output_name=self.model_name + '_train_data_all_models', 
            save_png=self.save_png, 
            save_html=self.save_html,
            add_split_lines=False)


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
        # Plot each individual model
        for plot_type in plot_types:
            plot_all_x_y_combinations(self.train_data_fit, 
                x_cols=self.x_vars, 
                y_cols=y_cols, 
                plot_type=plot_type, 
                output_dir=path.join(self.save_dir, 'train_data', 'single_iterations'), 
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
