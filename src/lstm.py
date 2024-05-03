from lstm_logger import logger as logging
from project_globals import DataNames as DN
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import math
 
from tensorflow.keras.layers import LSTM as keras_LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from timeseriesmodel import TimeSeriesModel
from sklearn.preprocessing import MinMaxScaler
from utils import parallelize
import copy

class LSTM(TimeSeriesModel):
    # Define an LSTM class that inherits from the model class, implemented using pytorch as similar to this link: 
    # https://www.datacamp.com/tutorial/lstm-python-stock-market
    # https://www.tensorflow.org/tutorials/structured_data/time_series
    # https://colah.github.io/posts/2015-08-Understanding-LSTMs/
    # https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21
    def __init__(self, 
        data, 
        units, 
        model_hyperparameters, 
        save_dir, 
        model_name, 
        x_vars, 
        y_vars, 
        seed:np.random.RandomState,
        test_split_filter=None, 
        train_split_filter=None, 
        evaluation_filters:list=[],
        save_html=True,
        save_png=True, 
        
    ):
        logging.info("Creating lstm model")
        super().__init__(data=data, 
            model_hyperparameters=model_hyperparameters, 
            save_dir=save_dir, 
            model_name=model_name,
            x_vars=x_vars,
            y_vars=y_vars,
            seed=seed,
            test_split_filter=test_split_filter,
            train_split_filter=train_split_filter,
            evaluation_filters=evaluation_filters,
            save_html=save_html,
            save_png=save_png)
        # Unpack the model hyperparameters into class member viarbles 
        # self.model = keras_LSTM(1)
        #self.model = Sequential()
        #self.model.add(keras_LSTM(units, **model_hyperparameters))
        #self.model.add(Dense(1))  # Output layer
    def fit(self): 
        """Performs test, train, and evaluation of the model
        """
        assert len(self.y_vars) == 1, 'Only one y_var is supported for LSTM models'
        # Train the model using the test train split
        self._set_hyperparameter_defaults()
        for column in self.x_vars:
            if self.data_dict[DN.normalized][DN.train_data][column].dtype == int: 
                # check for gaps in the data
                if not all(np.diff(self.data_dict[DN.normalized][DN.train_data][column]) == 1):
                    logging.error(f"Column {column} has gaps in the data")
                    return
        
        # make a deep copy of data dict
        data_dict = copy.deepcopy(self.data_dict)
        train_data_scaled = data_dict[DN.normalized][DN.train_data].copy(deep=True)
        train_data = data_dict[DN.not_normalized][DN.train_data].copy(deep=True)
        # Scale the data
        for y_var in self.y_vars:
            fit_data_one_var = self._fit_one_y_var(self.data_dict, y_var)

        # Not sure that this will work for multiple y vars           
        logging.warning('Not yet implemented for multiple y vars') 
        data_dict = fit_data_one_var
        super().fit(data_dict)

    
    def _fit_one_y_var(self, data_dict, y_var):
        """Helper method to train, test, and evaluate the model on one y variable

        Args:
            data_dict (dict): Dictionary of data to use for training, testing, and evaluation. Keys are the data names, 
                values are either dataframes in the case of train_data and test_data, or dictionaries in the case of
                evaluation data. The evaluation data dictionaries have keys that are the evaluation filter names, and
                the values are the dataframes.
            y_var (str): The y variable to use for training, testing, and evaluation
        """        
        # Temporarily remove lambdas because these are not pkl-able for multiprocessing
        tmp_test_split_filter=self.test_split_filter
        tmp_train_split_filter=self.train_split_filter
        tmp_evaluation_filters=self.evaluation_filters
        self.evaluation_filters = None
        self.test_split_filter = None
        self.train_split_filter = None
        # Call the create dataset function for each entry in the data_dict
        parallelize_args = []
        x_y_data_dict = {}
        data_dict_not_norm = data_dict[DN.not_normalized]
        data_dict = data_dict[DN.normalized]
        # Need to create the x and y data for each entry in the data_dict
        # For test and evaluation data, we have to bring in data from preceeding time steps, using the time_steps parameter and data_dict[DN.normalized][DN.all_data]

        x_y_data_dict[DN.train_data] = self._create_dataset(data=np.array(data_dict[DN.train_data][y_var]).reshape(-1, 1),
            time_steps=self.model_hyperparameters['time_steps'])
        
        
        test_data_input = data_dict[DN.test_data][self.x_vars + [y_var]]
        train_data_to_prepend = data_dict[DN.train_data].iloc[-self.model_hyperparameters['time_steps']:][self.x_vars + [y_var]]
        # Must have a full window of train data
        assert train_data_to_prepend.shape[0] == self.model_hyperparameters['time_steps'], "Train data to prepend is not the correct size"
        # Append the previous time steps to the test data using the time_steps parameter and data_dict[DN.all_data]
        test_data_input = pd.concat([train_data_to_prepend, test_data_input]).sort_values(by='Date')[y_var]
        x_y_data_dict[DN.test_data] = self._create_dataset(data=np.array(test_data_input).reshape(-1, 1),
            time_steps=self.model_hyperparameters['time_steps'])
        # Iterate over the evaluation data and create the x 
        x_y_data_dict.update({eval_filter: {} for eval_filter in self.evaluation_data_names})
        for eval_filter in self.evaluation_data_names: 
            eval_data_input = data_dict[eval_filter][self.x_vars + [y_var]]
            test_data_to_prepend = data_dict[DN.test_data].iloc[-self.model_hyperparameters['time_steps']:][self.x_vars + [y_var]]
            # If train data is smaller than the time steps, we need to prepend the remaining time steps with the train data
            if len(test_data_to_prepend) < self.model_hyperparameters['time_steps']:
                steps_to_prepend = self.model_hyperparameters['time_steps'] - len(test_data_to_prepend)
                logging.warning(f"Train data is smaller than the time steps = {self.model_hyperparameters['time_steps']}, prepending with train data for {steps_to_prepend} steps")
                train_data_to_prepend = data_dict[DN.train_data].iloc[-steps_to_prepend:][self.x_vars + [y_var]]
                test_data_to_prepend = pd.concat([train_data_to_prepend, test_data_to_prepend])
            eval_data_input = pd.concat([test_data_to_prepend, eval_data_input]).sort_values(by='Date')[y_var]
            # Append the previous time steps to the test data using the time_steps parameter and data_dict[DN.normalized][DN.all_data]
            x_y_data_dict[eval_filter] = self._create_dataset(data=np.array(eval_data_input).reshape(-1, 1),
                time_steps=self.model_hyperparameters['time_steps'])
            
        # Temporarily remove lambdas because these are not pkl-able for multiprocessing
        tmp_test_split_filter=self.test_split_filter
        tmp_train_split_filter=self.train_split_filter
        tmp_evaluation_filters=self.evaluation_filters
        self.evaluation_filters = None
        self.test_split_filter = None
        self.train_split_filter = None
        # Train for each seed, parallelizing task
        parallelize_args = []
        for seed_num in range(self.model_hyperparameters['num_sims']):
            seed = self.seed.random()
            parallelize_args.append([seed_num, x_y_data_dict, self.model_hyperparameters])

        # Execute all the models in parallel.. TODO.. we cannot return the model if we parallelize
        out_data = parallelize(self._gen_and_predict_for_seed, parallelize_args, run_parallel=False)    

        y_var_data = self._bould_output_data(out_data, copy.deepcopy(self.data_dict[DN.normalized]), copy.deepcopy(self.data_dict[DN.not_normalized]), y_var)
        
        # TODO get rid of the temp removal of the filters 
        self.test_split_filter = tmp_test_split_filter
        self.train_split_filter = tmp_train_split_filter
        self.evaluation_filters = tmp_evaluation_filters

        return y_var_data
    def _bould_output_data(self, out_data, data_dict, data_dict_not_norm, y_var):
        """Builds the output data for a single y variable. TODO ultimately make this a common function between lstm_sde and lstm 
        since they are trying to accomplish the same thing from an interface perspective

        Args:
            out_data (_type_): _description_
            data_dict (_type_): _description_
            data_dict_not_norm (_type_): _description_
            y_var (_type_): _description_

        Returns:
            _type_: _description_
        """            
        # Need to prepend nans for each model result
        nan_array = np.array([[np.nan] * self.model_hyperparameters['time_steps']]).T
        # Iterate over the output data and add it to the train data fit array
        fit_data = {'train_predict': {}, 'test_predict': {}}
        fit_data.update({f'{eval_filter}_predict' : {} for eval_filter in self.evaluation_data_names})

        # Build the output data
        for seed_num, model, train_data_fit_one_seed in out_data:
            # Pandas concatatenate the data into the
            data_dict[DN.test_data].loc[:, f'{y_var}_{seed_num}'] = train_data_fit_one_seed['test_predict']
            data_dict_not_norm[DN.test_data].loc[:, f'{y_var}_{seed_num}'] = self.scaler.inverse_transform(train_data_fit_one_seed['test_predict'])
            data_dict[DN.train_data].loc[:, y_var + f'_{seed_num}'] = np.concatenate([nan_array, train_data_fit_one_seed['train_predict']])
            data_dict_not_norm[DN.train_data].loc[:, y_var + f'_{seed_num}'] = self.scaler.inverse_transform(np.concatenate([nan_array, train_data_fit_one_seed['train_predict']]))
            # drop nans to account for windows for rollback data
            data_dict[DN.train_data].dropna(inplace=True)
            data_dict_not_norm[DN.train_data].dropna(inplace=True) 
            [data_dict_not_norm[eval_data].dropna(inplace=True) for eval_data in self.evaluation_data_names]

            for eval_filter in self.evaluation_data_names:
                data_dict[eval_filter].loc[: ,y_var + f'_{seed_num}'] = train_data_fit_one_seed[eval_filter]
                data_dict_not_norm[eval_filter].loc[: ,y_var + f'_{seed_num}'] = self.scaler.inverse_transform(train_data_fit_one_seed[eval_filter])
            self.model_objs.append(model)


        return {DN.normalized : data_dict, DN.not_normalized: data_dict_not_norm}
    
    def _gen_and_predict_for_seed(self, seed_num, x_y_data_dict, model_hyperparameters):
        """Generates and predicts the model for a given seed

        Args:
            seed_num (_type_): _description_
            x_train (_type_): _description_
            y_train (_type_): _description_
            model_hyperparameters (_type_): _description_

        Returns:
            _type_: _description_
        """        
        #seed = self.seed.random()
        model = self._generate_model_for_seed(x_y_data_dict[DN.train_data][0], x_y_data_dict[DN.train_data][1], model_hyperparameters)
        #model.compile()
        # TODO when we turn this to a parallel function and try to return the model, we get a pkl error
        # Is there a way to write this model to a file and then read it back in outside of the parallel function?
        model.fit(x_y_data_dict[DN.train_data][0], # x_train
            x_y_data_dict[DN.train_data][1], # y_train
            validation_data=(x_y_data_dict[DN.test_data][0], x_y_data_dict[DN.test_data][1]),
            epochs=model_hyperparameters['epochs'], 
            batch_size=model_hyperparameters['batch_size'], 
            verbose=model_hyperparameters['verbose'])
        # Now predict performance
        logging.info('Predicting performance for train, test, and evaluation data')
        predictions = {}
        predictions['train_predict'] = model.predict(x_y_data_dict[DN.train_data][0]) 
        predictions['test_predict'] = model.predict(x_y_data_dict[DN.test_data][0])
        for eval_filter in self.evaluation_data_names:
            predictions[eval_filter] = model.predict(x_y_data_dict[eval_filter][0])
        logging.info('Done predicting performance for train, test, and evaluation data')

        return seed_num, model, predictions

    def _generate_model_for_seed(self, x_train, y_train, model_hyperparameters):
        """Generates a model for a given seed

        Args:
            x_train (_type_): _description_
            y_train (_type_): _description_
            seed (_type_): _description_

        Returns:
            _type_: _description_
        """        
        model = Sequential()
        model.add(keras_LSTM(model_hyperparameters['hidden_nodes'], return_sequences=True, input_shape=(x_train.shape[1], 1)))
        
        #Build other layers other than the default input
        for i in range(model_hyperparameters['num_layers'] - 2):
           #assert len(model_hyperparameters['hidden_nodes']) > i, 'Not enough hidden nodes specified for the number of layers'
           model.add(keras_LSTM(model_hyperparameters['hidden_nodes'], return_sequences=True))
        model.add(keras_LSTM(model_hyperparameters['hidden_nodes']))
        
        #Add output layers
        model.add(Dense(model_hyperparameters['output_dimension']))
        model.compile(optimizer=model_hyperparameters['optimizer'], 
           loss=model_hyperparameters['loss'])
        #Train the model
        model.summary()

        return model
        

    def save(self):
        logging.info("Save not implemented yet")

    def plot(self):
        # Call the base model class plotting function 
        super().plot()
        # Add any additional custom plots 
        logging.info("Joey, implement any custom plots needed")    
    def report(self):
        super().report()

    def _create_dataset(self, data, time_steps=10):
        # Create a rolling window of time steps for the data
        # This function creates a dataset with time_steps number of columns
        # Each column is the previous column shifted by 1
        # The first column is the first time_steps rows of the data
        # The second column is the second time_steps rows of the data
        # etc.
        # Example:
        # data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # time_steps = 3
        # output = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]]
        x, y = [], []
        # 
        for i in range(len(data) - time_steps):
            x.append(data[i:(i + time_steps), 0])
            y.append(data[i + time_steps, 0])
        return np.array(x), np.array(y)

    def _set_hyperparameter_defaults(self):
        # Set the hyperparameters for the model
        # If the user specifies the time_steps in the model hyperparameters, use that
        # Otherwise, default to the length of the training data
        # Define the default hyperparameters for the model
        if 'time_steps' not in self.model_hyperparameters: 
            logging.info('No time steps specified in model hyperparameters, using default of 100')
            self.model_hyperparameters['time_steps'] = 100
        if 'num_layers' not in self.model_hyperparameters: 
            logging.info('No num_layers specified in model hyperparameters, using default of 1')
            self.model_hyperparameters['num_layers'] = 2
        if 'optimizer' not in self.model_hyperparameters: 
            logging.info('No optimizer specified in model hyperparameters, using default of adam')
            self.model_hyperparameters['optimizer'] = 'adam'
        if 'loss' not in self.model_hyperparameters:
            logging.info('No loss specified in model hyperparameters, using default of mean_squared_error')
            self.model_hyperparameters['loss'] = 'mean_squared_error'
        if 'output_dimension' not in self.model_hyperparameters:
            logging.info('No output_dimension specified in model hyperparameters, using default of 1')
            self.model_hyperparameters['output_dimension'] = 1
        if 'hidden_nodes' not in self.model_hyperparameters:
            logging.info('No hidden_nodes specified in model hyperparameters, using default of 50')
            self.model_hyperparameters['hidden_nodes'] = 50 
        if 'epochs' not in self.model_hyperparameters:
            logging.info('No epochs specified in model hyperparameters, using default of 100')
            self.model_hyperparameters['epochs'] = 2
        if 'batch_size' not in self.model_hyperparameters:
            logging.info('No batch_size specified in model hyperparameters, using default of 32')
            self.model_hyperparameters['batch_size'] = 32
        if 'verbose' not in self.model_hyperparameters:
            logging.info('No verbose specified in model hyperparameters, using default of 1')
            self.model_hyperparameters['verbose'] = 1
        if 'num_sims' not in self.model_hyperparameters:
            logging.info('No num_sims specified in model hyperparameters, using default of 1')
            self.model_hyperparameters['num_sims'] = 2
    
    @classmethod
    def load_from_previous_output(cls, class_params):# , save_dir, model_name):
        instance = super().load_from_previous_output(class_params)
        return instance
        # Any custom stuff needed here