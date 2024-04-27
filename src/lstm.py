import logging 
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import math
from model import Model
from tensorflow.keras.layers import LSTM as keras_LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from timeseriesmodel import TimeSeriesModel

from utils import parallelize


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

    def train(self):
        """Trains the model
        """        
        assert len(self.y_vars) == 1, 'Only one y_var is supported for LSTM models'
        # Train the model using the test train split
        self._set_hyperparameter_defaults()
        model_rmse = {}
        logging.info("Need to implement")
        # Check that there are no gaps in x_vars
        for column in self.x_vars:
            if self.train_data[column].dtype == int: 
                # check for gaps in the data
                if not all(np.diff(self.train_data[column]) == 1):
                    logging.error(f"Column {column} has gaps in the data")
                    return
        train_data_scaled = self.train_data.copy(deep=True)
        train_data = self.train_data.copy(deep=True)
        # Scale the data
        train_data_scaled = self.scaler.transform(df=train_data_scaled, df_name = 'train_data', columns=self.y_vars) #.reshape(-1, 1)
        for y_var in self.y_vars:
            y_trained = self._train_one_y_var(train_data_scaled[[y_var] + self.x_vars], y_var)
            train_data_scaled = pd.merge(train_data_scaled, y_trained,  on='Days_since_start')

            this_scaler = self.scaler._scalers['train_data']['data'][y_var]['scaler']
            # Unscale the data
            all_resps = [y_var] + self.model_responses['raw'] 
            train_data[all_resps] = this_scaler.inverse_transform(y_trained[all_resps])
        # Now unscale the data

            

        
        super().train(train_data)

    def _train_one_y_var(self, train_data_scaled, y_var):
        """Trains the model on one y variable

        Args:
            train_data_scaled (pd.DataFrame): Array with data for the y varaible 
            y_var (_type_): _description
        """        
        x_train, y_train = self._create_dataset(data=np.array(train_data_scaled[y_var]).reshape(-1, 1),
            time_steps=self.model_hyperparameters['time_steps'])
        
        train_data_fit = train_data_scaled.copy(deep=True)
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
            #this_model = self._generate_model_for_seed(x_train, y_train, seed, self.model_hyperparameters) 
            # parallelize_args.append([seed_num, this_model, x_train, y_train, self.model_hyperparameters])
            parallelize_args.append([seed_num, x_train, y_train, self.model_hyperparameters])

            # parallelize_args.append([1])

        # Execute all the models in parallel.. TODO.. we cannot return the model if we 
        out_data = parallelize(self._gen_and_predict_for_seed, parallelize_args, run_parallel=False)    
        
        # Need to prepend nans for each model result
        nan_array = np.array([[np.nan] * self.model_hyperparameters['time_steps']]).T
        # Iterate over the output data and add it to the train data fit array
        for seed_num, model, train_data_fit_one_seed in out_data:
            train_data_fit[f'{y_var}_{self.model_name}_{seed_num}'] = np.concatenate((nan_array, train_data_fit_one_seed))
            self.model_objs.append(model)
        
        self.test_split_filter = tmp_test_split_filter
        self.train_split_filter = tmp_train_split_filter
        self.evaluation_filters = tmp_evaluation_filters
        #train_data_fit_one_seed = this_model.predict(x_train)
        return train_data_fit
    
    def _gen_and_predict_for_seed(self, seed_num, x_train, y_train, model_hyperparameters):
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
        model = self._generate_model_for_seed(x_train, y_train, model_hyperparameters)
        #model.compile()
        # TODO when we turn this to a parallel function and try to return the model, we get a pkl error
        # Is there a way to write this model to a file and then read it back in outside of the parallel function?
        return seed_num, model, self._predict_model(seed_num, model, x_train, y_train, model_hyperparameters)
    
    def _predict_model_2(self, seed_num, x_train, y_train, model):
        print('hi')
        return model

    def _predict_model(self, seed_num, model, x_train, y_train, model_hyperparameters):
        """Predicts the model

        Args:
            model (_type_): _description_
            x_train (_type_): _description_

        Returns:
            _type_: _description_
        """        
        model.fit(x_train, 
            y_train, 
            epochs=model_hyperparameters['epochs'], 
            batch_size=model_hyperparameters['batch_size'], 
            verbose=model_hyperparameters['verbose'])
        return model.predict(x_train)
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
        model.add(keras_LSTM(model_hyperparameters['hidden_nodes'][0], return_sequences=True, input_shape=(x_train.shape[1], 1)))
        
        #Build other layers other than the default input
        for i in range(model_hyperparameters['num_layers'] - 2):
           assert len(model_hyperparameters['hidden_nodes']) > i, 'Not enough hidden nodes specified for the number of layers'
           model.add(keras_LSTM(model_hyperparameters['hidden_nodes'][i - 1], return_sequences=True))
        model.add(keras_LSTM(model_hyperparameters['hidden_nodes'][0]))
        
        #Add output layers
        model.add(Dense(model_hyperparameters['output_dimension']))
        model.compile(optimizer=model_hyperparameters['optimizer'], 
           loss=model_hyperparameters['loss'])
        #Train the model
        model.summary()

        return model
        

    def test(self):
        logging.info("Test not implemented yet")
        # Test the model
    
   # def train(self):
   #     import pdb; pdb.set_trace()
   #     logging.info("Train not implemented yet")
   #     # Train the model using the test train split    

    def predict(self):
        return
        logging.info("Predict not implemented yet")
        # Predict the model
        X_test = np.array(self.test_data[self.y_vars])
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        predictions = self.model.predict(X_test)
        # TODO: predictions needs to be unscaled

    def save(self):
        logging.info("Save not implemented yet")

    def plot(self):
        # Call the base model class plotting function 
        super().plot()
        # Add any additional custom plots 
        logging.info("Joey, implement any custom plots needed")    
    def report(self):
        logging.info("Report not implemented yet")

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
            self.model_hyperparameters['hidden_nodes'] = [50 for i in range(self.model_hyperparameters['num_layers'])]
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


        # Merge the train data fit with the train data 
        # # TODO (replace w/ model hyperparameters)
        # learning_rate = 0.001
        # beta_1 = 0.9
        # beta_2 = 0.999
        # epsilon = 0.0001F
        # decay = 0.0
        # adam_opt = Adam(learning_rate = learning_rate)
        #                # beta_1=beta_1,
        #                 #beta_2=beta_2,
        #                 #epsilon=epsilon,
        #                 #decay=decay,
        #                 #amsgrad=False)
        # #self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        # self.model.compile(optimizer=adam_opt, loss='mse', metrics=['accuracy'])

        # #self.model.compile(loss='mean_squared_error')
        
        # y_train = self.train_data[vars_to_use].to_numpy()
        
        # self.model.test = self.model.fit(y_train)

#if __name__ == "__main__":
    
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
    #     return_state=False,Dense
    #     go_backwards=False,
    #     stateful=False,
    #     unroll=False,
    #     use_cudnn='auto',
    #     **kwargs
    # )
    