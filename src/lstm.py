import logging 
import numpy as np
from model import Model
from tensorflow.keras.layers import LSTM as keras_LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from timeseriesmodel import TimeSeriesModel
class LSTM(TimeSeriesModel):
    # Define an LSTM class that inherits from the model class, implemented using pytorch as similar to this link: 
    # https://www.datacamp.com/tutorial/lstm-python-stock-market
    # https://www.tensorflow.org/tutorials/structured_data/time_series

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
        scaler=None,
        save_html=True,
        save_png=True
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
            scaler=scaler,
            save_html=save_html,
            save_png=save_png)
        # Unpack the model hyperparameters into class member viarbles 
        # self.model = keras_LSTM(1)
        #self.model = Sequential()
        #self.model.add(keras_LSTM(units, **model_hyperparameters))
        #self.model.add(Dense(1))  # Output layer


    def test(self):
        logging.info("Test not implemented yet")
        # Test the model
    
   # def train(self):
   #     import pdb; pdb.set_trace()
   #     logging.info("Train not implemented yet")
   #     # Train the model using the test train split    

    def predict(self):

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
        if 'num_seeds' not in self.model_hyperparameters:
            logging.info('No num_seeds specified in model hyperparameters, using default of 1')
            self.model_hyperparameters['num_seeds'] = 2
    def train(self):
        assert len(self.y_vars) == 1, 'Only one y_var is supported for LSTM models'
        # Train the model using the test train split
        self._set_hyperparameter_defaults()
        logging.info("Need to implement")
        # assert that there are no gaps in x_vars
        for column in self.x_vars:
            if self.train_data[column].dtype == int: 
                # check for gaps in the data
                if not all(np.diff(self.train_data[column]) == 1):
                    logging.error(f"Column {column} has gaps in the data")
                    return
        x_train, y_train = self._create_dataset(data=np.array(self.train_data[self.y_vars]), 
            time_steps=self.model_hyperparameters['time_steps'])
        
        self.train_data_fit = self.train_data.copy(deep=True)
        
        for seed_num in range(self.model_hyperparameters['num_seeds']):
            seed = self.seed.random()
            train_data_fit = self._train_one_seed(x_train, y_train, seed)
            # Prepend train data with np.nan equal to the number of time steps in hyperparameters
            nan_array = np.array([[np.nan] * self.model_hyperparameters['time_steps']]).T
            train_data_fit = np.concatenate((nan_array, train_data_fit), axis=0)
            # Add this data to the train fit array
            self.train_data_fit[f'{self.y_vars[0]}_{self.model_name}_{seed_num}'] = train_data_fit

    def _train_one_seed(self, x_train, y_train, seed):
        """Trains the model on one seed of the data

        Args:
            x_train (_type_): _description_
            y_train (_type_): _description_
        """        
        model = Sequential()
        # https://colah.github.io/posts/2015-08-Understanding-LSTMs/
        model.add(keras_LSTM(self.model_hyperparameters['hidden_nodes'][0], return_sequences=True, input_shape=(x_train.shape[1], 1)))
        # Build other layers other than the default
        for i in range(self.model_hyperparameters['num_layers'] - 2):
            assert len(self.model_hyperparameters['hidden_nodes']) > i, 'Not enough hidden nodes specified for the number of layers'
            model.add(keras_LSTM(self.model_hyperparameters['hidden_nodes'][i - 1], return_sequences=True))
        model.add(keras_LSTM(self.model_hyperparameters['hidden_nodes'][0]))
        
        # Add output layer
        model.add(Dense(self.model_hyperparameters['output_dimension']))
        model.compile(optimizer=self.model_hyperparameters['optimizer'], 
            loss=self.model_hyperparameters['loss'])
        # Train the model
        model.summary()

        model.fit(x_train, 
            y_train, 
            epochs=self.model_hyperparameters['epochs'], 
            batch_size=self.model_hyperparameters['batch_size'], 
            verbose=self.model_hyperparameters['verbose'])
        
        train_predict = model.predict(x_train)
        return train_predict 
        # Merge the train data fit with the train data 
        # # TODO (replace w/ model hyperparameters)
        # learning_rate = 0.001
        # beta_1 = 0.9
        # beta_2 = 0.999
        # epsilon = 0.0001
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
    