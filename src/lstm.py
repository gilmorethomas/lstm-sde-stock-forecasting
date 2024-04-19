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
        evaluation_filters:list=[]
    ):
        logging.info("Creating lstm model")
        super().__init__(data=data, 
            model_hyperparameters=model_hyperparameters, 
            save_dir=save_dir, 
            model_name=model_name,
            x_vars=x_vars,
            y_vars=y_vars,
            test_split_filter=test_split_filter,
            train_split_filter=train_split_filter,
            evaluation_filters=evaluation_filters)
        # Unpack the model hyperparameters into class member viarbles 
        # self.model = keras_LSTM(1)
        self.units = units
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


    def train(self):
        # Train the model using the test train split
        logging.info("Need to implement")
        
        # Define the number of time steps
        # TODO: Make user input in begining of code
        time_steps = 60
        
        # Function to create the dataset
        # Basically creates an array with each col -1 of the one before
        # Note: Subtracts the time step so all rows are the same length
        #   col1   col2  col3 .....
        #     0      1     2
        #     1      2     3
        #     2      3     4
        #     3      4     5
        #     4      5     6 
        #     5      6     7
        def create_dataset(data, time_steps=1):
            X, y = [], []
            for i in range(len(data) - time_steps):
                X.append(data[i:(i + time_steps), 0])
                y.append(data[i + time_steps, 0])
            return np.array(X), np.array(y)
        
        # Create the dataset with the specified time steps
        X, y = create_dataset(self.train_data[self.y_vars].values, time_steps)

        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        
        self.model = Sequential([
            keras_LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
            keras_LSTM(units=50, return_sequences=False),
            Dense(units=25),
            Dense(units=1)
        ])
        
        # Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        self.model.fit(X, y, epochs=10, batch_size=32)
        
        
        
        
        
        
        
        
        # vars_to_use = self.y_vars
        # use_x_vars = False
        # if use_x_vars: 
        #     vars_to_use = vars_to_use + self.x_vars
        # #self.train_data = self.train_data.iloc[0:10]
        # self.model = Sequential()
        # # input layer, to specify shape 
        # self.model.add(Input(shape=self.train_data[vars_to_use].shape))
        
        # self.model.add(keras_LSTM(50,return_sequences = True))
        # # more layers 
        # #self.model.add(keras_LSTM(50,return_sequences = True))
        # #self.model.add(keras_LSTM(50))
        # # output layer 
        # self.model.add(Dense(1))
        
        
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
    