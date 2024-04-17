import logging 

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
        test_split_filter=None, 
        train_split_filter=None, 
        evaluation_filters:list=[]
    ):
        logging.info("Importing LSTM packages")
        from tensorflow.keras.layers import LSTM as keras_LSTM

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
        self.model = keras_LSTM(units, **model_hyperparameters)

    def test(self):
        logging.info("Test not implemented yet")
        # Test the model
    
    def train(self):
        import pdb; pdb.set_trace()
        logging.info("Train not implemented yet")
        # Train the model using the test train split    

    def predict(self):
        logging.info("Predict not implemented yet")
        # Predict the model

    def save(self):
        logging.info("Save not implemented yet")

    def plot(self):
        logging.info("Plot not implemented yet")
    
    def report(self):
        logging.info("Report not implemented yet")
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
#     return_state=False,
#     go_backwards=False,
#     stateful=False,
#     unroll=False,
#     use_cudnn='auto',
#     **kwargs
# )
    def train(self):
        # Train the model using the test train split
        logging.info("Need to implement")
