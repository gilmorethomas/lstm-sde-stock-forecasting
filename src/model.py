from os import path 
import pickle
import logging

class Model(): 
    # Define a model class that takes a test train split and model hyperparameters
    # The model class should have a train method that trains the model
    # The model class should have a test method that tests the model
    # The model class should have a predict method that predicts the model
    # The model class should have a save method that saves the model
    # The model class should have a load method that loads the model
    # The model class should have a plot method that plots the model
    # The model class should have a report method that reports the model's performance

    def __init__(self, model_hyperparameters, save_dir, model_name):
        self.model_hyperparameters = model_hyperparameters
        self.save_dir = save_dir
        self.model_name = model_name
        self.model = None
    
    def split_data(self):
        # Split the data into test and train using sklearn.model_selection.train_test_split
        raise NotImplementedError("This should be implemented by the child class")
    def train(self):
        # train the model
        raise NotImplementedError("This should be implemented by the child class")
    
    def test(self):
        # test the model
        raise NotImplementedError("This should be implemented by the child class")
    
    def predict(self):
        # predict the model
        raise NotImplementedError("This should be implemented by the child class")
    
    def save(self):
        # Save the model to a file using .pkl serialization or some other method, which is dependent on the library
        if self.model is None:
            logging.error("No model exists yet")
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
    
    def plot(self):
        # Plot the model
        raise NotImplementedError("This should be implemented by the child class")
    
    def report(self):
        # Report the model's performance
        raise NotImplementedError("This should be implemented by the child class")
    
    def _validate_hyperparameters(self):
        # Validate the hyperparameters of the model
        raise NotImplementedError("This should be implemented by the child class")
