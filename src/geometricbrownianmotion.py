import logging 

from model import Model
class GeometricBrownianMotion(Model):
    # Define a Geometric Brownian Motion class that inherits from the model class
    def __init__(self, model_hyperparameters, save_dir, model_name):
        super().__init__(model_hyperparameters, save_dir, model_name)
        self.model_hyperparameters = model_hyperparameters