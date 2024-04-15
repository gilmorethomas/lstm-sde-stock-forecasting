import logging 

from model import Model
import numpy as np
import numpy as np
class GeometricBrownianMotion(Model):
    # Define a Geometric Brownian Motion class that inherits from the model class
    def __init__(self, 
        data, 
        model_hyperparameters, 
        save_dir, 
        model_name, 
        x_vars, 
        y_vars, 
        test_split_filter=None, 
        train_split_filter=None, 
        evaluation_filters:list=[]
    ):
        super().__init__(data=data, 
            model_hyperparameters=model_hyperparameters, 
            save_dir=save_dir, 
            model_name=model_name,
            x_vars=x_vars,
            y_vars=y_vars,
            test_split_filter=test_split_filter,
            train_split_filter=train_split_filter,
            evaluation_filters=evaluation_filters, 
            )
        self.model_hyperparameters = model_hyperparameters

    def generate_stock_prices(self, stock_data):
        # Split the stock data into train and test sets
        # Calculate the daily returns of the train data
        train_returns = np.diff(train_data) / train_data[:-1]

        # Calculate the mean and standard deviation of the train returns
        mu = np.mean(train_returns)
        sigma = np.std(train_returns)

        # Generate the stock prices using the Euler Maruyama method
        dt = 1  # Time step
        num_steps = len(test_data)  # Number of steps in the test data
        stock_prices = np.zeros(num_steps)
        stock_prices[0] = test_data[0]  # Set the initial stock price

        for i in range(1, num_steps):
            dW = np.random.normal(0, np.sqrt(dt))  # Generate a random Wiener process increment
            drift = mu * stock_prices[i-1] * dt  # Calculate the drift term
            diffusion = sigma * stock_prices[i-1] * dW  # Calculate the diffusion term
            stock_prices[i] = stock_prices[i-1] + drift + diffusion  # Update the stock price

        return stock_prices

    def train(self):
        """Either calculate the mu and sigma or use the provided values to train the model
        """        
        if self.model_hyperparameters['calculate_mu'] or self.model_hyperparameters['calculate_sigma']:
            logging.info("Calculating mu and sigma")
            self.train_params['mu'] = {}
            self.train_params['sigma'] = {}
            for col in self.train_data[self.y_vars]:
                try:
                    self.train_params['mu'][col] = np.mean(self.train_data[col])
                    self.train_params['sigma'][col] = np.std(self.train_data[col])
                except TypeError as e:
                    logging.error(f"Could not calculate mu and sigma for {col}")
        else:
            logging.info("Using provided mu and sigma")
            self.train_params['mu'] = {}
            self.train_params['sigma'] = {}
            for col in self.train_data[self.y_vars]:
                self.train_params['mu'][col] = self.model_hyperparameters['mu']
                self.train_params['sigma'][col] = self.model_hyperparameters['sigma']
    def test(self):
        ...
    def predict(self):
        ...
    def plot(self):
        ...
    def report(self):
        ...
    def simulate_1d_gbm(self, nsteps=1000, t=1, mu=0.0001, sigma=0.02, start=1):
        steps = [ (mu - (sigma**2)/2) + np.random.randn()*sigma for i in range(nsteps) ]
        y = start*np.exp(np.cumsum(steps))
        x = [ t*i for i in range(nsteps) ]
        return x, y
    