import logging 

from timeseriesmodel import TimeSeriesModel
import numpy as np
import pandas as pd

class GeometricBrownianMotion(TimeSeriesModel):
    # Define a Geometric Brownian Motion class that inherits from the model class
    def __init__(self, 
        data, 
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

        self.train_params['mu'] = {}
        self.train_params['sigma'] = {}
        if self.model_hyperparameters['calculate_mu'] or self.model_hyperparameters['calculate_sigma']:
            logging.info("Calculating mu and sigma")
            for col in self.train_data[self.y_vars]:
                try:
                    self.train_params['mu'][col] = np.mean(self.train_data[col])
                    self.train_params['sigma'][col] = np.std(self.train_data[col])
                except TypeError as e:
                    logging.error(f"Could not calculate mu and sigma for {col}")

        else:
            logging.info("Using provided mu and sigma")
            for col in self.train_data[self.y_vars]:
                self.train_params['mu'][col] = self.model_hyperparameters['mu']
                self.train_params['sigma'][col] = self.model_hyperparameters['sigma']
        self.train_data_fit = self._simulate_gbm_train()
        self.train_data_fit = pd.merge(self.train_data, self.train_data_fit,  on='Days_since_start')
    
    def _simulate_gbm_train(self):
        """Helper method that simulates GBM for the train data

        Args:
            train_data (_type_): _description_

        Returns:
            _type_: _description_
        """        
        nsteps = self.train_data['Days_since_start'].max() - self.train_data['Days_since_start'].min()
        # Build a train fit dataframe, which will contain one row per step
        train_data_fit = pd.DataFrame()
        train_data_fit['Days_since_start'] = range(nsteps + 1)
        for col in self.train_data[self.y_vars]:
            # Use one step per day
            num_sims = self.model_hyperparameters.get('num_sims', None)
            if num_sims is None:
                logging.info("num_sims was not provided, defaulting to 1")
                num_sims=1
            _, gbm_data = self.simulate_gbm_2(t_years=nsteps / 365.25, 
                                                                    nsteps=nsteps,
                                                                    num_sims=num_sims,
                                                                    mu=self.train_params['mu'][col], 
                                                                    sigma=self.train_params['sigma'][col],
                                                                    start=self.train_data[col].iloc[0])
            # Assign the simulated data to the train_data_fit dataframe, using multiple columns
            for i in range(num_sims):
                train_data_fit[f'{col}_GBM_{i}'] = gbm_data[:,i]
        return train_data_fit

    def test(self):
        ...
    def predict(self):
        ...
    def plot(self):
        ...
    def report(self):
        super().report()

    def simulate_gbm_2(self, nsteps=1000, t_years=0, num_sims=1, start=1, mu=0.0001, sigma=0.02):
        """_summary_

        Args:
            n (int, optional): Number of steps. Defaults to 1000.
            T (int, optional): Time in years. Defaults to 1.
            M (int, optional): Number of sims. Defaults to 1.
            start (int, optional): Initial price. Defaults to 1.
            mu (float, optional): Drift coefficient. Defaults to 0.0001.
            sigma (float, optional): Volatility. Defaults to 0.02.

        Returns:
            _type_: _description_
        """        
        # import matplotlib.pyplot as plt
        # # Parameters
        # # drift coefficent
        # mu = 0.1
        # # number of steps
        # n = 100
        # # time in years
        # T = 1
        # # number of sims
        # M = 100
        # # initial stock price
        # S0 = 100
        # # volatility
        # sigma = 0.3
        # calc each time step
        dt = t_years/nsteps
        # simulation using numpy arrays
        # Create a matrix of random numbers, using the random number generator from the random state manager
        St = np.exp(
            (mu - sigma ** 2 / 2) * dt
            + sigma * np.sqrt(dt) * self._rand_state_mgr.random((num_sims, nsteps)).T
        )
        # include array of 1's
        St = np.vstack([np.ones(num_sims), St])
        # multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0). 
        St = start * St.cumprod(axis=0)
        x = [ t_years*i for i in range(nsteps + 1 ) ]
        return x, St