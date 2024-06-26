from lstm_logger import logger as logging
from project_globals import DataNames as DN

from timeseriesmodel import TimeSeriesModel
import numpy as np
import pandas as pd
import copy

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
        evaluation_filters:list=[],
        save_html=True,
        save_png=True
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
            save_html=save_html,
            save_png=save_png
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
            dW = self.seed.normal(0, np.sqrt(dt))  # Generate a random Wiener process increment
            #dW = np.random.normal(0, np.sqrt(dt))  # Generate a random Wiener process increment
            drift = mu * stock_prices[i-1] * dt  # Calculate the drift term
            diffusion = sigma * stock_prices[i-1] * dW  # Calculate the diffusion term
            stock_prices[i] = stock_prices[i-1] + drift + diffusion  # Update the stock price

        return stock_prices
    def _set_hyperparameter_defaults(self):
        # Set the hyperparameters for the model
        # If the user specifies the time_steps in the model hyperparameters, use that
        # Otherwise, default to the length of the training data
        # Define the default hyperparameters for the model
        
        if 'window_size' not in self.model_hyperparameters: 
            logging.info('No time window specified in model hyperparameters, using default of last 10 days')
            self.model_hyperparameters['window_size'] = 100
        elif self.model_hyperparameters['window_size'] is None:
            window_size = self.data_dict[DN.normalized][DN.train_data].shape[0]
            logging.info(f'No time window specified in model hyperparameters, using all data, with size {window_size}')
            self.model_hyperparameters['window_size'] = window_size
        if 'dt' not in self.model_hyperparameters: 
            logging.info('No time step specified in model hyperparameters, using default of 10e-3')
            self.model_hyperparameters['dt'] = 10e-3
        if 'num_sims' not in self.model_hyperparameters: 
            logging.info('No number of simulations specified in model hyperparameters, using default of 5')
            self.model_hyperparameters['num_sims'] = 5
    
    def _calculate_mu_sigma(self):
        """Calculate the mu and sigma for the model across all y variables
        """        
        self.train_params['mu'] = {}
        self.train_params['sigma'] = {}
        if self.model_hyperparameters['calculate_mu'] or self.model_hyperparameters['calculate_sigma']:
            logging.info("Calculating mu and sigma")
            #
            for col in self.data_dict[DN.not_normalized][DN.train_data][self.y_vars]:
                try:
                    # check that the time increment is a day. If not, warn the user that their calculation may not be correct
                    if self.model_hyperparameters['dt'] != 1:
                        logging.warning("Time increment is not a day, so the calculation of mu and sigma may not be correct")
                    # Calculate daily returns
                    returns = ((self.data_dict[DN.not_normalized][DN.train_data][col] / self.data_dict[DN.not_normalized][DN.train_data][col].shift(1)) - 1)[1:]
                    self.train_params['mu'][col] = np.mean(returns[-self.model_hyperparameters['window_size']:])
                    self.train_params['sigma'][col] = np.std(returns[-self.model_hyperparameters['window_size']:])
                    logging.info(f'Calculated mu={self.train_params["mu"][col]}, sigma={self.train_params["sigma"][col]} for {col}')
                except TypeError:
                    logging.error(f"Could not calculate mu and sigma for {col}")

        else:
            logging.info("Using provided mu and sigma")
            for col in self.data_dict[DN.not_normalized][DN.train_data][self.y_vars]:
                self.train_params['mu'][col] = self.model_hyperparameters['mu']
                self.train_params['sigma'][col] = self.model_hyperparameters['sigma']

    def fit(self):
        """Either calculate the mu and sigma or use the provided values to train the model
        """   
        self._train_starting_message()
        # Set model defaults
        self._set_hyperparameter_defaults()

        # Calculate mu and sigma 
        self._calculate_mu_sigma()
        logging.info(f'Simulating GBM data for {self.model_name}') 
        logging.warning('GBM model does not properly normalize the data, since the data has not been fit before the execution of the model')

        # Simulate the training data 
        fit_data = {}
        data_dict = copy.deepcopy(self.data_dict)
        # Simulate GBM and add to the data dictionary as new columns for each seed 
        gbm_train = self._simulate_gbm_train(self.data_dict[DN.not_normalized][DN.train_data])
        gbm_test = self._simulate_gbm_train(self.data_dict[DN.not_normalized][DN.test_data])
        data_dict[DN.not_normalized][DN.train_data] = data_dict[DN.not_normalized][DN.train_data].merge(gbm_train, on='Days_since_start')
        data_dict[DN.not_normalized][DN.test_data] = data_dict[DN.not_normalized][DN.test_data].merge(gbm_test, on='Days_since_start')

        # Normalize the data
        gbm_train_norm = gbm_train.copy(deep=True)
        gbm_test_norm = gbm_test.copy(deep=True)
        self.scaler.fit(gbm_train_norm[self.model_responses[DN.raw]])
        self.scaler.fit(gbm_test_norm[self.model_responses[DN.raw]])
        gbm_train_norm[self.model_responses[DN.raw]] = self.scaler.transform(gbm_train_norm[self.model_responses[DN.raw]])
        gbm_test_norm[self.model_responses[DN.raw]] = self.scaler.transform(gbm_test_norm[self.model_responses[DN.raw]])
        data_dict[DN.normalized][DN.train_data] = data_dict[DN.normalized][DN.train_data].merge(gbm_train_norm, on='Days_since_start')
        data_dict[DN.normalized][DN.test_data] = data_dict[DN.normalized][DN.test_data].merge(gbm_test_norm, on='Days_since_start')
        
        # Simulate the evaluation data
        for eval_name in self.evaluation_data_names:
            gbm_eval = self._simulate_gbm_train(self.data_dict[DN.not_normalized][eval_name])
            data_dict[DN.not_normalized][eval_name] = data_dict[DN.not_normalized][eval_name].merge(gbm_eval, on='Days_since_start')
            # Normalize the data
            gbm_eval_norm = gbm_eval.copy(deep=True)
            self.scaler.fit(gbm_eval_norm[self.model_responses[DN.raw]])
            gbm_eval_norm[self.model_responses[DN.raw]] = self.scaler.transform(gbm_eval_norm[self.model_responses[DN.raw]])
            data_dict[DN.normalized][eval_name] = data_dict[DN.normalized][eval_name].merge(gbm_eval_norm, on='Days_since_start')
            
        super().fit(data_dict)
    
    def save(self):
        return
        # GBM model save not implemented yet 

    def _simulate_gbm_train(self, dataset):
        """Helper method that simulates GBM for the train data. This method is called in the train method

        Args:
            train_data (pd.DataFrame): Training data

        Returns:
            pd.DataFrame: Dataframe of simulated GBM data
        """       
        assert 'Days_since_start' in dataset.columns, "Days_since_start must be in dataset columns for gbm"
        nsteps = dataset['Days_since_start'].max() - dataset['Days_since_start'].min()
        # Build a train fit dataframe, which will contain one row per step
        train_data_fit = pd.DataFrame()
        scale_cols = []
        train_data_fit['Days_since_start'] = range(nsteps + 1) + dataset['Days_since_start'].min()
        # Run GBM for each y variable
        for col in dataset[self.y_vars]:
            # Use one step per day
            num_sims = self.model_hyperparameters.get('num_sims', None)
            if num_sims is None:
                logging.info("num_sims was not provided, defaulting to 1")
                num_sims=1
            _, gbm_data = _GeometricBrownianMotion.simulate_gbm(
                nsteps=nsteps,
                num_sims=num_sims,
                mu=self.train_params['mu'][col],
                sigma=self.train_params['sigma'][col],
                start=dataset[col].iloc[0],
                dt=self.model_hyperparameters['dt'])

            # Assign the simulated data to the train_data_fit dataframe, using multiple columns
            for i in range(num_sims):
                train_data_fit[f'{col}_{i}'] = gbm_data[:,i]

        return train_data_fit

    def plot(self):
        # Call the base model class plotting function 
        super().plot()
        # Add any additional custom plots 
    def report(self):
        super().report()
    
class _GeometricBrownianMotion():
    """A helper class that implements the specifics of model fitting and prediction for the Geometric Brownian Motion model.
    """    
    @staticmethod
    def simulate_gbm(nsteps=1000, num_sims=1, start=1, mu=0.0001, sigma=0.02, dt=1):
        """_summary_

        Args:
            nsteps (int, optional): Number of steps. Defaults to 1000.
            num_sims (int, optional): Number of sims. Defaults to 1.
            start (int, optional): Initial price. Defaults to 1.
            mu (float, optional): Drift coefficient. Defaults to 0.0001.
            sigma (float, optional): Volatility. Defaults to 0.02.
            dt (int, optional): Time increment. Defaults to 1.

        Returns:
            np.array: timestamps
            np.array: simulated stock prices
        """        
        # TODO pull from random state mgr
        # Create a matrix of random numbers, using the random number generator from the random state manager, using a normal distribution
        random_array = np.random.normal(0, 1, (nsteps, num_sims))
        # NEED TO UPDATE
        #self._rand_state_mgr.standard_normal((num_sims, nsteps)).T
        N = nsteps
        # Create an array to store the stock prices
        St = np.zeros((nsteps+1, num_sims))
        
        for i in range(num_sims):
            random_array_input = random_array[:,i]
            _, S = _GeometricBrownianMotion.euler_maruyama_geometric_brownian_motion(S0=start, mu=mu, sigma=sigma, dt=dt, N=N, random_array=random_array_input)
            # Store the stock prices in the St array
            St[:,i] = S
        
        # Create an array of time points
        x = [i for i in range(nsteps + 1 ) ]
        return x, St

    @staticmethod
    def euler_maruyama_geometric_brownian_motion(S0, mu, sigma, dt, N, random_array):
        """
        Simulate geometric Brownian motion using Euler-Maruyama method.

        Parameters:
            S0 (float): initial value of the process
            mu (float): expected return
            sigma (float): volatility
            dt (float): time increment size, typically days. Smaller values result in more accurate simulations, not tested
            N (int): number of time steps to simulate
            random_array (np.array): array of random numbers, of dimensions (N, num_sims)

        Returns:
            list: time points
            list: simulated process values
        """
        t = np.linspace(0, dt, N+1)
        # Create arrays to store the Wiener process and the stock price
        W = np.zeros(N+1)
        S = np.zeros(N+1)
        # Set the initial values
        S[0] = S0
        # Generate the Wiener process
        for i in range(1, N+1):
            dW   = (dt**0.5) * random_array[i-1]
            S[i] = S[i-1] * (1 + mu*dt + sigma*dW)
            W[i] = W[i-1] + dW

        return t, S
    
    @classmethod
    def load_from_previous_output(cls, class_params):# , save_dir, model_name):
        instance = super().load_from_previous_output(class_params)
        return instance