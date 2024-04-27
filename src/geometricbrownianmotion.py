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
            logging.info('No time window specified in model hyperparameters, using all data')
            self.model_hyperparameters['window_size'] = self.train_data.shape[0]
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
            for col in self.train_data[self.y_vars]:
                try:
                    # check that the time increment is a day. If not, warn the user that their calculation may not be correct
                    if self.model_hyperparameters['dt'] != 1:
                        logging.warning("Time increment is not a day, so the calculation of mu and sigma may not be correct")
                    # Calculate daily returns
                    returns = ((self.train_data[col] / self.train_data[col].shift(1)) - 1)[1:]
                    self.train_params['mu'][col] = np.mean(returns[-self.model_hyperparameters['window_size']:])
                    self.train_params['sigma'][col] = np.std(returns[-self.model_hyperparameters['window_size']:])
                    logging.info(f'Calculated mu={self.train_params["mu"][col]}, sigma={self.train_params["sigma"][col]} for {col}')
                except TypeError as e:
                    logging.error(f"Could not calculate mu and sigma for {col}")

        else:
            logging.info("Using provided mu and sigma")
            for col in self.train_data[self.y_vars]:
                self.train_params['mu'][col] = self.model_hyperparameters['mu']
                self.train_params['sigma'][col] = self.model_hyperparameters['sigma']

    def train(self):
        """Either calculate the mu and sigma or use the provided values to train the model
        """   
        self._train_starting_message()
        # Set model defaults
        self._set_hyperparameter_defaults()

        # Calculate mu and sigma 
        self._calculate_mu_sigma()

        # Simulate the training data 
        train_data_fit = self._simulate_gbm_train()

        # Call the base model class train function
        super().train(train_data_fit)
    
    def save(self):
        return
        # GBM model save not implemented yet 

    def _simulate_gbm_train(self):
        """Helper method that simulates GBM for the train data. This method is called in the train method

        Args:
            train_data (pd.DataFrame): Training data

        Returns:
            _type_: Dataframe of simulated GBM data
        """       
        logging.info(f'Simulating GBM train data for {self.model_name}') 
        nsteps = self.train_data['Days_since_start'].max() - self.train_data['Days_since_start'].min()
        # Build a train fit dataframe, which will contain one row per step
        train_data_fit = pd.DataFrame()
        scale_cols = []
        train_data_fit['Days_since_start'] = range(nsteps + 1)
        # Run GBM for each y variable
        for col in self.train_data[self.y_vars]:
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
                start=self.train_data[col].iloc[0],
                dt=self.model_hyperparameters['dt'])

            # Assign the simulated data to the train_data_fit dataframe, using multiple columns
            for i in range(num_sims):
                train_data_fit[f'{col}_{self.model_name}_{i}'] = gbm_data[:,i]

        return train_data_fit

    def test(self):
        ...
    def predict(self):
        ...
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
    
    @staticmethod
    def simulate_gbm_3(nsteps=1000, t_years=0, num_sims=1, start=1, mu=0.0001, sigma=0.02, dt=0.01):
        return _GeometricBrownianMotion._simulate_gbm_3(T=nsteps, scen_size=num_sims, start_price=start, mu=mu, sigma=sigma, dt=dt)
    @staticmethod
    def _simulate_gbm_3(start_price, T, dt, scen_size, mu=None, sigma=None, daily_data=None):
        """_summary_

        Args:
            daily_returns_absolute (_type_): Daily stock data (not normalized)
            start_price (_type_): _description_
            T (_type_): _description_
            dt (_type_): _description_
            scen_size (_type_): _description_
        """

        daily_returns = ((daily_data / daily_data.shift(1)) - 1)[1:] if daily_data is not None else None
        # Calculate mu and sigma if not provided using the daily returns data 
        mu = np.mean(daily_returns) if mu is None else mu
        sigma = np.std(daily_returns) if sigma is None else sigma

        So = start_price 
        N = T / dt
        t = np.arange(1, int(N) + 1)
        # Create a dictionary of Brownian increments
        br = {f'{scen}': np.random.normal(0, 1, int(N)) for scen in range(1, scen_size + 1)}
        # Create a dictionary of Brownian paths
        W = {str(scen): br[str(scen)].cumsum() for scen in range(1, scen_size + 1)}

        # Calculating drift and diffusion components
        drift = (mu - 0.5 * sigma ** 2) * t
        diffusion = {str(scen): sigma * W[str(scen)] for scen in range(1, scen_size + 1)}

        # Making the predictions
        S = np.array([So * np.exp(drift + diffusion[str(scen)]) for scen in range(1, scen_size + 1)])
        S = np.hstack((np.array([[So] for scen in range(scen_size)]), S))  # add So to the beginning series
        # Calculate the max and min of the simulations
        S_max = [S[:, i].max() for i in range(0, int(N))]
        S_min = [S[:, i].min() for i in range(0, int(N))]
        # Calculate the predicted stock price, which is the average of the max and min
        S_pred = .5 * np.array(S_max) + .5 * np.array(S_min)
        #final_df.index = test_set.index
        #mse = 1/len(final_df) * np.sum((final_df['pred'] - final_df['real']) ** 2)
        return t, S_pred
    
if __name__=='__main__':
        # main variables
    # stock_name    :   ticker symbol from yahoo finance
    # start_date    :   start date to download prices
    # end_date      :   end date to download prices
    # pred_end_date :   date until which you want to predict price
    # scen_size     :   different possible scenarios
    import yfinance as yf
    
    stock_name = 'AAPL'
    start_date = '2010-01-01'
    end_date = '2020-10-31'
    pred_end_date = '2020-12-31'
    scen_size = 25
    prices = yf.download(tickers=stock_name, start=start_date, end=pred_end_date)['Adj Close']
    train_set = prices.loc[:end_date] # DON'T NORMALIZE!! / prices.loc[:end_date].max()
    daily_returns = ((train_set / train_set.shift(1)) - 1)[1:]

    # Geometric Brownian Motion (GBM)

    # Parameter Definitions

    # So    :   initial stock price
    # dt    :   time increment -> a day in our case
    # T     :   length of the prediction time horizon(how many time points to predict, same unit with dt(days))
    # N     :   number of time points in prediction the time horizon -> T/dt
    # t     :   array for time points in the prediction time horizon [1, 2, 3, .. , N]
    # mu    :   mean of historical daily returns
    # sigma :   standard deviation of historical daily returns
    # b     :   array for brownian increments
    # W     :   array for brownian path


    # Parameter Assignments
    So = train_set[-1]
    dt = 1  # day   # User input
    n_of_wkdays = pd.date_range(start=pd.to_datetime(end_date,
                                                    format="%Y-%m-%d") + pd.Timedelta('1 days'),
                                end=pd.to_datetime(pred_end_date,
                                                format="%Y-%m-%d")).to_series().map(lambda x: 1 if x.isoweekday() in range(1, 6) else 0).sum()
                                                #format="%Y-%m-%d")).to_series().map(lambda x: 1 if x.isoweekday() in range(1, 6) else 1).sum()
    T = n_of_wkdays
    N = T / dt
    t = np.arange(1, int(N) + 1)
    mu = np.mean(daily_returns)
    sigma = np.std(daily_returns)
    _, sims = GeometricBrownianMotion._simulate_gbm_3(daily_data = train_set, start_price=So, T = T, dt = dt, scen_size=scen_size)
    # def simulate_gbm_3(daily_data, start_price, T, dt, scen_size):
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.suptitle('Monte-Carlo Simulation: ' + str(scen_size) + ' simulations', fontsize=20)
    plt.title('Asset considered: {}'.format(stock_name))
    plt.ylabel('USD Price')
    plt.xlabel('Prediction Days')
    print('Training')
    for i in range(scen_size):
        plt.plot(pd.date_range(start=train_set.index[-1],
                            end=pred_end_date,
                            freq='D').map(lambda x: x if x.isoweekday() in range(1, 6) else np.nan).dropna(), S[i, :])
        
