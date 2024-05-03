from lstm_logger import logger as logging
from project_globals import DataNames as DN
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from lstm_logger import logger as logging
import multiprocessing
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from utils import timer_decorator, drop_nans_from_data_dict
from plotting import plot, finalize_plot
from timeseriesmodel import TimeSeriesModel
from memory_profiler import profile
import copy
from itertools import product

class SDEBlock(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """    
    def __init__(self, d_lat, n_sde, t_sde):
        super(SDEBlock, self).__init__()
        self.d_lat = d_lat
        self.n_sde = n_sde
        self.t_sde = t_sde 
        dt = t_sde / n_sde # Discretization step
        self.dt = torch.tensor(dt, dtype=torch.float32)  # convert dt to a tensor

        # 4b.) Define the drift and diffusion networks (f* and g*). 
            # Each consists of two-layer neural network where first maps R^d_lat to R^2*d_lat and second maps back to R^D_Lat, using a tanh activation function
        self.drift = nn.Sequential(
            nn.Linear(d_lat, 2*d_lat), 
            nn.Tanh(),
            nn.Linear(2*d_lat, d_lat)
        )
        self.diffusion = nn.Sequential(
            nn.Linear(d_lat, 2*d_lat),
            nn.Tanh(),
            nn.Linear(2*d_lat, d_lat)
        )
    def forward(self, z0):
        # Generate n_sde latent variables paths of the form {z_0, ..., z_N_sde} where z(0) = z0
        z = z0
        for _ in range(self.n_sde):
            f = self.drift(z)
            g = self.diffusion(z)
            dW = torch.sqrt(self.dt) * torch.randn_like(z)
            z = z + f*self.dt + g*dW  # Euler-Maruyama method
        return z
    
class LSTMSDE(nn.Module):

    """ The main functionality of the LSTM SDE class is to generate N 
    latent variable paths of the form {z_0, ..., z_N__sde} where z(0) = z_0 and N_sde 
    deontes number of latent variables in each latent variable path

    Steps: 
    1.) Observed time-sequential data of dimension Dobs = 1 is fed to single-latyer LSTM netowrk of dimension D_lstm
        1a.) this implies that the observed data mapped from R^D_obs to R^D_LSTM inside the network 
    2.) The mapped observed data of R^D_LSTM is mapped to the initial latent variable z_0 { R^D_Lat through a linear layer mapping R^D_LSTM to R^D_Lat
    3.) The initial latent variable is fed to latent variable neural SDE framework (SDE block) 
    4.) The SDE block generates N latent variable paths of the form {z_0, ..., z_N__sde} where z(0) = z_0 and N_sde deontes number of latent variables in each latent variable path 
        4a.) This is done using the EM method, with drift and diffusion networks f* and g* 
        4b.) The dirft and diffusion networks consist of two-layer neural netws where the first layer maps to R^2*D_Lat and the second layer maps back to R^D_Lat
        4c.) The activation function for f* and g* is the tanh-function 
    
    """
    def __init__(self, input_size, lstm_size, output_size, num_layers, t_sde, n_sde, loss_fn):
        super(LSTMSDE, self).__init__()
        self.hidden_size = lstm_size
        self.num_layers = num_layers
        self.loss_fn = loss_fn
        # Define the layers of the network

        # Define the LSTM layer. Typically want a single layer LSTM network
        self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers = self.num_layers, batch_first=True)
        # Define the linear layer that maps the LSTM output to the initial latent variable
        #self.fc = nn.Linear(hidden_size, output_size)
        self.fc = nn.Linear(self.hidden_size, output_size)

        # Define the SDE block, which will take the initial latent variable as input and generate N latent variable paths
        self.sde_block = SDEBlock(output_size, n_sde, t_sde)

    def forward(self, x):
        """Define how the data will pass through the network

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """        
        # Step 1: Observed time-sequential data of dimension D_obs = input_size is fed to single-layer LSTM network of dimension D_lstm = hidden_size
        # Ensure that x has the correct size (batch_size, seq_len, input_size)

        # Data mapped from input size to hidden size 
        # Create the tensors filled with zeros, where we have the (number of layers in LSTM, the batch size, and the size of the hidden state)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) # initial hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) # initial cell state

        # Forward pass through the LSTM layer
        # This should be of shape (batch_size, seq_len, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # Step 2: The mapped observed data of R^D_LSTM is mapped to the initial latent 
        # variable z_0 { R^D_Lat through a linear layer mapping R^D_LSTM to R^D_Lat
        z0 = self.fc(out[:, -1, :])

        # Step 3: The initial latent variable is fed to latent variable neural SDE framework (SDE block)
        # The SDE block is not implemented in this code, but you would pass z0 to it here
        out = self.sde_block(z0)

        return out

class LSTMSDE_to_train(TimeSeriesModel):
    """A wrapper class for the LSTMSDE class. This is intended to interact with the rest of the library, while divorcing the 
    actual functionality from the preparation and interfacing with data and other models.
    """    
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
        save_png=True):

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
            self.model_hyperparameters = self._set_hyperparameter_defaults(self.model_hyperparameters)
            self._unpack_model_params(self.model_hyperparameters)
            self.save_dir = save_dir
            # Model dictionary. Primary key is y var, secondary is sed number
            self.lstm_sdes = {y_var : {seed_num: LSTMSDE(self.d_input, self.d_lstm, self.d_lat, self.num_layers, self.t_sde, self.n_sde, self.loss_fn) for seed_num in range(self.num_sims)} for y_var in self.y_vars}
            # Set the remaining hyperparameters, which require instantiation of the model
            self.optimizers = {} 
            y_var_seed_product = product(self.y_vars, range(self.num_sims))
            [self._set_optimizer_defaults(
                y_var=y_var, 
                model=self.lstm_sdes[y_var][seed_num]) 
                for y_var, seed_num in y_var_seed_product]
    def _set_hyperparameter_defaults(self, model_params):
        if 'loss_fn' not in model_params:
            logging.warning(f"No loss function specified. Defaulting to MSELoss")
            self.loss_fn = torch.nn.MSELoss()
        elif model_params['loss_fn'] == 'MSELoss':
            self.loss_fn = torch.nn.MSELoss()
        elif model_params['loss_fn'] == 'NLLLoss':
            self.loss_fn = torch.nn.NLLLoss()
        else:
            logging.warning(f"Loss function {model_params['loss_fn']} not recognized. Defaulting to MSELoss")
            self.loss_fn = torch.nn.MSELoss()
        if 'learning_rate' not in model_params:
            logging.warning("No learning rate specified. Defaulting to 0.01")
            model_params['learning_rate'] = 0.01
        if 'num_epochs' not in model_params:
            logging.warning("No number of epochs specified. Defaulting to 1000")
            model_params['num_epochs'] = 1000
        if 'num_layers' not in model_params:
            logging.warning("No number of layers specified. Defaulting to 1")
            model_params['num_layers'] = 1
        if 'd_lstm' not in model_params:
            logging.warning("No LSTM dimension specified. Defaulting to 64")
            model_params['d_lstm'] = 64
        if 'd_lat' not in model_params:
            logging.warning("No latent dimension specified. Defaulting to 1")
            model_params['d_lat'] = 1
        if 'd_input' not in model_params:
            logging.warning("No input dimension specified. Defaulting to 1")
            model_params['d_input'] = 1
        if 'd_hidden' not in model_params:
            logging.warning("No hidden dimension specified. Defaulting to 1")
            model_params['d_hidden'] = 1
        if 'N' not in model_params:
            logging.warning("No number of latent variable paths specified. Defaulting to 50")
            model_params['N'] = 50
        if 't_sde' not in model_params:
            logging.warning("No SDE time step specified. Defaulting to 1")
            model_params['t_sde'] = 1
        if 'n_sde' not in model_params:
            logging.warning("No number of latent variables in each latent variable path specified. Defaulting to 100")
            model_params['n_sde'] = 100
        if 'num_sims' not in model_params:
            logging.warning("No number of simulations specified. Defaulting to 5")
            model_params['num_sims'] = 1
        if 'batch_size' not in model_params:
            logging.warning("No batch size specified. Defaulting to 64")
            model_params['batch_size'] = 64
        if 'shuffle' not in model_params:
            logging.warning("No shuffle specified. Defaulting to True")
            model_params['shuffle'] = True
        if 'time_steps' not in model_params:
            logging.warning("No window size specified. Defaulting to 30")
            model_params['time_steps'] = 30
        return model_params 
    def _set_optimizer_defaults(self, model, y_var):
        """Sets default values for hyperparameters of optimizer.
        Note that this is in a separate function than the _set_hyperparameter_defaults function
        because the optimizers require that the model has been created. 

        Args:
            model_params (dict): model hyperparameters 
        """        
        model_params = self.model_hyperparameters

        if 'optimizer' not in model_params:
            logging.warning(f"No optimizer specified. Defaulting to Adam")
            optimizer = torch.optim.Adam(model.parameters(), lr=model_params['learning_rate'])
            model_params['optimizer'] = ['adam']
        elif model_params['optimizer'].lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=model_params['learning_rate'])
            model_params['optimizer'] = ['adam']
        elif model_params['optimizer'].lower() == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=model_params['learning_rate'])
            model_params['optimizer'] = ['sgd']
        else:
            logging.warning(f"Optimizer {model_params['optimizer']} not recognized. Defaulting to Adam")
            optimizer = torch.optim.Adam(model.parameters(), lr=model_params['learning_rate'])
            model_params['optimizer'] = ['adam']
        model_params['optimizer'] = self.optimizer
        self.optimizers[y_var] = optimizer
        self.model_hyperparameters = model_params
        


    def forward(self, x):
        """Wrapper method used to call the lstm_sde forward method

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """        
        return self.lstm_sde(x)
    #@profile
    def fit(self): #, dataloader, loss_fn, optimizer, n_epochs, x_inputs: dict, y_targets: dict):
        """Calls the train and fit methods of the model 

        Returns:
            _type_: _description_
        """        
        # Wrapper data structure that will be used in addition to things from the Model Class
        self.lstm_data = self._prefit_functions() 
        # iterate over each of the dataloaders 
        
        # Train on the train data 
        losses = {y_var: {seed_num: [] for seed_num in range(self.num_sims)} for y_var in self.y_vars}
        y_pred = copy.deepcopy(losses)
        rmse = copy.deepcopy(losses)
        y_vars_seeds = product(self.
        y_vars, range(self.num_sims))
        for y_var, seed_num in y_vars_seeds: 
            losses[y_var][seed_num] = self._train(model=self.lstm_sdes[y_var][seed_num],
                        dataloader=self.lstm_data[y_var][DN.dataloaders][DN.train_data], 
                        optimizer=self.optimizers[y_var], 
                        n_epochs=self.model_hyperparameters['num_epochs'], 
                        loss_fn=self.loss_fn)
            y_pred[y_var][seed_num], rmse[y_var][seed_num] = {}, {}
            # evaluate train
            y_pred[y_var][seed_num][DN.train_data], rmse[y_var][seed_num][DN.train_data] = self._eval(
                eval_type = DN.train_data, model=self.lstm_sdes[y_var][seed_num], 
                loss_fn=self.loss_fn, 
                x_input=self.lstm_data[y_var][DN.tensors][DN.x][DN.train_data], 
                y_target=self.lstm_data[y_var][DN.tensors][DN.y][DN.train_data])
            # evaluate test 
            y_pred[y_var][seed_num][DN.test_data], rmse[y_var][seed_num][DN.test_data] = self._eval(
                eval_type = DN.test_data, model=self.lstm_sdes[y_var][seed_num], 
                loss_fn=self.loss_fn, 
                x_input=self.lstm_data[y_var][DN.tensors][DN.x][DN.test_data], 
                y_target=self.lstm_data[y_var][DN.tensors][DN.y][DN.test_data])
            # evaluate evaluation_periods 
            for eval_name in self.evaluation_data_names:
                y_pred[y_var][seed_num][eval_name], rmse[y_var][seed_num][DN.train_data] = self._eval(
                    eval_type = DN.evaluation, 
                    model=self.lstm_sdes[y_var][seed_num], 
                    loss_fn=self.loss_fn, 
                    x_input=self.lstm_data[y_var][DN.tensors][DN.x][eval_name], 
                    y_target=self.lstm_data[y_var][DN.tensors][DN.y][eval_name])
                
        self.losses_over_time = losses
        self.rmse = rmse 
        data_dict = self._build_output_data(y_pred, copy.deepcopy(self.data_dict[DN.not_normalized]), copy.deepcopy(self.data_dict[DN.normalized])) 
        super().fit(data_dict)
    def _build_output_data(self, out_data, data_dict_not_norm, data_dict):
        """Builds the output data. Relies on the fact that the model produces normalized output data.

        Note that this is very similar to the LSTM _build_output_data function.

        TODO align this with LSTM class _build_output_data function

        Args:
            out_data (_type_): _description_
            data_dict (_type_): _description_

        Returns:
            _type_: _description_
        """        
        # Build empty nans to prepend. Add 1, think thisi s because the data is not counting the prediction for the next day? TODO look into that
        nan_array = np.array([[np.nan] * (self.model_hyperparameters['time_steps'])]).T
        # Build the empty structure  
        outputs_norm = {y_var: {seed_num: [] for seed_num in range(self.num_sims)} for y_var in self.y_vars}
        outputs = copy.deepcopy(outputs_norm)
        for y_var, seed_num in product(self.y_vars, range(self.num_sims)):
            # Pandas concatatenate the data into the
            this_data = out_data[y_var][seed_num]
            # Need to reshape the numpy array to have shape (n, 1)
            train_data_to_insert = this_data[DN.train_data].numpy().reshape(-1, 1)
            test_data_to_insert = np.concatenate([nan_array, this_data[DN.test_data].numpy().reshape(-1, 1)])
            data_dict[DN.train_data][f'{y_var}_{self.model_name}_{seed_num}'] = train_data_to_insert
            data_dict[DN.test_data][f'{y_var}_{self.model_name}_{seed_num}'] = test_data_to_insert
            data_dict_not_norm[DN.test_data][f'{y_var}_{self.model_name}_{seed_num}'] = self.scaler.inverse_transform(test_data_to_insert)
            data_dict_not_norm[DN.train_data][f'{y_var}_{self.model_name}_{seed_num}'] = self.scaler.inverse_transform(train_data_to_insert)
            # drop nans to account for windows for rollback data

            for eval_filter in self.evaluation_data_names:
                eval_data_to_insert = this_data[eval_filter].numpy().reshape(-1, 1)
                data_dict[eval_filter][f'{y_var}_{self.model_name}_{seed_num}'] = eval_data_to_insert
                data_dict_not_norm[eval_filter][f'{y_var}_{self.model_name}_{seed_num}'] = self.scaler.inverse_transform(eval_data_to_insert)
        data_dict = drop_nans_from_data_dict(data_dict, self, self.fit)
        data_dict_not_norm = drop_nans_from_data_dict(data_dict_not_norm, self, self.fit)
        return {DN.normalized : data_dict, DN.not_normalized : data_dict_not_norm}
    @timer_decorator
    def _train(self, model, dataloader, optimizer, n_epochs, loss_fn):
        """Trains the model 

        Args:
            dataloader (pytorch DataLoader): Dataloader containing the x and y data for given batch size
            model (pytorch model): The model to train
            loss_fn (pytorch loss function): The loss function to use
            optimizer (pytorch optimizer):  The optimizer to use
            n_epochs (int, optional): Number of epochs to train for. Defaults to 1000.
        """    
        losses_over_time = []
        mse_over_time = []
        for epoch in range(n_epochs):
            logging.debug(f'Training for {epoch=}')
            model.train()
            epoch_losses = []
            for X_batch, y_batch in dataloader:
                y_pred = model(X_batch)
                y_pred = y_pred.squeeze()  # remove extra dimensions from outputs
                loss = model.loss_fn(y_pred, y_batch)
                epoch_losses.append(np.sqrt(loss.item())) # not sure if sqrt here and on test makes sense
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_loss = np.mean(epoch_losses)
            # mse_over_time.append() May ultimately want to pull this in. zsince our loss function is mse, it should be the same thing I think ?
            logging.info(f'Epoch {epoch+1}/{n_epochs}, Mean Loss: {np.mean(epoch_losses)}')
            losses_over_time.append(np.mean(epoch_losses))
        return losses_over_time
    def save(self):
        logging.info('LSTM SDE save not implemented')   
    def _eval(self, eval_type, model, loss_fn, x_input, y_target):
        """Tests the model

        Args:
            model (pytorch model): The model to train
            loss_fn (pytorch loss function): The loss function to use
            x_input (pytorch tensor): The test  input
            y_target (pytorch tensor): The test targets
        Returns:
            _type_: _description_
        """    
        #model.eval()
        def predict_future_steps(model, initial_input_data, y_target, loss_fn):
            """Predicts n future steps using an autoregressive approach.

            Args:
                model: The trained model.
                initial_input_data: The input data to start the predictions.
                n_steps: The number of future steps to predict.

            Returns:
                A list of predictions.
            """
            input_data = initial_input_data.clone() 
            predictions = []
            n_steps = len(y_target)

            for _ in range(n_steps):
                # Use the model to predict the next step
                prediction = model.predict(input_data)
                predictions.append(prediction)

                # Append the prediction to the input data and remove the oldest value
                input_data = np.append(input_data[1:], prediction)
                
            with torch.no_grad():
                rmse = np.sqrt(loss_fn(prediction, y_target))
            
            return predictions, rmse

        if eval_type == 'evaluation': 
            y_pred_train, rmse = predict_future_steps(model, x_input, y_target, loss_fn)
        else: 
            with torch.no_grad():
                y_pred_train = model(x_input)
                y_pred_train = y_pred_train.squeeze()  # remove extra dimensions from outputs
                rmse = np.sqrt(loss_fn(y_pred_train, y_target))
        return y_pred_train, rmse
    

    #@profile
    def _prefit_functions(self): 
        """_summary_

        Returns:
            dict: Dictionary with all the data needed for the model. Formatted as follows

            {
                y_var: {
                    DN.data: {
                        DN.x: {
                            DN.train_data: np.array,
                            DN.test_data: np.array,
                            evaluation_filter1: np.array
                            evaluation_filter2: np.array
                        },
                        DN.y: {
                            DN.train_data: np.array,
                            DN.test_data: np.array,
                            evaluation_filter1: np.array
                            evaluation_filter2: np.array
                        }
                    },
                    DN.tensors: {
                        DN.x: {
                            DN.train_data: torch.tensor,
                            DN.test_data: torch.tensor,
                            evaluation_filter1: torch.tensor
                            evaluation_filter2: torch.tensor
                        },
                        DN.y: {
                            DN.train_data: torch.tensor,
                            DN.test_data: torch.tensor,
                            evaluation_filter1: torch.tensor
                            evaluation_filter2: torch.tensor
                        }
                    },
                    DN.dataloaders: {
                        DN.train_data: torch DataLoader,
                        DN.test_data: torch DataLoader,
                        evaluation_filter1: torch DataLoader,
                        evaluation_filter2: torch DataLoader

                    }
                }
        """        
        window_size = self.model_hyperparameters['time_steps']
        batch_size = self.model_hyperparameters['batch_size']
        shuffle = self.model_hyperparameters['shuffle']
        total_model_dict = {}
        if len(self.y_vars) > 1: 
            logging.warning('Multiple y variables detected, undefined behavior')
        for var in self.y_vars: 
            # Not sure why the reshape is required.. all this is doing is creating a rolling window of data. Should implement this differently 
            # (TODO) WARNING: THE REASON FOR A LOT OF THE MULTI-LINE ASSIGNMENTS IS BECAUSE THE CODE CRASHES WITH A RESOURCE_TRACKER_WARNING. NEED TO FIX 
            # Create a dictionary of tensors to pass to the model
            # Create PyTorch data loaders for the train and test data
            
            # create prepended data to account for time windowing
            train_data_to_prepend = self.data_dict[DN.normalized][DN.train_data].iloc[-self.model_hyperparameters['time_steps']:][self.x_vars + [var]]
            test_data_to_prepend = self.data_dict[DN.normalized][DN.test_data].iloc[-self.model_hyperparameters['time_steps']:][self.x_vars + [var]]

            train_data_df = pd.concat([train_data_to_prepend, self.data_dict[DN.normalized][DN.train_data]]).sort_values(by='Date')
            train_data = np.array(train_data_df[var]).reshape(-1, 1)
            test_data = np.array(self.data_dict[DN.normalized][DN.test_data][var]).reshape(-1, 1)
            model_dict = {} 
            data = {x: {} for x in [DN.x, DN.y]}
            tensors = {x: {} for x in [DN.x, DN.y]}
            dataloaders = {}

            x, y  = LSTMSDE_to_train._create_dataset(train_data, window_size) 
            x_torch = torch.from_numpy(x)
            y_torch = torch.from_numpy(y)
            dl = DataLoader(TensorDataset(x_torch, y_torch), batch_size=batch_size, shuffle=shuffle)
            data[DN.x][DN.train_data], data[DN.y][DN.train_data] = x, y
            tensors[DN.x][DN.train_data], tensors[DN.y][DN.train_data] = x_torch, y_torch
            dataloaders[DN.train_data] = dl 

            x, y =   LSTMSDE_to_train._create_dataset(test_data, window_size)
            x_torch = torch.from_numpy(x)
            y_torch = torch.from_numpy(y)
            data[DN.x][DN.test_data], data[DN.y][DN.test_data] = x, y
            tensors[DN.x][DN.test_data], tensors[DN.y][DN.test_data] = x_torch, y_torch
            dl = DataLoader(TensorDataset(x_torch, y_torch), batch_size=batch_size, shuffle=shuffle)
            dataloaders[DN.test_data] = dl

            for eval_filter in self.evaluation_filters:
                eval_data_df = pd.concat([test_data_to_prepend, self.data_dict[DN.normalized][eval_filter]]).sort_values(by='Date')
                eval_data = np.array(eval_data_df[var]).reshape(-1, 1)
                # Create the dataset and targets for each of the eval sets 
                eval_data, eval_targets = LSTMSDE_to_train._create_dataset(eval_data, window_size)
                data[DN.x][eval_filter] = eval_data
                data[DN.y][eval_filter] = eval_targets
                eval_data_tensor = torch.from_numpy(eval_data)
                eval_targets_tensor = torch.from_numpy(eval_targets)
                tensors[DN.x][eval_filter] = eval_data_tensor
                tensors[DN.y][eval_filter] = eval_targets_tensor
                dataloaders[eval_filter] = DataLoader(TensorDataset(eval_data_tensor, eval_targets_tensor), batch_size=batch_size, shuffle=shuffle)
            total_model_dict[var] = {'data' : data, DN.tensors : tensors, DN.dataloaders : dataloaders}
        return total_model_dict
        # cre


    @staticmethod
    def _create_dataset(dataset, time_step=1): 
            dataX, dataY = [], []
            for i in range(len(dataset)-time_step):
                a = dataset[i:(i+time_step), :]
                dataX.append(a)
                dataY.append(dataset[i + time_step, 0])
            return np.array(dataX), np.array(dataY)
    
    def plot(self, y_col='Close'):
        loss_df = pd.DataFrame()
        for y_var, seed_num in product(self.y_vars, range(self.num_sims)):
            these_losses = self.losses_over_time[y_col][seed_num]
            loss_df[f'{y_var}_{seed_num}'] = these_losses
            loss_df['Epoch'] = loss_df.index

            fig = plot(df = loss_df, 
                x_col = 'Epoch', 
                y_col = f'{y_var}_{seed_num}',
                plot_type = 'line', 
                trace_name = f'{y_var}_{seed_num}'
            )
        finalize_plot(fig, 
                       'Losses vs. Epoch', 
                       'train_loss', 
                       self.save_dir, 
                       save_png = self.save_png, 
                       save_html = self.save_html)
        
        super().plot()

    def load_from_previous_output(cls, save_dir, model_name):
        super().load_from_previous_output(save_dir, model_name)

if __name__ == "__main__":
    # Set manual seed for reproducibility

    
    torch.manual_seed(0)

    logging.getLogger().setLevel(logging.INFO)

    df = pd.read_csv(r"/Users/thomasgilmore/Documents/40_Software/lstm-sde-stock-forecasting/input/00_raw/AAPL/AAPL.csv").reset_index()
    # Scale the data
    df = df['Close']
    scaler = MinMaxScaler()
    df2 = scaler.fit_transform(np.array(df).reshape(-1, 1))

    
    sequence_length = 10 # number of days to look back when making a prediction. Note that this ultimately should be 1... The model should do all this internally and we provide a 1-D input
    # NOTE THIS MUST MATCH 
    # Split data into train and test sets
    train_size = int(len(df2) * 0.8)
    test_size = int(len(df2) * 0.15)
    eval_size = len(df2) - train_size - test_size
    train_data, test_data, eval_data = df2[0:train_size,:], df2[train_size:train_size+test_size,:], df2[train_size+test_size:len(df2),:1]
    


    print('Finished Training')



    # # Calculate confidences and accuracies for each bin
    # fraction_of_positives, mean_predicted_value = calibration_curve(test_targets, test_predictions, n_bins=10)

    # # Plot calibration curve
    # plt.plot(mean_predicted_value, fraction_of_positives, "s-")
    # plt.plot([0, 1], [0, 1], 'k--')  # Plot perfect calibration line
    # plt.xlabel('Mean predicted value')
    # plt.ylabel('Fraction of positives')
    # plt.title('Calibration Curve')
    # plt.show()