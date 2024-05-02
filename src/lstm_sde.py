from lstm_logger import logger as logging
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
from utils import timer_decorator
from plotting import plot, finalize_plot
from timeseriesmodel import TimeSeriesModel
from memory_profiler import profile
import copy

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
            # Model dictionary 
            self.lstm_sdes = {y_var : LSTMSDE(self.d_input, self.d_lstm, self.d_lat, self.num_layers, self.t_sde, self.n_sde, self.loss_fn) for y_var in self.y_vars}

            # Set the remaining hyperparameters, which require instantiation of the model
            self.optimizers = {} 
            [self._set_optimizer_defaults(y_var=y_var, model=self.lstm_sdes[y_var]) for y_var in self.y_vars]
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
        if 'window_size' not in model_params:
            logging.warning("No window size specified. Defaulting to 30")
            model_params['window_size'] = 30
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
        losses = {}
        y_pred, rmse = {}, {}
        for y_var in self.y_vars: 
            losses[y_var] = self._train(model=self.lstm_sdes[y_var],
                        dataloader=self.lstm_data[y_var]['dataloaders']['train'], 
                        optimizer=self.optimizers[y_var], 
                        n_epochs=self.model_hyperparameters['num_epochs'], 
                        loss_fn=self.loss_fn)
            y_pred[y_var], rmse[y_var] = {}, {}
            for datakey in self.lstm_data[y_var]['data']['x'].keys():
                y_pred[y_var], rmse[y_var] = self._eval(model=self.lstm_sdes[y_var], 
                   loss_fn=self.loss_fn, 
                   x_input=self.lstm_data[y_var]['tensors']['x'][datakey], 
                   y_target=self.lstm_data[y_var]['tensors']['y'][datakey])
        self.losses_over_time = losses
        self.rmse = rmse 
    
        return self._build_output_data(y_pred, self.data_dict) 
    def _build_output_data(self, out_data, data_dict):
        logging.warning('Not implemented yet, need to build the output data like the other models do')
        return self.data_dict
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
    def _eval(self, model, loss_fn, x_input, y_target):
        """Tests the model

        Args:
            model (pytorch model): The model to train
            loss_fn (pytorch loss function): The loss function to use
            x_input (pytorch tensor): The test  input
            y_target (pytorch tensor): The test targets
        Returns:
            _type_: _description_
        """    
        model.eval()
        with torch.no_grad():
            y_pred_train = model(x_input)
            y_pred_train = y_pred_train.squeeze()  # remove extra dimensions from outputs
            rmse = np.sqrt(loss_fn(y_pred_train, y_target))
        #print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
        return y_pred_train, rmse
    #@profile
    def _prefit_functions(self): 
        window_size = self.model_hyperparameters['window_size']
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
            train_data_to_prepend = self.data_dict['normalized']['train_data'].iloc[-self.model_hyperparameters['window_size']:][self.x_vars + [var]]
            test_data_to_prepend = self.data_dict['normalized']['test_data'].iloc[-self.model_hyperparameters['window_size']:][self.x_vars + [var]]

            train_data_df = pd.concat([train_data_to_prepend, self.data_dict['normalized']['train_data']]).sort_values(by='Date')
            train_data = np.array(train_data_df[var]).reshape(-1, 1)
            test_data = np.array(self.data_dict['normalized']['test_data'][var]).reshape(-1, 1)
            model_dict = {} 
            data = {x: {} for x in ['x', 'y']}
            tensors = {x: {} for x in ['x', 'y']}
            dataloaders = {}

            x, y  = LSTMSDE_to_train._create_dataset(train_data, window_size) 
            x_torch = torch.from_numpy(x)
            y_torch = torch.from_numpy(y)
            dl = DataLoader(TensorDataset(x_torch, y_torch), batch_size=batch_size, shuffle=shuffle)
            data['x']['train'], data['y']['train'] = x, y
            tensors['x']['train'], tensors['y']['train'] = x_torch, y_torch
            dataloaders['train'] = dl 

            x, y =   LSTMSDE_to_train._create_dataset(test_data, window_size)
            x_torch = torch.from_numpy(x)
            y_torch = torch.from_numpy(y)
            data['x']['test'], data['y']['test'] = x, y
            tensors['x']['test'], tensors['y']['test'] = x_torch, y_torch
            dl = DataLoader(TensorDataset(x_torch, y_torch), batch_size=batch_size, shuffle=shuffle)
            dataloaders['test'] = dl

            for eval_filter in self.evaluation_filters:
                eval_data_df = pd.concat([test_data_to_prepend, self.data_dict['normalized'][eval_filter]]).sort_values(by='Date')
                eval_data = np.array(eval_data_df[var]).reshape(-1, 1)
                # Create the dataset and targets for each of the eval sets 
                eval_data, eval_targets = LSTMSDE_to_train._create_dataset(eval_data, window_size)
                data['x'][eval_filter] = eval_data
                data['y'][eval_filter] = eval_targets
                eval_data_tensor = torch.from_numpy(eval_data)
                eval_targets_tensor = torch.from_numpy(eval_targets)
                tensors['x'][eval_filter] = eval_data_tensor
                tensors['y'][eval_filter] = eval_targets_tensor
                dataloaders[eval_filter] = DataLoader(TensorDataset(eval_data_tensor, eval_targets_tensor), batch_size=batch_size, shuffle=shuffle)
            total_model_dict[var] = {'data' : data, 'tensors' : tensors, 'dataloaders' : dataloaders}
        return total_model_dict
        # cre


    @staticmethod
    def _create_dataset(dataset, time_step=1): 
            dataX, dataY = [], []
            for i in range(len(dataset)-time_step-1):
                a = dataset[i:(i+time_step), :]
                dataX.append(a)
                dataY.append(dataset[i + time_step, 0])
            return np.array(dataX), np.array(dataY)
    
    def plot(self, y_col='Close'):
        these_losses = self.losses_over_time[y_col]
        loss_df = pd.DataFrame()
        loss_df['Loss'] = these_losses
        loss_df['Epoch'] = loss_df.index
        fig = plot(df = loss_df, 
             x_col = 'Epoch', 
             y_col = 'Loss', 
             plot_type = 'line', 
             trace_name = 'Loss'
        )
        finalize_plot(fig, 
                       'Losses vs. Epoch', 
                       'train_loss', 
                       self.save_dir, 
                       save_png = self.save_png, 
                       save_html = self.save_html)
        
        super().plot()

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