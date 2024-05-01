import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from lstm_logger import logger as logging
import multiprocessing
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from utils import timer_decorator

class SDEBlock(nn.Module):
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
    def __init__(self, input_size, lstm_size, output_size, num_layers, t_sde, n_sde):
        super(LSTMSDE, self).__init__()
        self.hidden_size = lstm_size
        self.num_layers = num_layers
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

class LSTMSDE_to_train(LSTMSDE):
    """A class extending the LSTMSDE class to include a fit method. 
    This class does not actually implement any of the model, but rather just calls the train and test methods of the model 
    and is the class that should be used to train the model of type LSTMSDE

    Args:
        LSTMSDE (_type_): _description_
    """    
    def __init__(self, input_size, lstm_size, output_size, num_layers, t_sde, n_sde):
        super(LSTMSDE_to_train, self).__init__(input_size, lstm_size, output_size, num_layers, t_sde, n_sde)
        self.lstm_sde = LSTMSDE(input_size, lstm_size, output_size, num_layers, t_sde, n_sde)
    def forward(self, x):
        """Wrapper method used to call the lstm_sde forward method

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """        
        return self.lstm_sde(x)
    def fit(self, dataloader, loss_fn, optimizer, n_epochs, x_inputs: dict, y_targets: dict):
        """Calls the train and fit methods of the model 

        Args:
            model (_type_): _description_
            loss_fn (_type_): _description_
            optimizer (_type_): _description_
            n_epochs (int, optional): _description_. Defaults to 1000.
            x_targets (dict): Dictionary of x targets, with keys being the target names and values being the target values
            y_targets (dict): Dictionary of y targets, with keys being the target names and values being the target values

        Returns:
            _type_: _description_
        """        
        losses_over_time = self._train(dataloader, loss_fn, optimizer, n_epochs)
        predictions_dict = {}
        rmse_dict = {}
        for x_input_name, x_input in x_inputs.items():
            # assuming that y targets has the same keys 
            y_target = y_targets[x_input_name]
            if not isinstance(x_input, torch.Tensor):
                x_input = torch.tensor(x_input, dtype=torch.float32)
                y_target = torch.tensor(y_target, dtype=torch.float32)
            predictions_dict[x_input_name], rmse_dict[x_input_name] = self._eval(model, loss_fn, x_input, y_target)
        return predictions_dict, rmse_dict, losses_over_time
    
    @timer_decorator
    def _train(self, dataloader, loss_fn, optimizer, n_epochs=1000):
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
            self.train()
            epoch_losses = []
            for X_batch, y_batch in dataloader:
                y_pred = self(X_batch)
                y_pred = y_pred.squeeze()  # remove extra dimensions from outputs
                loss = loss_fn(y_pred, y_batch)
                epoch_losses.append(np.sqrt(loss.item())) # not sure if sqrt here and on test makes sense
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_loss = np.mean(epoch_losses)
            # mse_over_time.append() May ultimately want to pull this in. zsince our loss function is mse, it should be the same thing I think ?
            logging.info(f'Epoch {epoch+1}/{n_epochs}, Mean Loss: {np.mean(epoch_losses)}')
            losses_over_time.append(np.mean(epoch_losses))
        return losses_over_time

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

# Function to create dataset
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), :]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


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
    train_size = int(len(df2) * 0.95)
    test_size = len(df2) - train_size
    train_data, test_data = df2[0:train_size,:], df2[train_size:len(df2),:1]
    # Create the train and test data and targets
    train_data, train_targets = create_dataset(train_data, sequence_length)
    test_data, test_targets = create_dataset(test_data, sequence_length)

    # Convert the train and test data to PyTorch tensors
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    train_targets_tensor = torch.tensor(train_targets, dtype=torch.float32)
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
    test_targets_tensor = torch.tensor(test_targets, dtype=torch.float32)
    batch_size = 32
    # Create PyTorch data loaders for the train and test data
    train_loader = DataLoader(TensorDataset(train_data_tensor, train_targets_tensor), batch_size=batch_size, shuffle=True
                              #num_workers=multiprocessing.cpu_count() # For this model, it seems that num_workers is less efficient when set
                              )
    test_loader = DataLoader(TensorDataset(test_data_tensor, test_targets_tensor), batch_size=batch_size, shuffle=True
                              #num_workers=multiprocessing.cpu_count() # For this model, it seems that num_workers is less efficient when set
                             )

    # Define the hyperparameters
    d_lstm = 64 # dimension of the LSTM network
    d_lat = 1 # dimension of the latent variable
    d_input = 1 # dimension of the input. TODO increasing this breaks things... 
    d_hidden = 1 # dimensionality of the hidden state of the LSTM. Determines how much information the network can store about the past 
    N = 50 # number of latent variable paths to simulate )
    t_sde = 1 # time step for SDE
    n_sde = 100 # number of latent variables in each latent variable path (i.e num days to simulate for each path)
    learning_rate = 10e-2 # learning rate for the optimizer
    num_epochs = 10 # number of epochs to train the model
    num_layers = 1 # number of layers in the LSTM network

    # Steps: 
    # 1.) Observed time-sequential data of dimension Dobs = 1 is fed to single-latyer LSTM netowrk of dimension D_lstm
        # 1a.) this implies that the observed data mapped from R^D_obs to R^D_LSTM inside the network 
    # 2.) The mapped observed data of R^D_LSTM is mapped to the initial latent variable z_0 { R^D_Lat through a linear layer mapping R^D_LSTM to R^D_Lat
    # 3.) The initial latent variable is fed to latent variable neural SDE framework (SDE block) 
    # 4.) The SDE block generates N latent variable paths of the form {z_0, ..., z_N__sde} where z(0) = z_0 and N_sde deontes number of latent variables in each latent variable path 
        # 4a.) This is done using the EM method, with drift and diffusion networks f* and g* 
        # 4b.) The dirft and diffusion networks consist of two-layer neural netws where the first layer maps to R^2*D_Lat and the second layer maps back to R^D_Lat
        # 4c.) The activation function for f* and g* is the tanh-function 
    


    # Define the optimizer

    # Define the loss function
    loss_fn = torch.nn.MSELoss()

    model = LSTMSDE_to_train(input_size=d_input,
                    num_layers=d_hidden,
                    lstm_size=d_lstm,
                    output_size=d_lat,
                    t_sde=t_sde,
                    n_sde=n_sde)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    fit_data = model.fit(train_loader, 
                         loss_fn, 
                         optimizer, 
                         num_epochs, 
                         x_inputs = {'train_data': train_data_tensor, 'test_data': test_data_tensor}, 
                         y_targets = {'train_data': train_targets_tensor, 'test_data': test_targets_tensor})
    #train_predictions, train_mse_over_time = model_2.train_model(train_loader, loss_fn, optimizer, num_epochs)
    print('Done training model 2')
    #test_predictions, test_mse_over_time = model_2.test_model(test_loader, loss_fn)
    #$train_rmse_over_time = train(train_loader, model, loss_fn, optimizer, num_epochs)
    print('Done training')
    # Evaluate the model on the training data
    #train_output, train_rmse = eval(model=model, loss_fn=loss_fn, x_input=train_data_tensor, y_target=train_targets_tensor)
    # Evaluate the model on the test data
    #test_output, test_rmse = eval(model=model, loss_fn=loss_fn, x_input=test_data_tensor, y_target=test_targets_tensor)

    #outputs = test(model, loss_fn, train_data_tensor, train_targets_tensor, test_data_tensor, test_targets_tensor)
    # We can't use NLLLoss because we are not doing classification
    # NLL loss in olaf's paper is defined differently than the standard NLL loss
    # loss_fn = torch.nn.NLLLoss() Olaf uses NLL but I think this is wrong.. that's for classification 

    
    # Calculate calibration error 
    #calibration_error = np.mean(np.abs(test_predictions - test_targets_tensor.numpy()[0:len(test_predictions)]))
    print('Finished Training')

    import plotly.graph_objects as go
    # Create a scatter plot for train targets vs train predictions

    time = list(range(len(train_targets)))

    # Create a line plot for time vs train targets
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=train_targets, mode='lines', name='Train Targets'))

    # Add a trace for time vs train predictions
    fig.add_trace(go.Scatter(x=time, y=fit_data[0]['train_data'], mode='lines', name='Train Predictions'))
    #fig.add_trace(go.Scatter(x=time, y=train_output, mode='lines', name='Train Predictions'))


    fig.update_layout(title='Time vs Train Targets and Predictions',
                    xaxis_title='Time',
                    yaxis_title='Value')
    fig.show()

    # Create a line plot for time vs train targets
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=time, y=test_targets, mode='lines', name='Test Targets'))

    # Add a trace for time vs train predictions
    fig2.add_trace(go.Scatter(x=time, y=fit_data[0]['test_data'], mode='lines', name='Test Predictions'))
    #fig2.add_trace(go.Scatter(x=time, y=test_output, mode='lines', name='Test Predictions'))

    fig2.update_layout(title='Time vs Test Targets and Predictions',
                    xaxis_title='Time (days)',
                    yaxis_title='Value')
    fig2.show()

    loss_time = list(range(len(fit_data[2])))
    # Create a line plot for train time vs train loss
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=loss_time, y=fit_data[2], mode='lines', name='Train Loss'))
    fig3.update_layout(title='Train Time vs Train Loss',
                    xaxis_title='Train Time (epochs)',
                    yaxis_title='Train Loss')
    fig3.show()

    # # Calculate confidences and accuracies for each bin
    # fraction_of_positives, mean_predicted_value = calibration_curve(test_targets, test_predictions, n_bins=10)

    # # Plot calibration curve
    # plt.plot(mean_predicted_value, fraction_of_positives, "s-")
    # plt.plot([0, 1], [0, 1], 'k--')  # Plot perfect calibration line
    # plt.xlabel('Mean predicted value')
    # plt.ylabel('Fraction of positives')
    # plt.title('Calibration Curve')
    # plt.show()