import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import logging
import multiprocessing
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

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
    def __init__(self, input_size, hidden_size, output_size, num_layers, t_sde, n_sde):
        super(LSTMSDE, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers = num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sde_block = SDEBlock(output_size, n_sde, t_sde)

    def forward(self, x):
        # Step 1: Observed time-sequential data of dimension D_obs = input_size is fed to single-layer LSTM network of dimension D_lstm = hidden_size
        # Ensure that x has the correct size (batch_size, seq_len, input_size)
        #x = x.unsqueeze(1)  # add a singleton dimension for seq_len if it's missing (i.e. if you are only using one step)

        # Data mapped from input size to hidden size 
        logging.info(f"Shape of x: {x.shape}")
        # Create the tensors filled with zeros, where we have the (number of layers in LSTM, the batch size, and the size of the hidden state)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) # initial hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) # initial cell state

        # Forward pass through the LSTM layer
        # This should be of shape (batch_size, seq_len, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        logging.info(f"Shape of x: {out.shape}")
        
        # Step 2: The mapped observed data of R^D_LSTM is mapped to the initial latent 
        # variable z_0 { R^D_Lat through a linear layer mapping R^D_LSTM to R^D_Lat
        z0 = self.fc(out[:, -1, :])

        # Step 3: The initial latent variable is fed to latent variable neural SDE framework (SDE block)
        # The SDE block is not implemented in this code, but you would pass z0 to it here
        out = self.sde_block(z0)

        return out


# Function to create dataset
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), :]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

if __name__ == "__main__":
    df = pd.read_csv(r"/Users/thomasgilmore/Documents/40_Software/lstm-sde-stock-forecasting/input/00_raw/AAPL/AAPL.csv").reset_index()
    # Scale the data
    df = df['Close']
    scaler = MinMaxScaler()
    #import pdb; pdb.set_trace()
    df2 = scaler.fit_transform(np.array(df).reshape(-1, 1))

    
    sequence_length = 1 # number of days to look back when making a prediction 
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

    # Create PyTorch data loaders for the train and test data
    train_loader = DataLoader(TensorDataset(train_data_tensor, train_targets_tensor), batch_size=32, 
                              #num_workers=multiprocessing.cpu_count() # For this model, it seems that num_workers is less efficient when set
                              )
    test_loader = DataLoader(TensorDataset(test_data_tensor, test_targets_tensor), batch_size=32, 
                              #num_workers=multiprocessing.cpu_count() # For this model, it seems that num_workers is less efficient when set
                             )


    # Define the hyperparameters
    #d_input = 1 # dimension of the input
    d_lstm = 64 # dimension of the LSTM network
    d_lat = 1 # dimension of the latent variable
    d_input = 1 # dimension of the input

    N = 5 # number of latent variable paths to simulate )
    t_sde = 1 # time step for SDE
    n_sde = 100 # number of latent variables in each latent variable path (i.e num days to simulate for each path)
    learning_rate = 10e-2 # learning rate for the optimizer
    num_epochs = 50 # number of epochs to train the model
    num_layers = 1 # number of layers in the LSTM network

    model = LSTMSDE(input_size=d_input, 
                    hidden_size=d_lstm, 
                    output_size=d_lat, 
                    num_layers = num_layers,
                    t_sde=t_sde, 
                    n_sde=n_sde)
    print(model)

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
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Define the loss function
    loss_fn = torch.nn.MSELoss()

    # We can't use NLLLoss because we are not doing classification
    # NLL loss in olaf's paper is defined differently than the standard NLL loss
    # loss_fn = torch.nn.NLLLoss() Olaf uses NLL but I think this is wrong.. that's for classification 


    # Training loop
    train_predictions = []
    train_mse_over_time = []  # Initialize list to store MSE over time

    for epoch in range(num_epochs):
        epoch_losses = []  # Initialize list to store losses for this epoch

        for i, (inputs, targets) in enumerate(train_loader):
            
            # Forward pass
            outputs = model(inputs)

            # Ensure that the outputs and targets have the same shape
            outputs = outputs.squeeze()  # remove extra dimensions from outputs
            #targets = targets.unsqueeze(1)  # add an extra dimension to targets
            # Append outputs to train_predictions
            # Calculate the loss
            loss = loss_fn(outputs, targets)
            epoch_losses.append(loss.item())  # Append loss for this batch

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Calculate average loss for this epoch and append to mse_over_time
        train_mse_over_time.append(np.mean(epoch_losses))
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Mean Loss: {np.mean(epoch_losses)}')
    
    # Pull final train predictions 
    train_predictions = outputs.detach().numpy()
    
    # Evaluation loop
    test_mse_over_time = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(train_loader):
            epoch_losses = []
            outputs = model(inputs).squeeze()
            loss = loss_fn(outputs, targets)
            print(f'Test Loss: {loss.item()}')
            epoch_losses.append(loss.item())
        test_mse_over_time.append(np.mean(epoch_losses))

    # Pull final test predictions
    test_predictions = outputs.detach().numpy() 
    # Calculate calibration error 
    calibration_error = np.mean(np.abs(test_predictions - test_targets_tensor.numpy()[0:len(test_predictions)]))
    print('Finished Training')

    import plotly.graph_objects as go
    # Create a scatter plot for train targets vs train predictions

    time = list(range(len(train_targets)))

    # Create a line plot for time vs train targets
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=train_targets, mode='lines', name='Train Targets'))

    # Add a trace for time vs train predictions
    fig.add_trace(go.Scatter(x=time, y=train_predictions, mode='lines', name='Train Predictions'))

    fig.update_layout(title='Time vs Train Targets and Predictions',
                    xaxis_title='Time',
                    yaxis_title='Value')
    fig.show()

    # Create a line plot for time vs train targets
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=time, y=train_targets, mode='lines', name='Train Targets'))

    # Add a trace for time vs train predictions
    fig2.add_trace(go.Scatter(x=time, y=train_predictions, mode='lines', name='Train Predictions'))

    fig2.update_layout(title='Time vs Train Targets and Predictions',
                    xaxis_title='Time',
                    yaxis_title='Value')
    fig2.show()


    train_time = list(range(len(train_mse_over_time)))
    test_time = list(range(len(test_mse_over_time)))

    # Create a line plot for train time vs train loss
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=train_time, y=train_mse_over_time, mode='lines', name='Train Loss'))
    fig1.update_layout(title='Train Time vs Train Loss',
                    xaxis_title='Train Time',
                    yaxis_title='Train Loss')
    fig1.show()

    # Create a line plot for test time vs test loss
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=test_time, y=test_mse_over_time, mode='lines', name='Test Loss'))
    fig2.update_layout(title='Test Time vs Test Loss',
                    xaxis_title='Test Time',
                    yaxis_title='Test Loss')
    fig2.show()
    # # Calculate confidences and accuracies for each bin
    # fraction_of_positives, mean_predicted_value = calibration_curve(test_targets, test_predictions, n_bins=10)

    # # Plot calibration curve
    # plt.plot(mean_predicted_value, fraction_of_positives, "s-")
    # plt.plot([0, 1], [0, 1], 'k--')  # Plot perfect calibration line
    # plt.xlabel('Mean predicted value')
    # plt.ylabel('Fraction of positives')
    # plt.title('Calibration Curve')
    # plt.show()