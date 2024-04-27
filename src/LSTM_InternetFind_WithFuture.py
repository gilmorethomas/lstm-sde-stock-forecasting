import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import math
from sklearn.metrics import mean_squared_error
import plotly.io as pio
pio.renderers.default='browser'

# Load the data
df = pd.read_csv(r"/Users/thomasgilmore/Documents/40_Software/lstm-sde-stock-forecasting/input/00_raw/AAPL/AAPL.csv")
df2 = df.reset_index()['Close']

# Scale the data
scaler = MinMaxScaler()
df2 = scaler.fit_transform(np.array(df2).reshape(-1, 1))

# Split data into train and test sets
train_size = int(len(df2) * 0.95)
test_size = len(df2) - train_size
train_data, test_data = df2[0:train_size,:], df2[train_size:len(df2),:1]

# Function to create dataset
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Create input-output datasets with time step 100
time_step = 100
X_train, Y_train = create_dataset(train_data, time_step)
X_test, Y_test = create_dataset(test_data, time_step)

# Define the model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()
# Train the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=64, verbose=1)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transform predictions to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Predict future values
n_future = 30  # Number of future days to predict
x_past = X_train[-1:]  # Initialize with the last sequence from training data
y_future = []  # Predicted target values
for i in range(n_future):
    y_pred = model.predict(x_past)
    y_future.append(y_pred[0, 0])
    x_past = np.append(x_past[:, 1:], y_pred, axis=1)

# Inverse transform to get actual values
predicted_values = scaler.inverse_transform(np.array(y_future).reshape(-1, 1))

# Calculate RMSE
train_rmse = math.sqrt(mean_squared_error(Y_train, train_predict))
test_rmse = math.sqrt(mean_squared_error(Y_test, test_predict))
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

# Plotting
look_back = time_step

train_data = scaler.inverse_transform(train_data)
test_data = scaler.inverse_transform(test_data)

array1 = np.array(train_predict)
array2 = np.array(train_data)
array3 = np.array(test_predict)
array4 = np.array(test_data)
array5 = np.array(predicted_values)

# Reshape arrays to be 1-dimensional
array1_1d = array1.flatten()
array2_1d = array2.flatten()
array3_1d = array3.flatten()
array4_1d = array4.flatten()
array5_1d = array5.flatten()

# Determine the length of the longest array
# Get the last date in the 'Date' column
last_date = df['Date'].iloc[-1]

# Generate a new date range starting from the day after the last date in 'Date'
new_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_future, freq=pd.infer_freq(df['Date']))

# Create a DataFrame with the new dates
new_dates_df = pd.DataFrame({'Date': new_dates})

# Concatenate the original DataFrame with the new dates DataFrame
df_extended = pd.concat([df, new_dates_df], ignore_index=True)
 
max_len = len(df_extended)

# Fill missing values with NaN
array1_filled = np.pad(array1_1d, (look_back, max_len - len(array1_1d)-look_back), constant_values=np.nan)
array2_filled = np.pad(array2_1d, (0, max_len - len(array2_1d)), constant_values=np.nan)
array3_filled = np.pad(array3_1d, (len(train_data)+look_back, max_len - len(array3_1d)-len(train_data)-look_back), constant_values=np.nan)
array4_filled = np.pad(array4_1d, (len(train_data), max_len - len(array4_1d)-len(train_data)), constant_values=np.nan)
array5_filled = np.pad(array5_1d, (max_len-len(array5_1d), max_len), constant_values=np.nan)

# Creating DataFrame
PlotData = pd.DataFrame({'Date': df_extended['Date'], 
                         'train_predict': array1_filled, 
                         'train_data': array2_filled,
                         'test_predict': array3_filled,
                         'test_data': array4_filled,
                         'future': array5_filled})


# Create traces
trace1 = go.Scatter(x=PlotData['Date'], y=PlotData['train_predict'], mode='lines', name='Train Predict')
trace2 = go.Scatter(x=PlotData['Date'], y=PlotData['train_data'], mode='lines', name='Train Data')
trace3 = go.Scatter(x=PlotData['Date'], y=PlotData['test_predict'], mode='lines', name='Test Predict')
trace4 = go.Scatter(x=PlotData['Date'], y=PlotData['test_data'], mode='lines', name='Test Data')
trace5 = go.Scatter(x=PlotData['Date'], y=PlotData['future'], mode='lines', name='Future Predict')

# Create layout
layout = go.Layout(title='Data',
                   xaxis=dict(title='Date'),
                   yaxis=dict(title='Value'))

# Create figure
fig = go.Figure(data=[trace1, trace2, trace3, trace4, trace5], layout=layout)

# Plot figure
fig.show()

