from dateutil.relativedelta import relativedelta
from pandas import to_datetime
from project_globals import DataNames as DN 
def create_test_train_split_params(
    start_date_train='2009-01-01', # FOR TESTING 
    start_date_test='2017-01-02', # FOR TESTING 
    evaluation_start_date='2019-01-02' # FOR TESTING
    #start_date_train='2009-01-01', # USE THIS, ABOVE IS FOR TESTING 
    #start_date_test='2017-01-02', # USE THIS, ABOVE IS FOR TESTING 
    #evaluation_start_date='2019-01-01' # USE THIS, ABOVE IS FOR TESTING 
    ,):
    """Helper method to create the test split parameters for all models

    Args:
        start_date_train (_type_): _description_
        start_date_split (_type_): _description_

    Returns:
        _type_: _description_
    """    
    params = {}
    params['train_split_filter'] =  lambda x: (x['Date'] >= start_date_train) &  (x['Date'] < start_date_test)
    params['test_split_filter']  =  lambda x: (x['Date'] >= start_date_test) & (x['Date'] <= evaluation_start_date)
    # Create a dictionary of evaluation filters. The keys are the evaluation filter names, and the values are the lambda functions
    eval_filters = {}

    eval_filters['5_year'] = lambda x: (x['Date'] >= evaluation_start_date) & (x['Date']    <= to_datetime(evaluation_start_date) + relativedelta(years=5))
    eval_filters['1_year'] = lambda x: (x['Date'] >= evaluation_start_date) & (x['Date']    <= to_datetime(evaluation_start_date) + relativedelta(years=1))
    eval_filters['6_month'] = lambda x: (x['Date'] >= evaluation_start_date) & (x['Date']   <= to_datetime(evaluation_start_date) + relativedelta(months=6))
    eval_filters['1_month'] = lambda x: (x['Date'] >= evaluation_start_date) & (x['Date']   <= to_datetime(evaluation_start_date) + relativedelta(months=1))
    eval_filters['5_day'] = lambda x: (x['Date'] >= evaluation_start_date) & (x['Date']     <= to_datetime(evaluation_start_date) + relativedelta(days=5))

    params['evaluation_filters'] = eval_filters
    return params

def create_models_dict(gbm=True, lstm=True, lstm_sde=True):
    """Creates the dictionary of predictive models to use

    Args:
        gbm (bool, optional): Whether or not to use the GBM models. Defaults to True.
        lstm (bool, optional): Whether or not to use the LSTM models. Defaults to True.
        lstm_sde (bool, optional): Whether or not to use the LSTM SDE models. Defaults to True.

    Returns:
        dict:   Dictionary of model parameters. Primary key is the model type. 
                Secondary key is the model name, with the value being the model parameters.
    """    
    models_dict = {}
    if gbm: 
        models_dict["GBM"] = {
            'gbm_calculated_1_day' : {
                DN.params: {
                    'calculate_mu' : True, 
                    'calculate_sigma': True,
                    'window_size': 1,
                    'dt' : 1, # day
                    'num_sims' : 5
                },
            },

            'gbm_calculated_10_days' : {
                DN.params: {
                    'calculate_mu' : True, 
                    'calculate_sigma': True,
                    'window_size': 10,
                    'dt' : 1, # day
                    'num_sims' : 5
                },
            },

            'gbm_calculated_20_days' : {
                DN.params: {
                    'calculate_mu' : True, 
                    'calculate_sigma': True,
                    'window_size': 20,
                    'dt' : 1, # day 
                    'num_sims' : 5
                },
            },
            'gbm_calculated_50_days' : {
                DN.params: {
                    'calculate_mu' : True, 
                    'calculate_sigma': True,
                    'window_size': 50,
                    'dt' : 1,
                    'num_sims' : 5
                },
            }  ,
            'gbm_calculated_500_days' : {
                DN.params: {
                    'calculate_mu' : True, 
                    'calculate_sigma': True,
                    'window_size': 500,
                    'dt' : 1,
                    'num_sims' : 5
                },
            }  ,
            'gbm_calculated_1000_days' : {
                DN.params: {
                    'calculate_mu' : True, 
                    'calculate_sigma': True,
                    'window_size': 1000,
                    'dt' : 1,
                    'num_sims' : 5
                },
            }  ,
            'gbm_calculated_2000_days' : {
                DN.params: {
                    'calculate_mu' : True, 
                    'calculate_sigma': True,
                    'window_size': 2000,
                    'dt' : 1,
                    'num_sims' : 5
                },
            }  ,
            'gbm_calculated_3000_days' : {
                DN.params: {
                    'calculate_mu' : True, 
                    'calculate_sigma': True,
                    'window_size': 3000,
                    'dt' : 1,
                    'num_sims' : 5
                },
            }  ,
            'gbm_calculated_all_days' : {
                DN.params: {
                    'calculate_mu' : True, 
                    'calculate_sigma': True,
                    'window_size': None,
                    'dt' : 1,
                    'num_sims' : 5
                },
            }  ,
        }
    if lstm_sde:
        models_dict["LSTMSDE"] = {
            'lstm_sde_baseline' : {
                DN.params: {
                    'num_sims' : 5,
                    'num_epochs' : 5, 
                    'time_steps' : 10,
                    'batch_size' : 32,
                    'shuffle' : True,
                    'd_lstm' : 64, # dimension of the LSTM network
                    'd_lat' : 1, # dimension of the latent variable
                    'd_input' : 1, # dimension of the input. TODO increasing this breaks things... 
                    'd_hidden' : 16, # dimensionality of the hidden state of the LSTM. Determines how much information the network can store about the past 
                    'N' : 50, # number of latent variable paths to simulate )
                    't_sde' : 1, # time step for SDE
                    'n_sde' : 100, # number of latent variables in each latent variable path (i.e num days to simulate for each path)
                    'learning_rate' : 10e-2, # learning rate for the optimizer
                    'loss' : 'mean_squared_error',
                    'optimizer' : 'adam' # which optimizer to use. Options are 'adam' and 'sgd'
                },
            },
            'lstm_sde_learningrate1' : {
                DN.params: {
                    'num_sims' : 5,
                    'num_epochs' : 10, 
                    'time_steps' : 10,
                    'batch_size' : 32,
                    'shuffle' : True,
                    'd_lstm' : 64, # dimension of the LSTM network
                    'd_lat' : 1, # dimension of the latent variable
                    'd_input' : 1, # dimension of the input. TODO increasing this breaks things... 
                    'd_hidden' : 16, # dimensionality of the hidden state of the LSTM. Determines how much information the network can store about the past 
                    'N' : 50, # number of latent variable paths to simulate )
                    't_sde' : 1, # time step for SDE
                    'n_sde' : 100, # number of latent variables in each latent variable path (i.e num days to simulate for each path)
                    'learning_rate' : 10e-3, # learning rate for the optimizer
                    'loss' : 'mean_squared_error',
                    'optimizer' : 'adam' # which optimizer to use. Options are 'adam' and 'sgd'
                },
            },
            'lstm_sde_learningrate2' : {
                DN.params: {
                    'num_sims' : 5,
                    'num_epochs' : 10, 
                    'time_steps' : 10,
                    'batch_size' : 32,
                    'shuffle' : True,
                    'd_lstm' : 64, # dimension of the LSTM network
                    'd_lat' : 1, # dimension of the latent variable
                    'd_input' : 1, # dimension of the input. TODO increasing this breaks things... 
                    'd_hidden' : 16, # dimensionality of the hidden state of the LSTM. Determines how much information the network can store about the past 
                    'N' : 50, # number of latent variable paths to simulate )
                    't_sde' : 1, # time step for SDE
                    'n_sde' : 100, # number of latent variables in each latent variable path (i.e num days to simulate for each path)
                    'learning_rate' : 10e-4, # learning rate for the optimizer
                    'loss' : 'mean_squared_error',
                    'optimizer' : 'adam' # which optimizer to use. Options are 'adam' and 'sgd'
                },
            },
            'lstm_sde_hidden1' : {
                DN.params: {
                    'num_sims' : 5,
                    'num_epochs' : 10, 
                    'time_steps' : 30,
                    'batch_size' : 32,
                    'shuffle' : True,
                    'd_lstm' : 64, # dimension of the LSTM network
                    'd_lat' : 1, # dimension of the latent variable
                    'd_input' : 1, # dimension of the input. TODO increasing this breaks things... 
                    'd_hidden' : 1, # dimensionality of the hidden state of the LSTM. Determines how much information the network can store about the past 
                    'N' : 100, # number of latent variable paths to simulate )
                    't_sde' : 1, # time step for SDE
                    'n_sde' : 100, # number of latent variables in each latent variable path (i.e num days to simulate for each path)
                    'learning_rate' : 10e-2, # learning rate for the optimizer
                    'loss' : 'mean_squared_error',
                    'optimizer' : 'adam' # which optimizer to use. Options are 'adam' and 'sgd'
                },
            },
            'lstm_sde_hidden2' : {
                DN.params: {
                    'num_sims' : 5,
                    'num_epochs' : 10, 
                    'time_steps' : 30,
                    'batch_size' : 32,
                    'shuffle' : True,
                    'd_lstm' : 64, # dimension of the LSTM network
                    'd_lat' : 1, # dimension of the latent variable
                    'd_input' : 1, # dimension of the input. TODO increasing this breaks things... 
                    'd_hidden' : 2, # dimensionality of the hidden state of the LSTM. Determines how much information the network can store about the past 
                    'N' : 100, # number of latent variable paths to simulate )
                    't_sde' : 1, # time step for SDE
                    'n_sde' : 100, # number of latent variables in each latent variable path (i.e num days to simulate for each path)
                    'learning_rate' : 10e-2, # learning rate for the optimizer
                    'loss' : 'mean_squared_error',
                    'optimizer' : 'adam' # which optimizer to use. Options are 'adam' and 'sgd'
                },
            },
            'lstm_sde_hidden3' : {
                DN.params: {
                    'num_sims' : 5,
                    'num_epochs' : 10, 
                    'time_steps' : 30,
                    'batch_size' : 32,
                    'shuffle' : True,
                    'd_lstm' : 64, # dimension of the LSTM network
                    'd_lat' : 1, # dimension of the latent variable
                    'd_input' : 1, # dimension of the input. TODO increasing this breaks things... 
                    'd_hidden' : 5, # dimensionality of the hidden state of the LSTM. Determines how much information the network can store about the past 
                    'N' : 100, # number of latent variable paths to simulate )
                    't_sde' : 1, # time step for SDE
                    'n_sde' : 100, # number of latent variables in each latent variable path (i.e num days to simulate for each path)
                    'learning_rate' : 10e-2, # learning rate for the optimizer
                    'loss' : 'mean_squared_error',
                    'optimizer' : 'adam' # which optimizer to use. Options are 'adam' and 'sgd'
                },
            },
            'lstm_sde_hidden4' : {
                DN.params: {
                    'num_sims' : 5,
                    'num_epochs' : 10, 
                    'time_steps' : 30,
                    'batch_size' : 32,
                    'shuffle' : True,
                    'd_lstm' : 64, # dimension of the LSTM network
                    'd_lat' : 1, # dimension of the latent variable
                    'd_input' : 1, # dimension of the input. TODO increasing this breaks things... 
                    'd_hidden' : 5, # dimensionality of the hidden state of the LSTM. Determines how much information the network can store about the past 
                    'N' : 100, # number of latent variable paths to simulate )
                    't_sde' : 1, # time step for SDE
                    'n_sde' : 100, # number of latent variables in each latent variable path (i.e num days to simulate for each path)
                    'learning_rate' : 10e-2, # learning rate for the optimizer
                    'loss' : 'mean_squared_error',
                    'optimizer' : 'adam' # which optimizer to use. Options are 'adam' and 'sgd'
                },
            },
            'lstm_sde_hidden5' : {
                DN.params: {
                    'num_sims' : 5,
                    'num_epochs' : 10, 
                    'time_steps' : 30,
                    'batch_size' : 32,
                    'shuffle' : True,
                    'd_lstm' : 64, # dimension of the LSTM network
                    'd_lat' : 1, # dimension of the latent variable
                    'd_input' : 1, # dimension of the input. TODO increasing this breaks things... 
                    'd_hidden' : 10, # dimensionality of the hidden state of the LSTM. Determines how much information the network can store about the past 
                    'N' : 100, # number of latent variable paths to simulate )
                    't_sde' : 1, # time step for SDE
                    'n_sde' : 100, # number of latent variables in each latent variable path (i.e num days to simulate for each path)
                    'learning_rate' : 10e-2, # learning rate for the optimizer
                    'loss' : 'mean_squared_error',
                    'optimizer' : 'adam' # which optimizer to use. Options are 'adam' and 'sgd'
                },
            },
            'lstm_sde_dlstm1' : {
                DN.params: {
                    'num_sims' : 5,
                    'num_epochs' : 5, 
                    'time_steps' : 10,
                    'batch_size' : 32,
                    'shuffle' : True,
                    'd_lstm' : 4, # dimension of the LSTM network
                    'd_lat' : 1, # dimension of the latent variable
                    'd_input' : 1, # dimension of the input. TODO increasing this breaks things... 
                    'd_hidden' : 16, # dimensionality of the hidden state of the LSTM. Determines how much information the network can store about the past 
                    'N' : 50, # number of latent variable paths to simulate )
                    't_sde' : 1, # time step for SDE
                    'n_sde' : 100, # number of latent variables in each latent variable path (i.e num days to simulate for each path)
                    'learning_rate' : 10e-2, # learning rate for the optimizer
                    'loss' : 'mean_squared_error',
                    'optimizer' : 'adam' # which optimizer to use. Options are 'adam' and 'sgd'
                },
            },
            'lstm_sde_dlstm2' : {
                DN.params: {
                    'num_sims' : 5,
                    'num_epochs' : 5, 
                    'time_steps' : 10,
                    'batch_size' : 32,
                    'shuffle' : True,
                    'd_lstm' : 8, # dimension of the LSTM network
                    'd_lat' : 1, # dimension of the latent variable
                    'd_input' : 1, # dimension of the input. TODO increasing this breaks things... 
                    'd_hidden' : 16, # dimensionality of the hidden state of the LSTM. Determines how much information the network can store about the past 
                    'N' : 50, # number of latent variable paths to simulate )
                    't_sde' : 1, # time step for SDE
                    'n_sde' : 100, # number of latent variables in each latent variable path (i.e num days to simulate for each path)
                    'learning_rate' : 10e-2, # learning rate for the optimizer
                    'loss' : 'mean_squared_error',
                    'optimizer' : 'adam' # which optimizer to use. Options are 'adam' and 'sgd'
                },
            },
            'lstm_sde_dlstm3' : {
                DN.params: {
                    'num_sims' : 5,
                    'num_epochs' : 5, 
                    'time_steps' : 10,
                    'batch_size' : 32,
                    'shuffle' : True,
                    'd_lstm' : 8, # dimension of the LSTM network
                    'd_lat' : 1, # dimension of the latent variable
                    'd_input' : 1, # dimension of the input. TODO increasing this breaks things... 
                    'd_hidden' : 16, # dimensionality of the hidden state of the LSTM. Determines how much information the network can store about the past 
                    'N' : 50, # number of latent variable paths to simulate )
                    't_sde' : 1, # time step for SDE
                    'n_sde' : 100, # number of latent variables in each latent variable path (i.e num days to simulate for each path)
                    'learning_rate' : 10e-2, # learning rate for the optimizer
                    'loss' : 'mean_squared_error',
                    'optimizer' : 'adam' # which optimizer to use. Options are 'adam' and 'sgd'
                },
            },
            'lstm_sde_dlstm4' : {
                DN.params: {
                    'num_sims' : 5,
                    'num_epochs' : 5, 
                    'time_steps' : 10,
                    'batch_size' : 32,
                    'shuffle' : True,
                    'd_lstm' : 8, # dimension of the LSTM network
                    'd_lat' : 1, # dimension of the latent variable
                    'd_input' : 1, # dimension of the input. TODO increasing this breaks things... 
                    'd_hidden' : 32, # dimensionality of the hidden state of the LSTM. Determines how much information the network can store about the past 
                    'N' : 50, # number of latent variable paths to simulate )
                    't_sde' : 1, # time step for SDE
                    'n_sde' : 100, # number of latent variables in each latent variable path (i.e num days to simulate for each path)
                    'learning_rate' : 10e-2, # learning rate for the optimizer
                    'loss' : 'mean_squared_error',
                    'optimizer' : 'adam' # which optimizer to use. Options are 'adam' and 'sgd'
                },
            },
            'lstm_sde_dlstmhidden1' : {
                DN.params: {
                    'num_sims' : 5,
                    'num_epochs' : 5, 
                    'time_steps' : 10,
                    'batch_size' : 32,
                    'shuffle' : True,
                    'd_lstm' : 32, # dimension of the LSTM network
                    'd_lat' : 1, # dimension of the latent variable
                    'd_input' : 1, # dimension of the input. TODO increasing this breaks things... 
                    'd_hidden' : 8, # dimensionality of the hidden state of the LSTM. Determines how much information the network can store about the past 
                    'N' : 50, # number of latent variable paths to simulate )
                    't_sde' : 1, # time step for SDE
                    'n_sde' : 100, # number of latent variables in each latent variable path (i.e num days to simulate for each path)
                    'learning_rate' : 10e-2, # learning rate for the optimizer
                    'loss' : 'mean_squared_error',
                    'optimizer' : 'adam' # which optimizer to use. Options are 'adam' and 'sgd'
                },
            },
            'lstm_sde_dlstmhidden2' : {
                DN.params: {
                    'num_sims' : 5,
                    'num_epochs' : 5, 
                    'time_steps' : 10,
                    'batch_size' : 32,
                    'shuffle' : True,
                    'd_lstm' : 16, # dimension of the LSTM network
                    'd_lat' : 1, # dimension of the latent variable
                    'd_input' : 1, # dimension of the input. TODO increasing this breaks things... 
                    'd_hidden' : 4, # dimensionality of the hidden state of the LSTM. Determines how much information the network can store about the past 
                    'N' : 50, # number of latent variable paths to simulate )
                    't_sde' : 1, # time step for SDE
                    'n_sde' : 100, # number of latent variables in each latent variable path (i.e num days to simulate for each path)
                    'learning_rate' : 10e-2, # learning rate for the optimizer
                    'loss' : 'mean_squared_error',
                    'optimizer' : 'adam' # which optimizer to use. Options are 'adam' and 'sgd'
                },
            },
            'lstm_sde_timesteps1' : {
                DN.params: {
                    'num_sims' : 5,
                    'num_epochs' : 5, 
                    'time_steps' : 5,
                    'batch_size' : 32,
                    'shuffle' : True,
                    'd_lstm' : 64, # dimension of the LSTM network
                    'd_lat' : 1, # dimension of the latent variable
                    'd_input' : 1, # dimension of the input. TODO increasing this breaks things... 
                    'd_hidden' : 16, # dimensionality of the hidden state of the LSTM. Determines how much information the network can store about the past 
                    'N' : 50, # number of latent variable paths to simulate )
                    't_sde' : 1, # time step for SDE
                    'n_sde' : 100, # number of latent variables in each latent variable path (i.e num days to simulate for each path)
                    'learning_rate' : 10e-2, # learning rate for the optimizer
                    'loss' : 'mean_squared_error',
                    'optimizer' : 'adam' # which optimizer to use. Options are 'adam' and 'sgd'
                },
            },
            'lstm_sde_timesteps2' : {
                DN.params: {
                    'num_sims' : 5,
                    'num_epochs' : 5, 
                    'time_steps' : 30,
                    'batch_size' : 32,
                    'shuffle' : True,
                    'd_lstm' : 64, # dimension of the LSTM network
                    'd_lat' : 1, # dimension of the latent variable
                    'd_input' : 1, # dimension of the input. TODO increasing this breaks things... 
                    'd_hidden' : 16, # dimensionality of the hidden state of the LSTM. Determines how much information the network can store about the past 
                    'N' : 50, # number of latent variable paths to simulate )
                    't_sde' : 1, # time step for SDE
                    'n_sde' : 100, # number of latent variables in each latent variable path (i.e num days to simulate for each path)
                    'learning_rate' : 10e-2, # learning rate for the optimizer
                    'loss' : 'mean_squared_error',
                    'optimizer' : 'adam' # which optimizer to use. Options are 'adam' and 'sgd'
                },
            },
            'lstm_sde_timesteps3' : {
                DN.params: {
                    'num_sims' : 5,
                    'num_epochs' : 5, 
                    'time_steps' : 60,
                    'batch_size' : 32,
                    'shuffle' : True,
                    'd_lstm' : 64, # dimension of the LSTM network
                    'd_lat' : 1, # dimension of the latent variable
                    'd_input' : 1, # dimension of the input. TODO increasing this breaks things... 
                    'd_hidden' : 16, # dimensionality of the hidden state of the LSTM. Determines how much information the network can store about the past 
                    'N' : 50, # number of latent variable paths to simulate )
                    't_sde' : 1, # time step for SDE
                    'n_sde' : 100, # number of latent variables in each latent variable path (i.e num days to simulate for each path)
                    'learning_rate' : 10e-2, # learning rate for the optimizer
                    'loss' : 'mean_squared_error',
                    'optimizer' : 'adam' # which optimizer to use. Options are 'adam' and 'sgd'
                },
            },
            'lstm_sde_timesteps4' : {
                DN.params: {
                    'num_sims' : 5,
                    'num_epochs' : 5, 
                    'time_steps' : 180,
                    'batch_size' : 32,
                    'shuffle' : True,
                    'd_lstm' : 64, # dimension of the LSTM network
                    'd_lat' : 1, # dimension of the latent variable
                    'd_input' : 1, # dimension of the input. TODO increasing this breaks things... 
                    'd_hidden' : 16, # dimensionality of the hidden state of the LSTM. Determines how much information the network can store about the past 
                    'N' : 50, # number of latent variable paths to simulate )
                    't_sde' : 1, # time step for SDE
                    'n_sde' : 100, # number of latent variables in each latent variable path (i.e num days to simulate for each path)
                    'learning_rate' : 10e-2, # learning rate for the optimizer
                    'loss' : 'mean_squared_error',
                    'optimizer' : 'adam' # which optimizer to use. Options are 'adam' and 'sgd'
                },
            },
        } 

    if lstm:
        # TODO allow test train split to also be a callback to a function, so we can filter specifically on the date
        # See valid model arguments documentation https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
        models_dict["LSTM"] = {
            # Baseline
            'lstm_5node_baseline' : {
                'units' : 1,
                'library_hyperparameters' : {
                    'activation' : 'relu',
                    'recurrent_activation' : 'sigmoid',
                    'num_sims' : 5,
                    'num_layers': 5,
                    'epochs': 10,
                    'hidden_nodes': 25,
                    'time_steps': 25,
                }
            },
            # 'lstm_5node_epochtest1' : {
            #     'units' : 1,
            #     'library_hyperparameters' : {
            #         'activation' : 'relu',
            #         'recurrent_activation' : 'sigmoid',
            #         'num_sims' : 5,
            #         'num_layers': 5,
            #         'epochs': 5,
            #         'hidden_nodes': 25,
            #         'time_steps': 10,
            #     }
            # },
            'lstm_5node_epochtest2' : {
                'units' : 1,
                'library_hyperparameters' : {
                    'activation' : 'relu',
                    'recurrent_activation' : 'sigmoid',
                    'num_sims' : 5,
                    'num_layers': 5,
                    'epochs': 15,
                    'hidden_nodes': 25,
                    'time_steps': 10,
                }
            },
            'lstm_5node_timemstep1' : {
                'units' : 1,
                'library_hyperparameters' : {
                    'activation' : 'relu',
                    'recurrent_activation' : 'sigmoid',
                    'num_sims' : 5,
                    'num_layers': 5,
                    'epochs': 10,
                    'hidden_nodes': 25,
                    'time_steps': 10,
                }
            },
            'lstm_5node_timemstep2' : {
                'units' : 1,
                'library_hyperparameters' : {
                    'activation' : 'relu',
                    'recurrent_activation' : 'sigmoid',
                    'num_sims' : 5,
                    'num_layers': 5,
                    'epochs': 10,
                    'hidden_nodes': 25,
                    'time_steps': 50,
                }
            },
            # 'lstm_5node_timestep3' : {
            #     'units' : 1,
            #     'library_hyperparameters' : {
            #         'activation' : 'relu',
            #         'recurrent_activation' : 'sigmoid',
            #         'num_sims' : 5,
            #         'num_layers': 5,
            #         'epochs': 10,
            #         'hidden_nodes': 25,
            #         'time_steps': 100,
            #     }
            # },

            'lstm_5node_hiddennodes1' : {
                'units' : 1,
                'library_hyperparameters' : {
                    'activation' : 'relu',
                    'recurrent_activation' : 'sigmoid',
                    'num_sims' : 5,
                    'num_layers': 5,
                    'epochs': 10,
                    'hidden_nodes': 10,
                    'time_steps': 25,
                }
            },
            # 'lstm_5node_hiddennodes2' : {
            #     'units' : 1,
            #     'library_hyperparameters' : {
            #         'activation' : 'relu',
            #         'recurrent_activation' : 'sigmoid',
            #         'num_sims' : 5,
            #         'num_layers': 5,
            #         'epochs': 10,
            #         'hidden_nodes': 25,
            #         'time_steps': 25,
            #     }
            # },
            # 'lstm_5node_hiddennodes3' : {
            #     'units' : 1,
            #     'library_hyperparameters' : {
            #         'activation' : 'relu',
            #         'recurrent_activation' : 'sigmoid',
            #         'num_sims' : 5,
            #         'num_layers': 5,
            #         'epochs': 10,
            #         'hidden_nodes': 50,
            #         'time_steps': 25,
            #     }
            # },
            # 'lstm_5node_layers1' : {
            #     'units' : 1,
            #     'library_hyperparameters' : {
            #         'activation' : 'relu',
            #         'recurrent_activation' : 'sigmoid',
            #         'num_sims' : 5,
            #         'num_layers': 10,
            #         'epochs': 10,
            #         'hidden_nodes': 25,
            #         'time_steps': 50,
            #     }
            # },
            'lstm_5node_layers2' : {
                'units' : 1,
                'library_hyperparameters' : {
                    'activation' : 'relu',
                    'recurrent_activation' : 'sigmoid',
                    'num_sims' : 5,
                    'num_layers': 25,
                    'epochs': 10,
                    'hidden_nodes': 25,
                    'time_steps': 50,
                }
            },
            # 'lstm_5node_layers3' : {
            #     'units' : 1,
            #     'library_hyperparameters' : {
            #         'activation' : 'relu',
            #         'recurrent_activation' : 'sigmoid',
            #         'num_sims' : 5,
            #         'num_layers': 50,
            #         'epochs': 10,
            #         'hidden_nodes': 25,
            #         'time_steps': 50,
            #     }
            # },

            # 'lstm_5node_layers4' : {
            #     'units' : 1,
            #     'library_hyperparameters' : {
            #         'activation' : 'relu',
            #         'recurrent_activation' : 'sigmoid',
            #         'num_sims' : 5,
            #         'num_layers': 50,
            #         'epochs': 10,
            #         'hidden_nodes': 25,
            #         'time_steps': 50,
            #     }
            # },
        }
    # For each model in the models dict, add in the test and train split parameters from create_test_train_split_params
    # Note that this applies the same test train split parameters to every model
    test_train_split_params = create_test_train_split_params()
    for model_type, model_dict in models_dict.items():
        for model_name, model in model_dict.items(): 
            model['test_split_filter'] = test_train_split_params['test_split_filter']
            model['evaluation_filters'] = test_train_split_params['evaluation_filters']
            model['train_split_filter'] = test_train_split_params['train_split_filter']
    return models_dict