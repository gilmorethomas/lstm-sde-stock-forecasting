from dateutil.relativedelta import relativedelta
from pandas import to_datetime
def create_test_train_split_params(
    start_date_train='2010-01-01', # FOR TESTING 
    start_date_test='2020-10-31', # FOR TESTING 
    evaluation_start_date='2020-12-31' # FOR TESTING
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
            'gbm_calculated_10_days' : {
                'model_hyperparameters': {
                    'calculate_mu' : True, 
                    'calculate_sigma': True,
                    'window_size': None,
                    'dt' : 1, # day
                    'num_sims' : 5
                },
            },

            'gbm_calculated_20_days' : {
                'model_hyperparameters': {
                    'calculate_mu' : True, 
                    'calculate_sigma': True,
                    'window_size': 20,
                    'dt' : 1, # day 
                    'num_sims' : 5
                },
            },

            'gbm_calculated_50_days' : {
                'model_hyperparameters': {
                    'calculate_mu' : True, 
                    'calculate_sigma': True,
                    'window_size': 50,
                    'dt' : 1,
                    'num_sims' : 5
                },
            }  ,
            # 'GBM Steady Large Increase' : {
            #     'model_hyperparameters': {
            #         'calculate_mu' : False, 
            #         'calculate_sigma': False, 
            #         'mu': 0.0001, 
            #         'sigma': 0.1,
            #         'num_sims' : 5
            #     },
            # },
            # 'GBM Unsteady 1' : {
            #     'model_hyperparameters': {
            #         'calculate_mu' : False, 
            #         'calculate_sigma': False, 
            #         'mu': 0.1, 
            #         'sigma': 0.05,
            #         'num_sims' : 5
            #     },
            # },
            # 'GBM Unsteady 2' : {
            #     'model_hyperparameters': {
            #         'calculate_mu' : False, 
            #         'calculate_sigma': False, 
            #         'mu': 0.1, 
            #         'sigma': 0.0,
            #         'num_sims' : 5
            #     },
            # }
        }
    if lstm_sde:
        models_dict["LSTM_SDE"] = {
            'lstm_sde_1' : {
                'model_hyperparameters': {
                    'num_sims' : 1,
                    'num_epochs' : 1, 
                    'window_size' : 2,
                    'batch_size' : 32,
                    'shuffle' : True,
                    'd_lstm' : 64, # dimension of the LSTM network
                    'd_lat' : 1, # dimension of the latent variable
                    'd_input' : 1, # dimension of the input. TODO increasing this breaks things... 
                    'd_hidden' : 1, # dimensionality of the hidden state of the LSTM. Determines how much information the network can store about the past 
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
            'lstm_5node_1' : {
                'units' : 1,
                'library_hyperparameters' : {
                    'activation' : 'relu',
                    'recurrent_activation' : 'sigmoid',
                    'num_sims' : 1
                }
            },
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