
def create_test_train_split_params(start_date_train='2009-01-01', 
    start_date_test='2017-01-02'):
    """Helper method to create the test split parameters for all models

    Args:
        start_date_train (_type_): _description_
        start_date_split (_type_): _description_

    Returns:
        _type_: _description_
    """    
    params = {}
    params['train_split_filter'] = lambda x: x['Date'] >= start_date_train  & x['Date'] < start_date_test
    params['test_split_filter']  = lambda x: x['Date'] >= start_date_test   & x['Date'] <= '2019-01-01'
    eval_filters = {'5_day', lambda x: x['Date']       >= '2019-01-02' & x['Date'] <= '2019-01-06',
                    '1_month', lambda x: x['Date']     >= '2019-01-02' & x['Date'] <= '2019-02-02',
                    '6_month', lambda x: x['Date']     >= '2019-01-02' & x['Date'] <= '2019-07-02',
                    '1_year', lambda x: x['Date']      >= '2019-01-02' & x['Date'] <= '2020-01-02',
                    '5_year', lambda x: x['Date']      >= '2019-01-02' & x['Date'] <= '2024-01-02'}, 
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
            'gbm_calculated' : {
                
                # Evaluation filters, corresponding to a dictionary of forecast dates starting on Jan 2 and ending in 
                # (1) 5 days, (2) 1 month, (3) 6 months, (4) 1 year, and (5) 5 years
                'model_hyperparameters': {'calculate_mu' : True, 'calculate_sigma': True}
            }   
            # 'gbm_2' : 'model_hyperparameters': {'mu': 0.0001, 'sigma': 0.1},
        }
    if lstm_sde:
        models_dict["LSTM_SDE"] = None 

    if lstm:
        # TODO allow test train split to also be a callback to a function, so we can filter specifically on the date
        # See valid model arguments documentation https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
        models_dict["LSTM"] = {
            'lstm_5node_1' : {
                'units' : 1,
                'library_hyperparameters' : {
                    'activation' : 'relu',
                    'recurrent_activation' : 'sigmoid',
                }
            },
        }
    # For each model in the models dict, add in the test and train split parameters from create_test_train_split_params
    test_train_split_params = create_test_train_split_params()
    for model_type, model_dict in models_dict.items():
        for model_name, model in model_dict.items(): 
            model['train_split_filter'] = test_train_split_params['train_split_filter']
            model['test_split_filter'] = test_train_split_params['test_split_filter']
            model['evaluation_filters'] = test_train_split_params['evaluation_filters']
    return models_dict