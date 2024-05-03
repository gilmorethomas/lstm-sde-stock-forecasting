class TimeSeriesGlobals: 
    temp = 'temp'
    ...

class DataNames: 
    all_data = 'all_data'
    train_data = 'train_data'
    test_data = 'test_data'
    normalized = 'normalized'
    not_normalized = 'not_normalized'
    dataloaders = 'dataloaders'
    raw = 'raw'
    proc = 'proc'
    tensors = 'tensors'
    x = 'x'
    y = 'y'
    evaluation = 'evaluation'
    params = 'model_hyperparameters'
    # Structure for a data dict can be expected to be 
    # data_dict = {
    #    DN.normalized'normalized' : {
    #        'train_data' : pd.DataFrame,
    #        'test_data' : pd.DataFrame
    #    },
    #    'not_normalized' : {
    #        'train_data' : pd.DataFrame,
    #        'test_data' : pd.DataFrame
    #    }
    # }

class ModelStructure: 
    report = 'report'
    perf = 'model_performance'
    predictions = 'predictions' 
    normalized = 'normalized'
    not_normalized = 'not_normalized'
    data = 'data'

class ModelTypes: 
    lstm = 'LSTM'
    lstm_sde = 'LSTMSDE'
    gbm = 'GBM'