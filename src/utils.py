from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# from pathos.multiprocessing import ProcessingPool as ProcessPoolExecutor
#from tqdm.auto import tqdm 
from threadpoolctl import threadpool_limits 
import multiprocess as mp 
import time 
from lstm_logger import logger as logging
import colorlog
import pandas as pd 

def timer_decorator(func): 
    def wrapper(*args, **kwargs): 
        start = time.time()
        ret_val = func(*args, **kwargs)
        end = time.time()
        print(f"Function {func.__name__} took {end - start} seconds to run")
        return ret_val
    return wrapper

def parallelize(func, args_list, executor=ProcessPoolExecutor(), run_parallel = True):
    if not run_parallel or len(args_list) <= 1:
        return _run_sequential(func, args_list)
    return _run_parallel(func, args_list, executor)


def _run_parallel(func, args_list, executor):
    """Helper function to run a function in parallel 
    LIMITATION: This function cannot take in objects that are not serializable. 
    That would include something like a lambda function.

    Args:
        func (_type_): _description_
        args_list (_type_): _description_
        executor (_type_): _description_

    Returns:
        _type_: _description_
    """    
    results = []
    # futures = {executor.submit(func, *args) : 1 for args in args_list}

    # with threadpool_limits(limits=1, user_api=None):
    #     for future in tqdm(futures, total=len(futures)): 
    #         try:
    #             results.append(future.result())
    #         except Exception as e:
    #             print(f"Exception occurred during execution: {e}")
    #             results.append(None)
    return results

def _run_sequential(func, args_list):
    # Run sequentially 
    results = []
    for args in args_list:
    #for args in tqdm(args_list):
        try:
            results.append(func(*args))
        except Exception as e:
            print(f"Exception occurred during execution: {e}")
            results.append(None)
    return results

def drop_nans_from_data_dict(data_dict, calling_class=None, context=None): 
    """Drops nans from the data dictionary and warns user

    Args:
        data_dict (dictionary): n-level dictionary of the following format
            {key: value} where value can be a DataFrame or another dictionary
            For instance: 
            {
                'key1': pd.DataFrame,
                'key2': {
                    'key3': pd.DataFrame,
                    'key4': pd.DataFrame
                }
                'key5' : {
                    'key6': {
                        'key7': pd.DataFrame,
                        'key8': pd.DataFrame
                }
            }

        context (function): The function that is calling this function

    Returns:
        _type_: _description_
    """      
    log_messages  = []
    data_dict, log_messages = _drop_nans_from_data_dict(data_dict, log_messages)
    if not len(log_messages) == 0:
        if calling_class is not None and context is not None:
            log_messages = [(f"Dropping NaNs from data dictionary in in class {type(calling_class)}, function {context.__name__}")] + log_messages
        else:
            log_messages = ["Dropping NaNs from data dictionary"] + log_messages
        logging.warning("\n".join(log_messages))
    return data_dict
def _drop_nans_from_data_dict(data_dict, log_messages, keys_to_level=None): 
    """Helper function to drop nans from the data dictionary

    Args:

    """ 
    if keys_to_level is None:
        keys_to_level = []

    for key, value in data_dict.items():
        if isinstance(value, pd.DataFrame):
            if value.isnull().values.any():
                num_nans = value.isnull().sum().sum()
                # Deal with keys at the first level
                if len(keys_to_level) == 0:
                    log_messages.append(f"The DataFrame associated with key '{key}' contains {num_nans} NaN values.")
                # Deal with keys at any other level
                else: 
                    log_messages.append(f"The DataFrame associated with key '{key}' contains NaN values at level {keys_to_level}.")
            value.dropna(inplace=True)
        elif isinstance(value, dict):
            data_dict[key] = _drop_nans_from_data_dict(value, log_messages, keys_to_level + [key]) 
    return data_dict, log_messages