from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# from pathos.multiprocessing import ProcessingPool as ProcessPoolExecutor
from tqdm.auto import tqdm 
from threadpoolctl import threadpool_limits 
import multiprocess as mp 
import time 

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
    futures = {executor.submit(func, *args) : 1 for args in args_list}

    with threadpool_limits(limits=1, user_api=None):
        for future in tqdm(futures, total=len(futures)): 
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Exception occurred during execution: {e}")
                results.append(None)
    return results

def _run_sequential(func, args_list):
    # Run sequentially 
    results = []
    for args in tqdm(args_list):
        try:
            results.append(func(*args))
        except Exception as e:
            print(f"Exception occurred during execution: {e}")
            results.append(None)
    return results