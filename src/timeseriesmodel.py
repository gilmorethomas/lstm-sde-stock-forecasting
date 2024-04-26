from model import Model
import logging 
import numpy as np 
class TimeSeriesModel(Model):
    """Defines a Time Series Model that inherits from model. This class is currently 
    only used to validate that expected date columns exist in x variables and data columns

    Args:
        Model (_type_): _description_
    """    
    def __init__(self, 
        data, 
        model_hyperparameters, 
        save_dir, 
        model_name,
        x_vars:list,
        y_vars:list,
        seed:np.random.RandomState,
        test_split_filter=None, 
        train_split_filter=None, 
        evaluation_filters:dict={},
        save_html=True,
        save_png=True
    ):
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
        logging.info("Checking that required variables for time series models are present")
        assert 'Date' in x_vars and 'Date' in data.columns, "Date must be in x_vars and data.columns"
        assert 'Days_since_start' in x_vars and 'Days_since_start' in data.columns, "Days_since_start must be in x_vars and data.columns"