import logging 
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Scaler:
    def __init__(self):
        self._scalers = {}
        self._valid_scaler_types = ['MinMaxScaler', 'StandardScaler']
        self._default_scaler_type = 'MinMaxScaler'
    def scaler_names(self):
        """Returns the names of the scalers

        Returns:
            list: List of scaler names
        """        
        return list(self._scalers.keys())
    def _check_empty(self, df, vars):
        """Check if the dataframe or vars are empty

        Args:
            df (_type_): _description_
            vars (_type_): _description_

        Returns:
            _type_: _description_
        """        
        if df is None or df.shape[0] == 0:
            import pdb; pdb.set_trace()
            logging.error('Dataframe or vars are empty')
            return True
        return False
    def _check_existing_key(self, df_key):
        """Checks if the key exists in the scaler dictionary

        Args:
            df_key (string): df name
        """
        if df_key in self._scalers.keys():
            return True
        return False

    def _check_valid_scaler_type(self, scaler_type):
        """Checks if the scaler type is valid

        Args:
            scaler_type (string): scaler type

        Returns:
            bool: True if valid, False if invalid
        """        
        if scaler_type not in self._valid_scaler_types:
            logging.error(f"Invalid scaler type: {scaler_type}")
            return False
        return True

    def fit(self, df, df_name, scaler_type='MinMaxScaler'):
        """Fits the scaler to the dataframe

        Args:
            df (_type_): _description_
            df_name (string): df name
        """        
        # If the key exists in the dataframe, warn the user
        if self._check_existing_key(df_name):
            logging.warning(f"Scaler for {df_name} already exists, overriding existing scaler")
            
        self._scalers[df_name] = {}
        for var in df.select_dtypes('float64').columns:
            if not self._check_valid_scaler_type(scaler_type):
                return
            if scaler_type == 'StandardScaler':
                scaler = StandardScaler()
            if scaler_type == 'MinMaxScaler':
                scaler = MinMaxScaler()
            scaler.fit(df[[var]])
            self._scalers[df_name][var] = scaler

    def transform(self, df, df_name, columns=None):
        """Transforms the dataframe using the scaler. If a scaler with the given name does not exist, one will be created

        Args:
            df (pd.DataFrame): Dataframe to transform
            df_name (str): unique name of the dataframe
            columns (list[str], optional): Columns to scale. Defaults to None, which corresponds to all.

        Returns:
            pd.DataFrame: scaled dataframe 
        """        
        if self._check_empty(df, columns):
            return
        # If the key does not exist in the dictionary, warn the user that one is being created for them
        if not self._check_existing_key(df_name):
            logging.warning(f"Scaler for {df_name} does not exist, creating new scaler for transform and defaulting to {self._default_scaler_type}")
            # Call the fit method to create the scaler
            self.fit(df, df_name, self._default_scaler_type)
        # if no columns are passed, transform all float64 columns
        if columns is None:
            columns = df.select_dtypes('float64').columns
        
        df2 = df.copy(deep=True)
        for var in columns:
            df2[var] = self._scalers[df_name][var].transform(df2[[var]])
        return df2


    def inverse_transform(self, df, df_name, columns=None):
        """Inverse transforms the dataframe using the scaler
       
        Args:
            df (pd.DataFrame): Dataframe to transform
            df_name (str): unique name of the dataframe
            columns (list[str], optional): Columns to scale. Defaults to None, which corresponds to all.


        Returns:
            pd.DataFrame: scaled dataframe 
        """        
        if self._check_empty(df, columns):
            return
        if not self._check_existing_key(df_name):
            logging.warning("Unable to perform inverse transform")
            return
        # if no columns are passed, transform all float64 columns
        if columns is None:
            columns = df.select_dtypes('float64').columns

        df2 = df.copy(deep=True)
        for var in columns:
            df2[var] = self._scalers[df_name][var].inverse_transform(df2[[var]])
        return df2