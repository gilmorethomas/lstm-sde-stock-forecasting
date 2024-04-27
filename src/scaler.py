import logging 
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Scaler:
    def __init__(self, default_scaler_type = 'MinMaxScaler', log_scaler_updates=False, base_column=None, base_df_name=None):
        """Scaler class that bases the scaling on the base column

        Args:
            default_scaler_type (str, optional): _description_. Defaults to 'MinMaxScaler'.
            log_scaler_updates (bool, optional): _description_. Defaults to False.
            base_column (_type_, optional): _description_. Defaults to None.
        """        
        self._scalers = {}
        self._valid_scaler_types = ['MinMaxScaler', 'StandardScaler']
        self._default_scaler_type = default_scaler_type
        self._log_scaler_updates = log_scaler_updates
        self.base_column = base_column
        self.base_df_name = base_df_name
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
        cur_scaler = self._scalers[df_name]
        cur_scaler['base_column'] = self.base_column 
        cur_scaler['data'] = {}
        for var in df.select_dtypes('float64').columns:
            if not self._check_valid_scaler_type(scaler_type):
                return
            if scaler_type == 'StandardScaler':
                scaler = StandardScaler()
            if scaler_type == 'MinMaxScaler':
                scaler = MinMaxScaler()
            # Build the current scaler entry in the dictionary

            cur_scaler['data'][var] = {}
            cur_scaler['data'][var]['scaler'] = scaler

            # If the base column is set, fit the scaler to the base column
            if self.base_column is not None and self.base_df_name is not None:
                if var == self.base_column and df_name == self.base_df_name:
                    # Grab the min and max from the base column
                    scaler.fit(df[[var]])
                    cur_scaler['data'][var]['data_min'] = scaler.data_min_[0]
                    cur_scaler['data'][var]['data_max'] = scaler.data_max_[0]
                # if it is a different column or dataframe, fit the scaler to the base column
                else: 
                    # Different dataframe but same column
                    # Set the min and max to the values from the base column
                    # Set this scaler data min based on dataframe values 
                    scaler.data_min_ =  self._scalers[self.base_df_name]['data'][self.base_column]['data_min']
                    scaler.data_max_ =  self._scalers[self.base_df_name]['data'][self.base_column]['data_max']

                    # scaler.data_max_ =  df[var].max()

                    # Set the scaler scale based on the base column
                    scaler.scale_ = 1.0 / (self._scalers[self.base_df_name]['data'][self.base_column]['data_max'] - self._scalers[self.base_df_name]['data'][self.base_column]['data_min'])
                    # Set the scaler min based on the base column
                    scaler.min_ = -scaler.data_min_ * scaler.scale_
                    #scaler.max_ = 1 - scaler.data_max_ * scaler.scale_
                                        
                    
                    #try:
                        #scaler.fit(df[[var]])
                    #except Exception as e: 
                        #logging.error(f'Unable to fit scaler for dataframe {df_name} and column {var}\n\tException:{e}')
                # Different dataframe but same column
                # elif var == self.base_column and df_name != self.base_df_name:
                #     # Fit the scaler to the base column and base dataframe
                #     base_df = self._scalers[self.base_df_name]
                #     base_scaler = base_df['data'][self.base_column]['scaler']
                #     # Set the min and max from the df, since the min and max will be invalid to use, as it will already be scaled
                #     cur_scaler['data'][var]['data_min'] = df[var].min()
                #     cur_scaler['data'][var]['data_max'] = df[var].max()

                #     # Fit the current scaler to the min and max of the base scaler 
                #     scaler.fit(base_scaler.transform(df[[var]]))
                # # Different column, want to fit to the base dataframe and base colum
                # else:

                #     # Fit the scaler to the base column and base dataframe
                #     base_df = self._scalers[self.base_df_name]
                #     base_scaler = base_df['data'][self.base_column]['scaler']
                #     # Set the min and max from the df, since the min and max will be invalid to use, as it will already be scaled
                #     cur_scaler['data'][var]['data_min'] = df[var].min()
                #     cur_scaler['data'][var]['data_max'] = df[var].max()

                #     # Fit the current scaler to the min and max of the base scaler base column 
                #     scaler.fit(base_scaler.transform(df[[self.base_column]]))

            # If the base column is not set, fit the scaler to the current column
            else: 
                scaler.fit(df[[var]])
                cur_scaler['data'][var]['data_min'] = scaler.data_min_[0]
                cur_scaler['data'][var]['data_max'] = scaler.data_max_[0]
        # Update the scaler dictionary
        self._scalers[df_name] = cur_scaler
        self._log_scaler_update()
    def _log_scaler_update(self):
        """Logs the state of the scaler object
        """
        if self._log_scaler_updates:
            logging.info(f'####### Scaler Object State Updated #######\n {self.__str__()}')

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
            df2[var] = self._scalers[df_name]['data'][var]['scaler'].transform(df2[[var]])
        # Log the scaler object when created for debugging purposes
        self._log_scaler_update()
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
            df2[var] = self._scalers[df_name]['data'][var]['scaler'].inverse_transform(df2[[var]])
        self._log_scaler_update()
        return df2

    def __str__(self):
        ret = "########## Scaler Object ##########\n"
        ret += f"Default Scaler Type: {self._default_scaler_type}\n"
        ret += f"Base Column for Scaling: {self.base_column}\n"
        ret += f"Base Dataframe Name Used for Scaling: {self.base_df_name}\n"
        for key in self._scalers.keys():
            ret += f"Scaler for dataset: {key}\n"
            # Print metadata about the scaler, including min, max, and type of scaler
            for var in self._scalers[key]['data'].keys():
                ret += f"\tVar: {var}\n"
                for metadata in self._scalers[key]['data'][var].keys():
                    ret += f"\t\t{metadata}: {self._scalers[key]['data'][var][metadata]}\n"
        ret += "########## End Scaler Object ##########\n"

        return ret 