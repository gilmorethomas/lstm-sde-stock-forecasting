import logging 
from os import path, makedirs
# import plotting module 
from plotting import Plotting


class AnalysisManager(): 
    # Create an analysis manager whose job is to manage the analysis objects
    # It should store a list of analysis objects and call the methods of the analysis objects
    # It should also take a raw directory and an output directory
    # Raw directory is where the raw data is stored
    # Output directory is where the output data is stored


    def __init__(self, raw_dir, output_dir, **kwargs):
        import pdb; pdb.set_trace()
        self.analysis_objects_dict = {}
        self.raw_dir = raw_dir
        self.output_dir = output_dir

    def set_preprocessing_callback(self, preprocessing_callback):
        self.preprocessing_callback = preprocessing_callback
        # set the preprocessing callback for the analysis objects

    
    def add_analysis_objs(self, analysis_dict): 
        # add a dictionary of analysis objects, with key being the name of the analysis object
        # and the value being the dataset
        for dataset_name, dataset_df in analysis_dict.items(): 
            logging.info(f"Creating analysis object for {dataset_name}")
            analysis = Analysis(dataset_name, dataset_df, path.join(self.output_dir, dataset_name), preprocessing_callback=self.preprocessing_callback)
            self.analysis_objects_dict[dataset_name] = analysis
    
    def preprocess_datasets(self):
        # preprocess the datasets for each analysis object
        logging.info("Preprocessing datasets")
        for analysis_name, analysis_obj in self.analysis_objects_dict.items(): 
            analysis_obj.preprocess_dataset()

    def get_analysis_objs(self):
        return self.analysis_objects_dict
    def run_analysis(self, run_descriptive=True, run_predictive=True):
        # run analysis on each object
        for analysis_name, analysis_obj in self.analysis_objects_dict.items(): 
            logging.info(f"Running analysis for {analysis_name}")
            analysis_obj.run_analysis(run_descriptive, run_predictive)
    def set_models_for_analysis_objs(self, models_dict, analysis_objs_to_use=None): 
        # set models for each analysis object
        # if analysis_objs_to_use is None, set models for all analysis objects,
        # otherwise, set models for the analysis objects in the list
        if analysis_objs_to_use is None: 
            analysis_objs_to_use = self.analysis_objects
        for analysis in analysis_objs_to_use:
            analysis._set_models(models_dict)


    def _run_all_in_one_plot(self):
        # run an all-in-one plot with all the datasets
        raise NotImplementedError("This method is not implemented yet")
            
class Analysis(): 
    def __init__(self, dataset_name, dataset_df, output_directory, preprocessing_callback=None): 
        self.dataset_name = dataset_name
        self._raw_dataset_df = dataset_df
        self.preprocessing_callback = preprocessing_callback
        self.output_directory = output_directory
        if not(path.exists(output_directory)):
            logging.info(f"Creating output directory {output_directory}")
            makedirs(output_directory)
    def preprocess_dataset(self):
        # preprocess the dataset, using the preprocessing_callback if provided
        logging.info(f"Preprocessing dataset for {self.dataset_name} using {self.preprocessing_callback}")
        if self.preprocessing_callback is not None: 
            self.dataset_df = self.preprocessing_callback(self._raw_dataset_df)

    def run_analysis(self, run_descriptive=True, run_predictive=True): 
        if run_descriptive: 
            self.run_descriptive()
        if run_predictive: 
            self.run_predictive()
    def run_descriptive(self):
        # run descriptive analytics
        self.report_stats() 
        self.run_plots(plot_types=["line", "scatter"])

    def report_stats(self):
        # report stats for each analysis object
        # Report mean, standard deviation, median, min, max, and number of observations
        # build dataframe for each column's stats
        stats_df = self.dataset_df.describe().T
        # stats_df["skew"] = self.dataset_df.skew()
        # stats_df["kurtosis"] = self.dataset_df.kurtosis()
        # stats_df["missing_values"] = self.dataset_df.isnull().sum()
        # stats_df["unique_values"] = self.dataset_df.nunique()
        # stats_df["dtype"] = self.dataset_df.dtypes
        # stats_df["range"] = stats_df["max"] - stats_df["min"]
        # stats_df["mean_absolute_deviation"] = self.dataset_df.mad()
        # stats_df["variance"] = self.dataset_df.var()
        # stats_df["standard_deviation"] = self.dataset_df.std()
        # stats_df["coefficient_of_variation"] = stats_df["standard_deviation"] / stats_df["mean"]
        # stats_df["interquartile_range"] = self.dataset_df.quantile(0.75) - self.dataset_df.quantile(0.25)
        # stats_df["outliers"] = self.dataset_df[(self.dataset_df < (self.dataset_df.quantile(0.25) - 1.5 * stats_df["interquartile_range"])) | (self.dataset_df > (self.dataset_df.quantile(0.75) + 1.5 * stats_df["interquartile_range"]))].count()
        # stats_df["outlier_percentage"] = stats_df["outliers"] / stats_df["count"]
        # stats_df["z_score"] = (self.dataset_df - self.dataset_df.mean()) / self.dataset_df.std()
        output_file = path.join(self.output_directory, f'{self.dataset_name}_stats.csv')
        logging.info(f"Stats for {self.dataset_name} being written to {output_file}")

        stats_df.to_csv(output_file)

    def run_plots(self, plot_types):
        # run plots for each analysis object
            # plot the dataset using the analysis class plotting function 
        self.plot_dataset(plot_types)
        # also create an all-in-one plot with all the datasets, adding an extra variable that is 
        # self._run_all_in_one_plot()
        
        # Run descriptive time series analysis 
        
    def run_predictive(self):
        # run predictive analytics
        ...
        raise NotImplementedError("This method is not implemented yet")

    def _set_models(self, models_dict):
        # set models for the analysis object
        raise NotImplementedError("This method is not implemented yet")

    def _get_models(self):
        # get models for the analysis object
        raise NotImplementedError("This method is not implemented yet")

    def plot_dataset(self, plot_types):
        # Use the plotting class to plot the entirety of the dataset (all columns as options)
        for plot_type in plot_types:
            plotter = Plotting(self.dataset_df, plot_type, self.output_directory)
            plotter.plot()