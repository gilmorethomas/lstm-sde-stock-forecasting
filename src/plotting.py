import plotly.graph_objects as go
import pandas as pd
import numpy as np
import logging

class Plotting(): 
    def __init__(self, df, plot_type, output_dir): 
        self.df = df
        self.plot_type = plot_type
        self.output_dir = output_dir
    def plot(self):
        # Use plotly to plot each combination of columns in the dataframe 

        # You should save each of these plots as a png and html file in the output directory
        # The filenames should be the column names of the two columns being plotted

        # For example, if you are plotting the columns "A" and "B", the files should be named "A_B.png" and "A_B.html",
        # where A_B png is stored in self.output_dir/pgn and A_B.html is stored in self.output_dir/html

        # You should use the plot type to determine the type of plot to create (scatter, line, bar, etc.)
        
        raise NotImplementedError("This method is not implemented yet")