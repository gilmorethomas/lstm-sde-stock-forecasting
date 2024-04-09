import plotly.io as pio
import plotly.express as px
import pandas as pd
import numpy as np
import os
import logging
from itertools import product 
pio.renderers.default='browser'

class Plotting(): 
    def __init__(self, df, plot_type, output_dir): 
        self.df = df
        self.plot_type = plot_type
        self.output_dir = output_dir
        
        # Use plotly to plot each combination of columns in the dataframe 

        # You should save each of these plots as a png and html file in the output directory
        # The filenames should be the column names of the two columns being plotted

        # For example, if you are plotting the columns "A" and "B", the files should be named "A_B.png" and "A_B.html",
        # where A_B png is stored in self.output_dir/pgn and A_B.html is stored in self.output_dir/html

        # You should use the plot type to determine the type of plot to create (scatter, line, bar, etc.)
        
    def plot(self):
         if not os.path.exists(self.output_dir):
             os.makedirs(self.output_dir)
         if not os.path.exists(os.path.join(self.output_dir, 'png')):
             os.makedirs(os.path.join(self.output_dir, 'png'))
         if not os.path.exists(os.path.join(self.output_dir, 'html')):
             os.makedirs(os.path.join(self.output_dir, 'html'))
            
         for col1, col2 in product(self.df.columns, self.df.columns): 
            filename = f"{col1}_{col2}"

            fig = self.get_plotType(self.df, col1, col2, self.plot_type)
            # TODO: add DataFrame name in this.
            try:
                fig.write_html(f"{self.output_dir}/html/{self.plot_type}_{filename}.html", auto_open=False)
                # TODO: Get this write image to work, not sure why it isn't working
                #fig.write_image(f"{self.output_dir}/png/{filename}.png", auto_open=False)
                print(f"Plotted and saved: {filename}")
            except Exception as e:
                print(f"Error plotting {filename}: {e}")
                        
    def get_plotType(self, df, col1, col2, plot_type):
        # Determine plot type based on data type of columns
        # TODO: Need to pass down DataFrame Name to put into Title
        
        if plot_type == 'scatter':
            fig = px.scatter(df, x=col1, y=col2, title=f"{col1} vs {col2}")
        elif plot_type == 'line':
            fig = px.line(df, x=col1, y=col2, title=f"{col1} vs {col2}")
        elif plot_type == 'bar':
            fig = px.bar(df, x=col1, y=col2, title=f"{col1} vs {col2}")
        # Add more plot types as needed

        return fig
    
if __name__ == "__main__":
    # Example dataframe
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [2, 3, 4, 5, 6],
        'C': ['a', 'b', 'c', 'd', 'e'],
        'D': [10, 20, 30, 40, 50]
    }
    df = pd.DataFrame(data)
    
    output_dir = 'joey_output'
    
    plot_type = 'scatter'

    # Example usage of Plotter class
    plotter = Plotting(df, plot_type, output_dir)
    plotter.plot()
        