import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        
         LastNameInFilePath = os.path.split(self.output_dir)[-1];
         if not os.path.exists(self.output_dir):
             os.makedirs(self.output_dir)
# =============================================================================
#          if not os.path.exists(os.path.join(self.output_dir, 'png')):
#              os.makedirs(os.path.join(self.output_dir, 'png'))
# =============================================================================
         if not os.path.exists(os.path.join(self.output_dir, 'html')):
             os.makedirs(os.path.join(self.output_dir, 'html'))
         if self.plot_type == 'scatter':
           test = self.df.columns[0];
           StackedPlot = self.get_subplot(self.df, test, self.plot_type, LastNameInFilePath);
           StackedPlot.write_html(f"{self.output_dir}/html/{self.plot_type}_StackedPlot.html", auto_open=False)
            
         #for col1, col2 in product(self.df.columns, self.df.columns): 
         ListOfcolumns = self.df.columns.tolist()
         col1 = ListOfcolumns[0];
         for col2 in ListOfcolumns[1:]:
             if col1 == col2:
                 
                 continue  # Skip this iteration
             filename = f"{col1}_{col2}"
                         
             fig = self.get_plotType(self.df, col1, col2, self.plot_type, LastNameInFilePath)
             # TODO: add DataFrame name in this.
             try:
                fig.write_html(f"{self.output_dir}/html/{self.plot_type}_{filename}.html", auto_open=False)
                # TODO: Get this write image to work, not sure why it isn't working
                #fig.write_image(f"{self.output_dir}/png/{filename}.png", auto_open=False)
                print(f"Plotted and saved: {LastNameInFilePath} {self.plot_type} {filename}")
             except Exception as e:
                 print(f"Error plotting {LastNameInFilePath} {filename}: {e}")
                        
    def get_plotType(self, df, col1, col2, plot_type, LastNameInFilePath):
        # Determine plot type based on data type of columns
        # TODO: Need to pass down DataFrame Name to put into Title
        
        if plot_type == 'scatter':
            fig = px.scatter(df, x=col1, y=col2, title=f"{col1} vs {col2}")
        elif plot_type == 'line':
            fig = px.line(df, x=col1, y=col2, title=f"{col1} vs {col2}")
        elif plot_type == 'bar':
            fig = px.bar(df, x=col1, y=col2, title=f"{col1} vs {col2}")
        # Add more plot types as needed
        
        fig.update_layout(
            title=f"{LastNameInFilePath}: {col1} vs {col2}",
            xaxis_title=col1,
            yaxis_title=col2,
            template='plotly_dark',  # Use dark theme
            hovermode='closest',  # Show hover information for the closest point
            legend=dict(
                orientation='h',  # Horizontal legend
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
        )

        return fig
    
    def get_subplot(self, df, col1, plot_type, LastNameInFilePath):
        
        ListOfNames = df.columns.tolist();
        
        fig = make_subplots(rows = len(ListOfNames)-1, 
                            cols = 1, 
                            shared_xaxes=True,
                            vertical_spacing=0.02)
                            #subplot_titles= [col1 + " vs " + s for s in ListOfNames[1:]])
        RowCnt = 0;
        for col2 in ListOfNames:
            if col1 == col2:
                continue
            RowCnt= RowCnt + 1;
            fig.add_trace(go.Scatter(x=df[col1], y=df[col2], 
                                     mode='markers', 
                                     name=f"Plot {RowCnt}: {col1} vs {col2}"),
                                     row = RowCnt, 
                                     col = 1)
            fig.update_traces(marker=dict(size = 1))
            fig.update_yaxes(title_text=col2, row=RowCnt, col=1)
            
        fig.update_xaxes(title_text="Date", row=RowCnt, col=1)
        fig.update_layout(title_text= LastNameInFilePath +  " Data",
                          showlegend=False,
                          template='plotly_dark')
        # Show the plot
        return fig
# =============================================================================
# if __name__ == "__main__":
#     # Example dataframe
#     data = {
#         'A': [1, 2, 3, 4, 5],
#         'B': [2, 3, 4, 5, 6],
#         'C': ['a', 'b', 'c', 'd', 'e'],
#         'D': [10, 20, 30, 40, 50]
#     }
#     df = pd.DataFrame(data)
#     
#     output_dir = 'joey_output'
#     
#     plot_type = 'scatter'
# 
#     # Example usage of Plotter class
#     plotter = Plotting(df, plot_type, output_dir)
#     plotter.plot()
# =============================================================================
        