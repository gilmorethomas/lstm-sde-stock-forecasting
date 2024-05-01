import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
import os
from lstm_logger import logger as logging
from itertools import product 
pio.renderers.default='browser'

class PlottingConstants():
    """ Class to hold constants for plotting

    """
    @staticmethod
    def default_color_sequence(i):
        return PlottingConstants._default_color_sequence[i % len(PlottingConstants._default_color_sequence)]
    
    _default_color_sequence = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf'   # blue-teal
    ]
    debug_plots = False

def plot_all_x_y_combinations(df, x_cols, y_cols, plot_type, output_dir, output_name, save_png=True, save_html=True, debug_plots=PlottingConstants.debug_plots):
    # Plot all x, y combinations
    if x_cols is None:
        logging.info("No x columns provided, using all columns") if debug_plots else None
        x_cols = df.columns
    if y_cols is None:
        logging.info("No y columns provided, using all columns") if debug_plots else None
        y_cols = df.columns
    for x_col, y_col in product(x_cols, y_cols):
        if x_col == y_col:
            continue
        if x_col not in df.columns or y_col not in df.columns:
            logging.error(f"Column {x_col} or {y_col} not in DataFrame")
            continue
        output_file = f"{output_name}_{x_col}_vs_{y_col}"
        fig = plot(df=df,
            x_col=x_col, 
            y_col=y_col, 
            plot_type=plot_type,
            trace_name = _make_title_replacements(output_file))
        title = _make_title_replacements(f'{x_col} vs. {y_col}')
        finalize_plot(fig=fig, 
            title=title,
            filename=f'{output_file}_{plot_type}', 
            output_dir=output_dir, 
            save_png=save_png, 
            save_html=save_html)
def _make_title_replacements(title):
    # Replace 'newline' with a newline character
    title = title.replace('newline' , '<br>')
    # Replace underscores with space 
    title = title.replace('_', ' ')
    # Capitalize the first letter of each word
    title = title.title()
    # Replace 'Vs' with 'vs'
    title = title.replace('Vs', 'vs')
    return title

def plot_multiple_dfs(trace_name_df_dict, title, x_cols, y_cols, plot_type, output_dir, output_name, save_png=True, save_html=True, add_split_lines=False):
    """Plots multiple dataframes with the same x and y columns. Overlays the plots on the same figure

    Args:
        trace_name_df_dict (dict(trace_name : trace dataframe)): dictionary of traces and dfs to plot
        x_cols (list): _description_
        y_cols (list): _description_
        plot_type (str): _description_
        output_dir (str, path-like): _description_
        output_name (str): _description_
        save_png (bool, optional): Whether or not to save png. Defaults to True.
        save_html (bool, optional): Whether or not to save png. Defaults to True.
    """    
    if trace_name_df_dict is None or len(trace_name_df_dict) == 0:
        logging.error("No dataframes provided")
        return
    # Assert that all dataframes have the same columns for columns that are specified in x_cols and y_cols
    for df in trace_name_df_dict.values():
        assert all([col in df.columns for col in x_cols + y_cols]), "All x_cols and y_cols must be in the DataFrame"
    
    for x_col, y_col in product(x_cols, y_cols):
        # Set the figure to None to create a new figure for each x, y combination, but add trace for each dataframe
        fig = None
        max_y = max([df[y_col].max() for df in trace_name_df_dict.values()])
        min_y = min([df[y_col].min() for df in trace_name_df_dict.values()])
        num_traces = 0
        for trace_name, df in trace_name_df_dict.items():
            #logging.info(f"Plotting {x_col} vs {y_col} for {trace_name}")
            if x_col == y_col:
                continue
            trace_name = _make_title_replacements(trace_name)
            fig = plot(df=df,
                fig=fig,
                x_col=x_col, 
                y_col=y_col, 
                plot_type=plot_type,
                color=PlottingConstants.default_color_sequence(num_traces), 
                trace_name = trace_name)
            if add_split_lines:
                # Add a vline at the end of each dataframe to split the traces
                fig.add_trace(go.Scatter(x=[df[x_col].iloc[-1], df[x_col].iloc[-1]],
                    y=[min_y, max_y],
                    mode='lines',
                    # Get the color of the trace just added and match that color
                    line=dict(color=PlottingConstants.default_color_sequence(num_traces), width=2, dash='dash'),
                    name=f'{trace_name} Split Line'))
            num_traces += 1
        out_title = _make_title_replacements(f'{title} newline {x_col} vs. {y_col}')

        finalize_plot(fig=fig, title=out_title, filename=f'{output_name}_{x_col}_vs_{y_col}' , output_dir=output_dir, save_png=save_png, save_html=save_html)

def plot(df, x_col, y_col, plot_type, trace_name, fig=None, color=None, debug_plots=PlottingConstants.debug_plots):
    """Plots a single x, y combination
    
    Args:
        x_col (str): The x column to plot
        y_col (str): The y column to plot
        plot_type (str): The type of plot to make
        output_dir (str): The output directory to save the plot
        output_name (str): The output name of the plot
        fig (plotly.graph_objects.Figure, optional): The figure to add the plot to. Defaults to None. If you pass a figure, it will add the plot to the figure and return the figure.
    """
    if fig is not None: 
        logging.info(f"Adding plot {x_col} vs {y_col} to existing figure with name {trace_name}") if debug_plots else None
    else: 
        # Create a new plotly graph objects figure
        fig = go.Figure()
        fig.update_layout(
            title=f"{x_col} vs {y_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
        )
    if plot_type == 'scatter':
        # add a trace to the figure, using markers and color 
        fig.add_trace(go.Scatter(x=df[x_col],
            y=df[y_col], 
            mode='markers', 
            marker=dict(color=color), 
            name=trace_name))
        
    elif plot_type == 'line':
        # add a line trace to the figure, using lines instead of markers
        fig.add_trace(go.Scatter(x=df[x_col],
            y=df[y_col], 
            mode='lines', 
            marker=dict(color=color), 
            name=trace_name))    
    elif plot_type == 'bar':
        # add a bar trace to the figure
        fig.add_trace(go.Bar(x=df[x_col], 
            y=df[y_col], 
            marker=dict(color=color), 
            name=f"{x_col} vs {y_col}"))
    else: 
        logging.error(f"Plot type {plot_type} not implemented")
        return None
    return fig
    
def finalize_plot(fig, title, filename, output_dir, save_png=True, save_html=True):
    """Writes a plot to a png and html file

    Args:
        fig (plotly.graph_objects.Figure): The plotly figure to save
        filename (str, path-like): The filename to save the plot as
        output_dir (str, path-like): The directory to save the plot in
        save_png (bool, optional): Whether or not to save png. Defaults to True.
        save_html (bool, optional): Whether or not to save html. Defaults to True.

    Returns:
        _type_: _description_
    """     
    if fig is None: 
        logging.error("No figure provided")
    #logging.info(f"Finalizing plot {filename}")
    fig.update_layout(
        title=title,
        template='presentation',  # Use dark theme
        hovermode='closest',  # Show hover information for the closest point
    )

    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    if save_html and not os.path.exists(os.path.join(output_dir, 'html')):
            os.makedirs(os.path.join(output_dir, 'html'))
    if save_png and not os.path.exists(os.path.join(output_dir, 'png')):
            os.makedirs(os.path.join(output_dir, 'png'))
    
    if save_png:
        fig.write_image(f"{output_dir}/png/{filename}.png")
    if save_html: 
        fig.write_html(f"{output_dir}/html/{filename}.html")