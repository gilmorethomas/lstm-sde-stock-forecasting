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

def plot_all_x_y_combinations(df, x_cols, y_cols, plot_type, output_dir, output_name, save_png=True, save_html=True):
    # Plot all x, y combinations
    if x_cols is None:
        logging.info("No x columns provided, using all columns")
        x_cols = df.columns
    if y_cols is None:
        logging.info("No y columns provided, using all columns")
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
            trace_name = output_file)
        finalize_plot(fig=fig, 
            filename=output_file, 
            output_dir=output_dir, 
            save_png=save_png, 
            save_html=save_html)
    

def plot(df, x_col, y_col, plot_type, trace_name, fig=None):
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
        logging.info(f"Adding plot {x_col} vs {y_col} to existing figure with name {trace_name}")
    else: 
        # Create a new plotly graph objects figure
        fig = go.Figure()
        fig.update_layout(
            title=f"{x_col} vs {y_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
        )
    if plot_type == 'scatter':
        # add a trace to the figure
        fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='markers', name=trace_name))
    elif plot_type == 'line':
        # add a line trace to the figure
        fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='lines', name=trace_name))
    elif plot_type == 'bar':
        # add a bar trace to the figure
        fig.add_trace(go.Bar(x=df[x_col], y=df[y_col], name=f"{x_col} vs {y_col}"))
    else: 
        logging.error(f"Plot type {plot_type} not implemented")
        return None
    return fig
    
def finalize_plot(fig, filename, output_dir, save_png=True, save_html=True):
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
    logging.info(f"Finalizing plot {filename}")
    fig.update_layout(
        title=f"{filename}",
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