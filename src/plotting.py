import plotly.io as pio
import plotly.graph_objects as go
from os import path, makedirs
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

def plot_all_x_y_combinations(df, x_cols, y_cols, plot_type, output_dir, output_name, save_png=True, save_html=True):
    # Plot all x, y combinations
    if x_cols is None:
        logging.debug("No x columns provided, using all columns") 
        x_cols = df.columns
    if y_cols is None:
        logging.debug("No y columns provided, using all columns") 
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

def plot(df, x_col, y_col, plot_type, trace_name, fig=None, color=None, groupbyCols=None):
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
        logging.debug(f"Adding plot {x_col} vs {y_col} to existing figure with name {trace_name}") 
    else: 
        # Create a new plotly graph objects figure
        fig = go.Figure()
        fig.update_layout(
            title=f"{x_col} vs {y_col}",
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
        if groupbyCols is not None:
            data = [] 
            for group, groupdf in df.groupby(groupbyCols): 
                # add a bar trace to the figure
                # get rid of the group name in the x columns 
                groupdf = groupdf.copy()
                #import pdb; pdb.set_trace()
                #groupdf.rename(columns={x_col: x_col.replace('gbm', '').replace('lstm_sde', '').replace('lstm', '')}, inplace=True)
                data.append(go.Bar(x=groupdf[x_col], 
                    y=groupdf[y_col], 
                    marker=dict(color=color), 
                    name=' '.join(group), 
                    cliponaxis=False,))
                
            layout = go.Layout(barmode='group')
            fig = go.Figure(data=data, layout=layout)
        else: 
            # add a bar trace to the figure
            fig.add_trace(go.Bar(x=df[x_col], 
                y=df[y_col], 
                marker=dict(color=color), 
                name=f"{x_col} vs {y_col}"))
    else: 
        logging.error(f"Plot type {plot_type} not implemented")
        return None
    return fig

def plot_multiple_y_cols(df, x_col, y_cols, plot_type, output_dir, output_name, title, fig=None, trace_name_to_prepend = '', save_png=True, save_html=True, make_title_replacements=True, finalize=True):
    """Plots a single x column against multiple y columns. This will plot all y columns on the same plot

    Args:
        df (pd.DataFrame): The DataFrame to plot
        x_col (str): The x column to plot
        y_cols (list): The y columns to plot
        plot_type (str): The type of plot to make
        output_dir (str): The output directory to save the plot
        output_name (str): The output name of the plot
        title (str): The title of the plot
        fig (plotly.graph_objects.Figure, optional): The figure to add the plot to. Defaults to None. If you pass a figure, it will add the plot to the figure and return the figure.
        trace_name_to_prepend (str, optional): The string to prepend to the trace name. Defaults to ''.
        save_png (bool, optional): Whether or not to save png. Defaults to True.
        save_html (bool, optional): Whether or not to save html. Defaults to True.
        finalize (bool, optional): Whether or not to finalize (save) the plot. Defaults to True.
    """    
    if y_cols is None:
        logging.debug("No y columns provided, using all columns") 
        y_cols = df.columns

    for y_col in y_cols:
        if y_col not in df.columns:
            logging.error(f"Column {y_col} not in DataFrame")
            continue
        output_file = f"{output_name}_{x_col}_vs_{y_col}"
        if make_title_replacements:
            trace_name = _make_title_replacements(f'{trace_name_to_prepend}_{y_col}')
        else: 
            trace_name = trace_name
        fig = plot(df=df,
            x_col=x_col, 
            y_col=y_col, 
            plot_type=plot_type,
            trace_name = trace_name,
            fig = fig)
        if finalize:
            finalize_plot(fig=fig, 
                title=_make_title_replacements(title),
                filename=f'{output_file}_{plot_type}', 
                output_dir=output_dir, 
                save_png=save_png, 
                save_html=save_html)
    return fig 

def finalize_plot(fig, title, filename, output_dir, save_png=True, save_html=True, make_title_replacements=True, plot_type=None, xaxis_title=None, yaxis_title=None):
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
    if plot_type is not None and plot_type == 'bar':
        fig.update_layout(barmode='group')
        #fig.update_xaxes(autorangeoptions={'maxallowed': 'max'})
        fig.update_xaxes(tickangle=45, automargin=True)
        #fig.updatelayout(margin=dict(l=100, r=100, t=100, b=100))
        
        # Set the y axis limits to be the 75th percentile of data + 20% of the range of the 25th to 75th percentile
        # This is to make sure that the y axis is not too high
        import numpy as np 
        #y_75 = np.percentile(fig.data[0].y, 75)
        #y_25 = np.percentile(fig.data[0].y, 25)
        #if y_25 < 0:
        #    yaxis_range1 = 2*y_25
        #else:
        #    yaxis_range1 = .5*y_25
        #@if y_75 < 0:
        #    yaxis_range2 = 2*y_75
        
        #fig.update_yaxes(range=[yaxis_range1, y_75*2])

    if xaxis_title is not None:
        fig.update_xaxes(title=xaxis_title)
    if yaxis_title is not None:
        fig.update_yaxes(title=yaxis_title)
    html_dir = path.join(output_dir, 'html')
    png_dir = path.join(output_dir, 'png')
    if not path.exists(output_dir):
            makedirs(output_dir)
    if save_html and not path.exists(html_dir):
            makedirs(html_dir)
    if save_png and not path.exists(png_dir):
            makedirs(png_dir)
    if make_title_replacements: 
        title = _make_title_replacements(title)
    if save_png:
        fig.write_image(path.join(png_dir, f'{filename}.png'))
    if save_html: 
        fig.write_html(path.join(html_dir, f'{filename}.html'))