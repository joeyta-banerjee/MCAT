import pandas as pd
import numpy as np
import os, sys

import iqplot
import bokeh.io

def categorical_plot(df, variable, cats, format = "ECDF", conf_int = False, palette = ["blue"]):
    ''' Plots the ECDF of times separated by concentration

    Parameters
    ___________
    df : pandas DataFrame
    Contains univariate data to be plotted

    variable : str
    name of column in df to be used as variable

    cats : str
    column name to separate categories by

    format : str (optional)
    type of graph to plot. options are ECDF, stripbox
    default : ECDF
    
    conf_int : bool (optional)
    if given and plot type is ECDF, conf_int is the value for the conf_int keyword argument in iqplot

    Returns
    _________
    p : bokeh figure
    Figure containing all of the plots, use bokeh.io.show() to
    see figure
    '''
    if (format == "ECDF"):
        p = iqplot.ecdf(df, q = variable, cats = cats, conf_int = conf_int, palette = palette)
    elif(format == "stripbox"):
        p = iqplot.stripbox(df, q = variable, cats = cats, palette = palette)
    p.title.text = format + " of " + variable + " separated by " + cats
    return p
