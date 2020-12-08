import pandas as pd
import numpy as np
import os, sys

import iqplot
import bokeh.io

def categorical_plot(df, variable, cats, format = "ECDF"):
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

    Returns
    _________
    p : bokeh figure
    Figure containing all of the plots, use bokeh.io.show() to
    see figure
    '''
    if (format == "ECDF"):
        p = iqplot.ecdf(df, q = variable, cats = cats)
    elif(format == "stripbox"):
        p = iqplot.stripbox(df, q = variable, cats = cats)
    p.title.text = format + " of " + variable + " separated by " + cats
    return p
