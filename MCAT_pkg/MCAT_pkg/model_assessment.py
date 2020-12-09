import numpy as np
import pandas as pd
import scipy
import scipy.stats as st
import bebi103
import bokeh.io 

def ecdf(x, data):
    """Give the value of an ECDF at arbitrary points x

    Parameters
    __________
    x : array
    points to calculate ECDF for

    data : array
    input data to generate the ECDF based on

    Returns
    __________
    output : array of ECDF values for each point in x
    """
    y = np.arange(len(data) + 1) / len(data)
    return y[np.searchsorted(np.sort(data), x, side="right")]

def AIC(params, log_likelihood_fun, data):
    """Calculates the AIC (akaike criterion)
    Parameters
    _________
    params : tuple
    MLE parameters for distribution

    log_likelihood_fun : function
    calculates the log likelihood for the desired distribution

    data : array
    empirical dataset to calculate log likelihood with respect to

    Returns
    _________
    output : float
    AIC value
    """

    L = log_likelihood_fun(params, data);
    return -2*(L) + 2*len(params)

def predictive_ecdf(data, gen_function, params, size = 1000, title = None, xlabel = "Time to Catastrophe (s)", 
                    color = "blue", data_color = "gray"):
    """ Compares ECDF of theoretical distribution to experimental
    Parameters
    __________
    data : array
    input data array

    gen_function : function
    generative function to sample from

    params : tuple
    parameters to use for generative function

    size : int (optional), default = 1000
    number of samples to draw from the generative distribution
    
    palette : list (optional), default = ["blue"]
    if given, used as palette argument to bebi103.viz.fill_between
    """
    single_samples = np.array([gen_function(*params, size = len(data))
                            for _ in range (size)])
    n_theor = np.arange(0, single_samples.max() + 1)

    p = bebi103.viz.predictive_ecdf(
    samples = single_samples,
    data = data,
    discrete = True,
    color = color,
    data_color=data_color,
    x_axis_label=xlabel,
    y_axis_label="ECDF",
    )
    if(title != None):
        p.title.text = title
    return p



def QQ_plot(data, gen_function, params, size = 1000, axis_label = None, title = None, color = "green"):
    """ creates a QQ_plot comparing the empirical and theoretical value

    Parameters
    __________
    data : array
    input data

    gen_function : function
    function to generate points from the desired distribution

    params : tuple
    MLE parameters

    size : int (optional), default = 1000
    number of samples to generate from the generative distribution

    axis_label : string (optional)
    if given, axis_label is used as the label for both axes of the returned
    plot

    title : string (optional)
    if given, used as the title for the returned plot
    
    palette : list (optional)
    if given, used as the argument to palette in bebi103.viz.qqplot

    Returns
    ________
    output : p (bokeh figure)
    """
    single_samples = np.array([gen_function(*params, size = len(data))
                            for _ in range (size)])
    p = bebi103.viz.qqplot(
    data=data,
    samples=single_samples,
    patch_kwargs = {"color":color},
    line_kwargs = {"color":color}
    )
    if(axis_label != None):
        p.xaxis.axis_label = axis_label
        p.yaxis.axis_label = axis_label
    if(title != None):
        p.title.text = title
    return p
