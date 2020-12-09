import numpy as np
import iqplot
import bokeh.io
import bokeh
import scipy
import scipy.stats as st

def __ecdf(x, data):
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

def draw_bs_sample(data):
    '''Draw a boostrap sample from a 1D data set.'''
    return np.random.choice(data, size = len(data))

def draw_bs_reps(data, stat_fun, size=1):
    '''Draw boostrap replicates computed with stat_fun from 1D dataset'''
    return np.array([stat_fun(draw_bs_sample(data)) for _ in range (size)])

def draw_bs_reps_mean(data, size=1):
    '''Draw boostrap replicates of the mean from 1D data set'''
    out = np.empty(size)
    for i in range(size):
        out[i] = np.mean(draw_bs_sample(data))
    return out

def draw_bs_reps_test_stat(x, y, size=1):
    """
    Generate bootstrap replicates with Kolmogrov Smirnov
    as the test statistic.
    """
    out = np.empty(size)
    for i in range(size):
        out[i] = st.ks_2samp(draw_bs_sample(x), draw_bs_sample(y))[0]

    return out

def __L(x, epsilon, data):
    '''Calculates L(x) for a given x value, epsilon, and data set'''
    ecdf_vals = __ecdf(x, data)
    
    return [max(0, val - epsilon) for val in ecdf_vals]

def __U(x, epsilon, data):
    '''Calculates U(x) for a given x value, epsilon, and data set'''
    ecdf_vals = __ecdf(x, data)
    
    return [min(1, val + epsilon) for val in ecdf_vals]

def plot_conf_int(data, title, xlabel, color = 'tomato'):
    ''' plots an ECDF with confidence intervals 
    data : array
    contains raw data points from experiment
    title : string
    title for output graph
    xlabel : x axis label for output graph
    color : string (optional)
    if given, color for upper and lower confidence interval bounds
    
    Returns
    ________
    output : bokeh figure 
    '''
    x = np.linspace(0, max(data), 100)
    epsilon = np.sqrt(np.log(2/0.05)/(2 * len(data)))
    p = bokeh.plotting.figure(title=title, x_axis_label=xlabel, 
                   y_axis_label='ECDF', width = 400, height = 400)
    l = __L(x, epsilon, data)
    u = __U(x, epsilon, data) 
    p.circle(x, l, color = color)
    p.circle(x, u, color = color)
    # overlay with experimental ECDF
    iqplot.ecdf(data, p = p, conf_int = True)
    return p