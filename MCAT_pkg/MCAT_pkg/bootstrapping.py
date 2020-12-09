import numpy as np
import iqplot
import bokeh.io
import bokeh
import scipy
import bebi103
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
    '''Draw a boostrap sample from a 1D data set
    Parameters 
    __________
    data : array
        input data to sample from
    
    Returns
    __________
    output: array
        new dataset drawn from the original dataset
    '''
    return np.random.choice(data, size = len(data))

def draw_bs_reps(data, stat_fun, size=1):
    '''Draw boostrap replicates computed with stat_fun from 1D dataset
    Parameters 
    __________
    data : array
        input data to sample from
    stat_fun : function
        statistical function to calculate on each dataset replicate
    size : int, optional, default = 1
        number of dataset replicates to generate
    
    Returns
    __________
    output: array
        test statistic for each of size bootstrap replicate(s)
    '''
    return np.array([stat_fun(draw_bs_sample(data)) for _ in range (size)])

def draw_bs_reps_mean(data, size=1):
    '''Draw boostrap replicates of the mean from 1D data set
    Parameters 
    __________
    data : array
        input data to sample from
    size : int, default = 1
        number of datasets to draw
    
    Returns
    __________
    output: array
        mean of new dataset drawn from the original dataset
    '''
    out = np.empty(size)
    for i in range(size):
        out[i] = np.mean(draw_bs_sample(data))
    return out

def draw_bs_reps_test_stat(x, y, size=1):
    """
    Generate bootstrap replicates with Kolmogrov Smirnov
    as the test statistic
    Parameters 
    __________
    x : array
        input dataset 1 to sample from
    y : array
        input dataset 2
    
    Returns
    __________
    output: array
        Kolmogorov Smirnov test statistic for each trial
    """
    out = np.empty(size)
    for i in range(size):
        out[i] = st.ks_2samp(draw_bs_sample(x), draw_bs_sample(y))[0]

    return out

def __L(x, epsilon, data):
    '''Calculates L(x) for a given x value, epsilon, and data set
    
    Parameters
    ___________
    x : array
        points to calculate L(x) for
    epsilon : float
        epsilon value to use in equation
    data : array
        raw data
        
    Returns
    ___________
    output : array
        value of L(x) for each point in x
    '''
    ecdf_vals = __ecdf(x, data)
    
    return [max(0, val - epsilon) for val in ecdf_vals]

def __U(x, epsilon, data):
    '''Calculates U(x) for a given x value, epsilon, and data set
    x : array
        points to calculate U(x) for
    epsilon : float
        epsilon value to use in equation
    data : array
        raw data
        
    Returns
    ___________
    output : array
        value of U(x) for each point in x
    '''
    ecdf_vals = __ecdf(x, data)
    
    return [min(1, val + epsilon) for val in ecdf_vals]

def plot_conf_int(data, title, xlabel, color = 'green', palette = ["limegreen"]):
    ''' plots an ECDF with confidence intervals 
    data : array
        contains raw data points from experiment
    title : string
        title for output graph
    xlabel : string
        x axis label for output graph
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
    p.line(x, l, color = color)
    p.line(x, u, color = color)
    # overlay with experimental ECDF
    iqplot.ecdf(data, p = p, conf_int = True, palette = palette)
    return p

def viz_compare_conf_int(data, data2, fun = np.mean, xlabel = "mean", title = "Comparison of Confidence Intervals",
                         label1 = "sample1", label2 = "sample2", size = 1000, color = "dimgray"):
    """Function to visualize overlap between two confidence intervals
    Parameters
    _________
    
    data : array
        first dataset
    data2 : array
        second dataset
    fun : statistical function (optional), default is mean
        if given, statistical function to calculate for each dataset
    label : string (optional), default is "mean"
        if given, used as axis label for confidence intervals graph
    size : int (optional), default = 1000
        number of samples for each dataset
    color : string (optional), default = "dimgray"
        if given, color of intervals in visualization
        
    Returns
    _________
    output : bokeh figure containing the visualization
        
    """
    bs_reps = draw_bs_reps(data, fun, size = size)
    bs_reps2 = draw_bs_reps(data2, fun, size = size)
    mean = np.mean(bs_reps)
    mean2 = np.mean(bs_reps2)
    conf_int = np.percentile(bs_reps, [2.5, 97.5])
    conf_int2 = np.percentile(bs_reps2, [2.5, 97.5])
    dict1 = {"estimate": mean, "conf_int": conf_int, "label" : label1}
    dict2 = {"estimate":mean2, "conf_int":conf_int2, "label" : label2}
    p = bebi103.viz.confints([dict1, dict2],  
                             marker_kwargs = {"color":color},
                            line_kwargs = {"color":color})
    p.xaxis.axis_label = xlabel
    p.title.text = title
    return p
