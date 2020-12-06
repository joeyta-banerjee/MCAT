import numpy as np
import pandas as pd
import scipy
import scipy.stats as st

def ecdf(x, data):
    """Give the value of an ECDF at arbitrary points x."""
    y = np.arange(len(data) + 1) / len(data)
    return y[np.searchsorted(np.sort(data), x, side="right")]

def AIC(params, log_likelihood_fun, data):
    L = log_likelihood_fun(*params, data);
    return -2*(L - len(params))

def predictive_regression(data, gen_function):
    single_samples = np.array([gen_function(*params, size = len(data))
                            for _ in range (size)])
    n_theor = np.arange(0, single_samples.max() + 1)

    ecdfs = np.array([ecdf(n_theor, sample) for sample in single_samples])
    ecdf_low, ecdf_high = np.percentile(ecdfs, [2.5, 97.5], axis=0)
    return ecdf_low, ecdf_high



def QQ_plot(data, gen_function, params, size = 1000, axis_label = None, title = None):
    single_samples = np.array([gen_function(*params, size = len(data))
                            for _ in range (size)])
    p = bebi103.viz.qqplot(
    data=data,
    samples=single_samples,
)
    if(axis_label != None):
        p.xaxis.axis_label = axis_label
        p.yaxis.axis_label = axis_label
    if(title != None):
        p.title.text = title
    return p
