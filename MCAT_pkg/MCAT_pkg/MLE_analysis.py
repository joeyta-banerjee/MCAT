import numpy as np
import pandas as pd
import warnings
import scipy
import scipy.stats as st


def log_like_iid_gamma_log_params(params, t):
    """Log likelihood for i.i.d. Gamma measurements with
    input being parameters and data 

    Parameters
    ----------
    params : array
        parameters alpha and b.
    t : array
        Array of input data (times)

    Returns
    -------
    output : float
        Log-likelihood.
    """
    alpha, b = params
    if(alpha <= 0 or b <= 0):
        return -np.inf
    return np.sum(st.gamma.logpdf(t, alpha, scale = 1/b))

rg = np.random.default_rng(3252)
def mle_iid_gamma(t):
    """Perform maximum likelihood estimates for parameters for i.i.d.
    gamma measurements, parametrized by alpha, b=1/beta
    
    Parameters
    __________
    t : array
        data array

    Returns
    __________
    output : tuple
        Parameter estimates for alpha and beta
    
    
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, t: -log_like_iid_gamma_log_params(params, t),
            x0=np.array([3, 3]),
            args=(t,),
            method='Powell'
        )
    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)

def gen_gamma(alpha, b, size):
    """
    Parameters
    __________
    alpha : float
        parameter value for alpha
    
    b : float
        parameter value for b = 1/beta
    
    size : int 
        number of points to sample from the distribution
    
    Returns
    __________
    output : array
        distribution based on input parameters"""
    return rg.gamma(alpha, 1 / b, size=size)

def log_like(t, b_1, delta_b):
    """Log likelihood for exponential distribution

    Parameters
    __________
    t : array
        data array
    b_1 : parameter for first arrival time
        delta_b : parameter for difference between two arrival times

    Returns
    __________
    output : float
        Log-likelihood.
    """
    res = -b_1 * t + np.log(1 - np.exp(-delta_b*t)) + np.log(b_1 + delta_b) + np.log(b_1) - np.log(delta_b)
    return res
vec_log_like = np.vectorize(log_like)

def log_like_iid_exp_log_params(params, t):
    """Log likelihood for i.i.d. exponential measurements with
    input being parameters.

    Parameters
    ----------
    params : array
        Parameters beta and delta_beta
    t : array
        Array of times.

    Returns
    -------
    output : float
        Log-likelihood.
    """
    b_1, delta_b = params
    # For the calculation, we need to take the values of b and delta_b and sum
    # the log PDF for each value in the data set
    result = 0
    if(b_1 <= 0 or delta_b <= 0):
        return -np.inf
    result = vec_log_like(t, b_1, delta_b)
    return np.sum(result)

def gen_exponential(b, delta_b, size):
    '''Generates exponential values given b and delta_b
    Parameters
    __________
    b : float
        parameter for first arrival time
    delta_b : float 
        parameter for difference between two arrival times

    Returns
    __________
    output : float
        Log-likelihood.
    '''
    return (rg.exponential(1/b, size=size) + rg.exponential(1/(b + delta_b), size = size))

def mle_iid_exp(t):
    """Perform maximum likelihood estimates for parameters for i.i.d.
   exponentially distributed measurements, parametrized by beta, delta_beta
   
   Parameters
   ___________
   t : array
       input array of times
   
   Returns
   ___________
   output : tuple
       parameters
   """
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, t: -log_like_iid_exp_log_params(params, t),
            x0=np.array([1, 1]),
            args=(t,),
            method='Powell'
        )
    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)
