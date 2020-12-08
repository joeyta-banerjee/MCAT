import numpy as np

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
