import numpy as np

def projsplx(y):
    """projsplx projects a vector to a simplex
    by the algorithm presented in 
    (Chen an Ye, "Projection Onto A Simplex", 2011)"""
    assert len(y.shape) == 1
    N = y.shape[0]
    y_flipsort = np.flipud(np.sort(y))
    cumsum = np.cumsum(y_flipsort)
    t = (cumsum - 1) / np.arange(1,N+1).astype('float')
    
    t_iter = t[:-1]
    t_last = t[-1]    
    y_iter = y_flipsort[1:]
    
    if np.all((t_iter - y_iter) < 0):
        t_hat = t_last
    else:
        # find i such that t>=y
        eq_idx = np.searchsorted(t_iter - y_iter, 0, side='left')
        t_hat = t_iter[eq_idx]

    x = y - t_hat
    # there may be a numerical error such that the constraints are not exactly met.
    x[x<0.] = 0.
    x[x>1.] = 1.
    assert np.abs(x.sum() - 1.) <= 1e-5
    assert np.all(x >= 0) and np.all(x <= 1.)
    return x