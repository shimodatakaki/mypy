"""

"""
from scipy import optimize
import numpy as np

def nelder_mead(f, x0, verbose=False, maxiter=200, callback=None):
    """
    nelder mead optimization
    :param f: f(x, *args) whose x in set R^n
    :param x0: (n+1)*n np.array
    :param verbose:
    :param maxiter:
    :param callback:
    :return:
    """
    n = x0.shape[1] #number of parameters
    assert x0.shape[0] == n + 1
    new_opt = {"disp": verbose, "maxiter": maxiter, "initial_simplex": x0}

    res = optimize.minimize(f, x0=np.zeros(n), method='Nelder-Mead', tol=1e-6, options=new_opt, callback=callback)
    return res