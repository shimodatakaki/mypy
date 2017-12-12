import numpy as np
from scipy import linalg

BAD_RESULT = 1000

def nonlinear_least_square(beta_init, update,
                           MAX_ITER=100, _LAMBDA=10 ** -9, REL_ERROR=10 ** -9, RELAX=0.4, PENALTY=10, verbose=True):
    """
    # Levenberg–Marquardt algorithm, to minimize non-linear least squares
    :param beta: <numpy 1D-array> parameter
    :param update: <function> to update r, jacobian, cost (basically sum(r**2)[0])
    :param MAX_ITER: max iteration
    :param _LAMBDA: initial lambda
    :param REL_ERROR: admissible relative error
    :param RELAX: relax ratio to boost optimization
    :param PENALTY: penalty ratio to improve calculation
    :return: <numpy 1D-array> beta
    """
    _lambda = _LAMBDA
    old_cost, new_cost = np.array(np.inf), np.array(0)
    beta = np.array(beta_init)
    beta_old = np.array(beta)
    for _ in range(MAX_ITER):
        r, jacob, new_cost = update(beta)
        if 0 < (1 - new_cost / old_cost) < REL_ERROR:  # ok
            break
        elif old_cost > new_cost:  # boost
            old_cost = np.array(new_cost)
            _lambda *= RELAX
            beta_old = np.array(beta)
        elif old_cost < new_cost:  # too great delta, go back to previous beta with PENALTY*_lambda
            _lambda *= PENALTY
            beta = np.array(beta_old)
            continue
        delta = linalg.inv(
            jacob.T * jacob + _lambda * np.diag(np.diag(jacob.T * jacob))) * jacob.T * np.matrix(r)
        beta -= np.array(delta).T[0]

    if verbose:
        print("Cost Function: ", new_cost)
        if new_cost > BAD_RESULT:
            print("Too bad result!!!")
    return beta
