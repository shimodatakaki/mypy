"""

"""

from cvxopt import matrix, solvers
import numpy as np


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None, opt=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
    if A is not None:
        args.extend([matrix(A), matrix(b)])
    sol = solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))


def cvxopt_solve_lp(c, G, h, A=None, b=None, opt=None):
    args = [matrix(c), matrix(G), matrix(h)]
    if A is not None:
        args.extend([matrix(A), matrix(b)])
    sol = solvers.lp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((len(c),))


def cvxopt_solve(solver, args, G=None, h=None, A=None, b=None,
                 opt={'abstol': 10 ** -7, "reltol": 10 ** -6, 'feastol': 10 ** -7}, MAX_ITER_SOL=10, verbose=True):
    if solver == 'qp':
        solver_func = cvxopt_solve_qp
    elif solver == 'lp':
        solver_func = cvxopt_solve_lp
    else:
        return None
    theta = None
    for i in range(MAX_ITER_SOL):
        new_opt = {}
        for k, v in opt.items():
            if not str(v).isalpha():
                new_opt[k] = v * 10 ** i
        for k, v in new_opt.items():
            solvers.options[k] = v
        theta = solver_func(*args, G=G, h=h, A=A, b=b, opt=new_opt)
        if theta is not None:
            break
        if verbose:
            print(new_opt)
            print("This condition is ill. Relaxing the conditon by 1000%...")
    return theta
