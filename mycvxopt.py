"""

"""

from cvxopt import matrix, solvers
import numpy as np


def solve_qp(P, q, G=None, h=None, A=None, b=None, opt=None):
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


def solve_lp(c, G, h, A=None, b=None, opt=None):
    args = [matrix(c), matrix(G), matrix(h)]
    if A is not None:
        args.extend([matrix(A), matrix(b)])
    sol = solvers.lp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((len(c),))


def solve(solver, args, G=None, h=None, A=None, b=None,
          opt={'abstol': 10 ** -7, "reltol": 10 ** -6, 'feastol': 10 ** -7}, MAX_ITER_SOL=10, verbose=True):
    if solver == 'qp':
        solver_func = solve_qp
    elif solver == 'lp':
        solver_func = solve_lp
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


def solve_min_infinity_norm(F, f, G=None, h=None, A=None, b=None, verbose=True):
    """
    Transform infinity norm minimization problem:
    min ||tau||_infty = ||Fx - f||_infty s.t. Gx<=h, Ax=b into
    min c^T[tau x] s.t. tau e >= Fx - f and tau e >= -(Fx - f).
    :param F:
    :param f:
    :param G:
    :param h:
    :param A:
    :param b:
    :param verbose:
    :return:
    """
    _, n = F.shape
    c = np.block([1, np.zeros(n)])
    e = np.ones((len(F), 1))
    G_inf = np.block([[-e, F], [-e, -F]])
    h_inf = np.block([[f], [-f]])
    if G is not None:
        G = np.block([[G_inf], [np.zeros((len(G), 1)), G]])
        h = np.block([[h_inf], [h]])
    else:
        G = G_inf
        h = h_inf
    if A is not None:
        A = np.block([np.zeros((len(A), 1)), A])
    x = solve("lp", [c], G=G, h=h, A=A, b=b, verbose=verbose)
    if verbose:
        print("Minimized Infinity Norm:", x[0])
    return x[0], x[1:]


def constraints(G, ub, lb):
    G = np.block([[G], [-G]])
    h = np.block([[ub], [-lb]])
    return G, h