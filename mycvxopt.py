"""

"""

from cvxopt import matrix, solvers
import numpy as np


def solve_qp(P, q, G=None, h=None, A=None, b=None):
    """
    Solve quadratic programming
    :param P:
    :param q:
    :param G:
    :param h:
    :param A:
    :param b:
    :return:
    """
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [matrix(P), matrix(q)]
    if G is None:
        G = np.zeros((1, len(P)))
        h = np.zeros((1, 1))
    args.extend([matrix(G), matrix(h)])
    if A is not None:
        args.extend([matrix(A), matrix(b)])
    sol = solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))


def solve_lp(c, G, h, A=None, b=None):
    """
    Sovle linear programming
    :param c:
    :param G:
    :param h:
    :param A:
    :param b:
    :return:
    """
    args = [matrix(c), matrix(G), matrix(h)]
    if A is not None:
        args.extend([matrix(A), matrix(b)])
    sol = solvers.lp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((len(c),))


def solve_socp(c, Gl=None, hl=None, Gql=None, hql=None, A=None, b=None):
    """
    Sovle Second Order Cone Programming
    :param c:
    :param Gl:
    :param hl:
    :param Gql:
    :param hql:
    :param A:
    :param b:
    :return:
    """
    if Gl is None:
        Gl = np.zeros((1, len(c)))
        hl = np.zeros((1, 1))
    if Gql is None:
        Gql = [np.zeros((1, len(c)))]
        hql = [np.zeros((1, 1))]
    args = [matrix(c), matrix(Gl), matrix(hl), [matrix(Gq) for Gq in Gql], [matrix(hq) for hq in hql]]
    if A is not None:
        args.extend([matrix(A), matrix(b)])
    sol = solvers.socp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((len(c),))


def solve(solver, args, G=None, h=None, A=None, b=None, Gql=None, hql=None,
          opt={'abstol': 10 ** -7, "reltol": 10 ** -6, 'feastol': 10 ** -7}, MAX_ITER_SOL=10, verbose=True):
    """
    Solve various optimization proglems
    :param solver:
    :param args:
    :param G:
    :param h:
    :param A:
    :param b:
    :param Gq:
    :param hq:
    :param opt:
    :param MAX_ITER_SOL:
    :param verbose:
    :return:
    """
    solver_func = {'qp': solve_qp, 'lp': solve_lp, "socp": solve_socp}[solver]
    theta = None
    for i in range(MAX_ITER_SOL):
        new_opt = {}
        for k, v in opt.items():
            if not str(v).isalpha():
                new_opt[k] = v * 10 ** i
        for k, v in new_opt.items():
            solvers.options[k] = v
        if solver == "socp":
            theta = solver_func(*args, Gl=G, hl=h, A=A, b=b, Gql=Gql, hql=hql)
        else:
            theta = solver_func(*args, G=G, h=h, A=A, b=b)
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
    """
    Transform lb <= Gx <= ub then return Gx <= h
    :param G:
    :param ub:
    :param lb:
    :return:
    """
    G = np.block([[G], [-G]])
    h = np.block([[ub], [-lb]])
    return G, h


def qc2socp(A, b, c, d):
    """
    Transfrom ||A*x + b*x|| <= c*x + d into
    G*x + [s0; s1] = h, s.t. s0 >= ||s1||2, G = [[-c.T], [-A]], and h = [d; b]
    for second-oder cone programming
    :param A:
    :param b:
    :param c:
    :param d:
    :return:
    """
    Gq = np.block([[-c.T], [-A]])
    hq = np.append(d, b)
    Gq.reshape((Gq.shape[0]*Gq.shape[1]//len(c), len(c)))
    return Gq, hq
