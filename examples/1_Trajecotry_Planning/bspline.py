import matplotlib.pyplot as plt
import numpy as np
#from bspline_lab import Bspline
import scipy.interpolate as si

TABLE = {}  # data for memoization


def memoize(f):
    def func(*r_args):
        args = tuple(x for x in r_args if not isinstance(x, list))
        if not args in TABLE:
            TABLE[args] = f(*r_args)
        return TABLE[args]

    return func


@memoize
def __base(x, p, j, u):
    basis = Bspline(u, p)
    return basis(x)[j]


def base(x, p, j, u):
    """
    B-Spline Basis
    :param x: point (0-D)
    :param p: degree of B-Spline
    :param j: knot number
    :param u: knot vector (1-D)
    :return: B
    """

    if p == 0:
        if x == u[j + 1]:  # or < u[j+1]
            return 1
        return 1.0 if u[j] <= x < u[j + 1] else 0.0
    if u[j + p] == u[j]:
        c1 = 0.0
    else:
        c1 = (x - u[j]) / (u[j + p] - u[j]) * base(x, p - 1, j, u)
    if u[j + p + 1] == u[j + 1]:
        c2 = 0.0
    else:
        c2 = (u[j + p + 1] - x) / (u[j + p + 1] - u[j + 1]) * base(x, p - 1, j + 1, u)
    return c1 + c2


def __basis(c, n, degree):
    """ bspline basis function
        c        = number of control points.
        n        = number of points on the curve.
        degree   = curve degree
    """
    # Create knot vector and a range of samples on the curve
    kv = np.array([0] * degree + range(c - degree + 1) + [c - degree] * degree, dtype='int')  # knot vector
    u = np.linspace(0, c - degree, n)  # samples range

    # Cox - DeBoor recursive function to calculate basis
    def coxDeBoor(u, k, d):
        # Test for end conditions
        if (d == 0):
            if (kv[k] <= u and u < kv[k + 1]):
                return 1
            return 0

        Den1 = kv[k + d] - kv[k]
        Den2 = 0
        Eq1 = 0
        Eq2 = 0

        if Den1 > 0:
            Eq1 = ((u - kv[k]) / Den1) * coxDeBoor(u, k, (d - 1))

        try:
            Den2 = kv[k + d + 1] - kv[k + 1]
            if Den2 > 0:
                Eq2 = ((kv[k + d + 1] - u) / Den2) * coxDeBoor(u, (k + 1), (d - 1))
        except:
            pass

        return Eq1 + Eq2

    # Compute basis for each point
    b = np.zeros((n, c))
    for i in xrange(n):
        for k in xrange(c):
            b[i][k % c] += coxDeBoor(u[i], k, degree)

    b[n - 1][-1] = 1

    return b


def base_derivate(x, p, j, u, k):
    """
    k-th derivative of the base
    :param x:
    :param p:
    :param j:
    :param u:
    :param k: k-th derevative
    :return:
    """

    def a(_k, _i):
        """
        coefficent
        :param _k:
        :param i:
        :return:
        """
        den = u[j + p + _i - _k + 1] - u[j + _i]
        if den == 0:
            return 0

        if _i == 0:
            if _k == 0:
                return 1
            else:
                return a(_k - 1, 0) / den
        elif _i == _k:
            return -a(_k - 1, _k - 1) / den
        else:
            return (a(_k - 1, _i) - a(_k - 1, _i - 1)) / den

    from math import factorial
    return factorial(p) / factorial(p - k) * sum(a(k, i) * base(x, p - k, j + i, u) for i in range(k + 1))


"""
def bspline(x, u, c, p, k):
    TABLE = {}
    m = len(c) - 1
    if k == 0:
        return sum(c[j] * base(x, p, j, u) for j in range(m + 1))
    else:
        return sum(c[j] * base_derivate(x, p, j, u, k) for j in range(m + 1))

def vec_bspline(x, u, p, k, m):
    if k == 0:
        return [base(x, p, j, u) for j in range(m + 1)]
    else:
        return [base_derivate(x, p, j, u, k) for j in range(m + 1)]
"""


def bspline(x, u, c, p, k):
    TABLE = {}
    m = len(c) - 1
    if k == 0:
        return sum(c[j] * base(x, p, j, u) for j in range(m + 1))
    else:
        return sum(c[j] * base_derivate(x, p, j, u, k) for j in range(m + 1))


def vec_bspline(x, u, p, k, m):
    if k == 0:
        return [base(x, p, j, u) for j in range(m + 1)]
    else:
        return [base_derivate(x, p, j, u, k) for j in range(m + 1)]


def main():
    k = 4  # degree: jerk continous
    t = [0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 6.0, 7.5, 9.0, 12.5, 16.5, 18.0, 18.0, 18.0, 18.0, 18.0]  # len = 16
    # t = [0, 0, 0, 0, 0, 18, 18, 18, 18, 18]  #len = 16
    c = [3, 4.25, 7.25, 7.39, -13.66, 7.44, 4.98, 12.59, 13.25, 9.12, 8]  # len(c) = 11, m=10

    spl = bspline(0.5, t, c, k, 0)
    print(spl, "== 1.375")

    # from scipy.interpolate import BSpline
    # spl = BSpline(t, c, k)
    # print(spl.basis_element(t))

    derivatives = 0
    VALUE = ["position", "velocity", "acceleration", "jerk"]

    # fig, ax = plt.subplots()
    # xx = np.linspace(0.4, 0.6, 10**4)
    # ax.plot(xx, [bspline(x, t, c, k, 1) for x in xx], 'r-', lw=3, label='naive')
    # ax.grid(True)
    # ax.legend(loc='best')


    fig, ax = plt.subplots()
    ts = 0.01
    T = 18
    xx = np.linspace(0, T - 10 ** -13, T / ts + 1)
    ax.plot(xx, [bspline(x, t, c, k, derivatives) for x in xx], 'r-', lw=3, label=VALUE[derivatives])

    # from scipy.interpolate import BSpline
    # spl = BSpline(t, c, k)
    # ax.plot(xx, spl(xx), 'b-', lw=4, alpha=0.7, label='BSpline_Package')
    if derivatives == 0:
        t = [0, 5, 7, 8, 10, 15, 18 - 10 ** -13]
        q = [3, -2, -5, 0, 6, 12, 8]
        ax.plot(t, q, 'b+', lw=3, label='Pos. Fit')
    elif derivatives == 1:
        t = [0, 18 - 10 ** -13]
        q = [2, -3]
        ax.plot(t, q, 'g+', lw=3, label='Vel. Fit')
    elif derivatives == 2:
        t = [0, 18 - 10 ** -13]
        q = [0, 0]
        ax.plot(t, q, 'g+', lw=3, label='Acc. Fit')

    ax.grid(True)
    ax.legend(loc='best')
    plt.show()

    print([vec_bspline(18 - ts, t, k, derivatives, len(c) - 1)])


def main2():
    points = [[0, 0], [0, 2], [2, 3], [4, 0], [6, 3], [8, 2], [8, 0]]
    points = np.array(points)
    x = points[:, 0]

    t = range(len(points))
    ipl_t = np.linspace(0.0, len(points) - 1, 100)

    k = 4
    x_tup = si.splrep([_t for _t in t], x, k=k)

    x_list = list(x_tup)

    der = 1
    # ==============================================================================
    # Print
    # ==============================================================================
    print("Knot", [_t for _t in t])
    print("Control", x)
    print("Degree", k)
    print("Derivative", der)

    # ==============================================================================
    # Plot
    # ==============================================================================

    fig = plt.figure()

    for i in range(7):
        vec = np.zeros(7)
        vec[i] = 1.0
        x_list = list(x_tup)
        x_list[1] = vec.tolist()

        for j, z in enumerate(x_list):
            print(["Knot:", "Control:", "Degree"][j])
            print(z)
        print()

        x_i = si.splev(ipl_t, x_list, der=der)
        plt.plot(ipl_t, x_i, lw=10, label="scipy_B" + str(i))

        plt.plot(ipl_t, [bspline(y, x_list[0], x_list[1], x_list[2], der) for y in ipl_t], alpha=0.7,
                 label="Shimoda_B" + str(i))

    plt.legend(loc='best')
    plt.xlim([0.0, max(t)])
    plt.title('Basis splines')

    plt.show()


if __name__ == "__main__":
    main2()
