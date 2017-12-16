"""
Example0: B-spline test
"""
from mytrajectory import Bspline
import numpy as np
import scipy.interpolate as si


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


def test2():
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


def test3():
    points = [[0, 0], [0, 2], [2, 3], [4, 0], [8, 3], [8, 2], [8, 0]]
    points = np.array(points)
    x = points[:, 0]

    N = 10

    nc = 7
    t = [i * N / (nc - 1) for i in range(nc)]
    k = 4

    der = 0

    ipl_t = np.linspace(0.0, max(t), 100)

    bspl = Bspline(t, nc, k, verbose=True)

    # ==============================================================================
    # Plot
    # ==============================================================================
    fig = plt.figure()
    x_i = bspl.basis(ipl_t, der=der)

    # print(len(x_i))

    # for _x in x_i:
    #     print(_x)
    # print()

    for i, z in enumerate(zip(*x_i)):
        print(z)
        plt.plot(ipl_t, z, lw=2.2, label="scipy_B" + str(i))
    plt.legend(loc='best')
    plt.title('Basis splines')
    plt.xlim([min(t), max(t)])

    from bspline import bspline
    for der in range(4):
        x_i = bspl.basis(ipl_t, der=der)

        fig = plt.figure()
        plt.plot(ipl_t, bspl.bspline(x, ipl_t, der=der), lw=10)

        plt.plot(ipl_t, [bspline(_x, bspl.tck[0], x, k, der) for _x in ipl_t], 'r-', lw=3)
        plt.title('PYTHON B-Spline \n' + ["Pos.", "Vel.", "Acc.", "Jer."][der])
        plt.xlim([min(t), max(t)])

    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test2()
    test3()
