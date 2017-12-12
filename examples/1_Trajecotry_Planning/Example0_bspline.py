"""
TEST
"""
from mytrajectory import Bspline
import numpy as np

def test():
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
        plt.title('PYTHON B-Spline \n'+["Pos.", "Vel.", "Acc.", "Jer."][der])
        plt.xlim([min(t), max(t)])

    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test()