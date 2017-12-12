"""

"""

import scipy.interpolate as si
import numpy as np


class Bspline():
    """
    Construct bspline
    """
    KNOT = 0
    CONTROL = 1
    DEGREE = 2

    def __init__(self, u, nc, p, verbose=False):
        """

        :param u: knot vector
        :param nc: number of control points
        :param p: degree
        """
        assert len(u) == nc
        assert nc > p
        self.nc0 = nc
        self.tck = si.splrep(u, np.zeros(nc), k=p)
        if verbose:
            print("Knot:")
            print(self.tck[0])
            print("Number of control points:")
            print(self.nc0)
            print("Degree:")
            print(self.tck[2])

    def basis(self, x_list, der=0, u=[], nc=0):
        """

        :param x_list: x for B_i^p(x) in x_list
        :param der: der-th derivative
        :return: [ [B_i^p(_x) for i in range(nc)] for _x in x_list ]
        """
        if nc == 0:
            nc = self.nc0
        if not u:
            u = self.tck[self.KNOT]

        ret = []
        for i in range(nc):
            vec = np.zeros(nc)
            vec[i] = 1.0
            tck = list(self.tck)
            tck[0] = u
            tck[1] = vec.tolist()
            ret.append(si.splev(x_list, tck, der=der))
        return [x for x in zip(*ret)]

    def bspline(self, c, x_list, der=0):
        ret = []
        for a in self.basis(x_list, der):
            ret.append(sum(b * cp for b, cp in zip(a, c)))
        return ret



