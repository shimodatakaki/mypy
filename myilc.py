import numpy as np
from scipy.special import comb
import myfbcd
import mycvxopt


class SuperILCDesign(object):
    def __init__(self, o, g, ts, cfb, z=None):
        """
        Super ILC design
        :param o:
        :param g:
        :param ts:
        :param cfb:
        :param z:
        """
        self.o = o
        self.g = g
        self.ts = ts
        self.F = len(o)

        self.cfb = cfb
        self.L = g * cfb
        self.S = 1 / (1 + self.L)

        if z is None:
            z = myfbcd.calc_z(self.o, self.ts)
        self.z = z

    def nominal_inverse(self, kc):
        """
        ZPETC inverse:
        (z-1)**2 * (z+1) / 4 / k / z, where k = kc * ts**2 / 2
        :param kc: Pn = kc/s**2
        :return:
        """
        jn = 1 / kc  # nominal inertia
        self.ginvn = jn * 2 / self.ts ** 2 * (self.z - 1) ** 2 * (self.z + 1) / 4 / self.z

    def ZEPEQ(self, n):
        """
        Zero Phase Error Q filter:
        Q = (z+1)**2n / 4**n / z**n = (cos(theta/2))**2n
        :param n:
        :return:
        """
        theta = self.o * self.ts
        self.Q = np.cos(theta / 2) ** (2 * n)

    def calc_infinity_norm(self):
        """
        return | V |_infinity, where
        V = Q * S * (1 - P * Pninv)
        :return:
        """
        self.V = self.Q * self.S * (1 - self.g * self.ginvn)
        return max(abs(self.V))

    def ZEPETC_fir(self, kc):
        """
        calculate fir coefficients of ZPETC
        :param kc:
        :return:
        """
        k = kc * self.ts ** 2 / 2
        fir = np.array([1, -1, -1, 1]) / 4 / k
        return fir

    def ZEPEQ_fir(self, n):
        """
        calculate fir coefficients of ZEPE Q filter
        :param n:
        :return:
        """
        fir = np.array([comb(2 * n, 2 * n - i) for i in range(2 * n + 1)]) / 4 ** n
        return fir

    def save_fir(self, kc, n, ndata=0, max_line_number=5):
        """
        save to .h file
        :param kc:
        :param n:
        :param ndata:
        :param max_line_number:
        :return:
        """
        MZEPETC = -1
        NPARZEPETC = 4
        zepetc_fir = self.ZEPETC_fir(kc)

        MILC = -n
        NPARILC = n - MILC + 1
        zepeq_fir = self.ZEPEQ_fir(n)

        ctrl_ilc_par = "ctrl_ilc_par.h"

        def write_define(f, s, v):
            f.write("#define " + s + " (" + str(v) + ")\n")

        def write_vector(f, vector):
            for i, x in enumerate(vector):
                f.write(str(x))
                if i == len(vector) - 1:
                    break
                f.write(", ")
                if i and not i % max_line_number:
                    f.write("\n")
            f.write("};\n")

        with open(ctrl_ilc_par, "w") as f:
            f.write("//   " + ctrl_ilc_par + "\n")
            f.write("//   Copyright: Shimoda Takaki, 2018, The University of Tokyo.")
            f.write("\n")
            f.write("//   https://github.com/shimodatakaki/mypy")
            f.write("\n\n")
            f.write("#pragma once")
            f.write("\n\n")
            # write_define(f, "NDATAILC", ndata)
            # f.write("\n")
            write_define(f, "MZEPETC", MZEPETC)
            write_define(f, "NPARZEPETC", NPARZEPETC)
            f.write("\n")
            write_define(f, "MILC", MILC)
            write_define(f, "NPARILC", NPARILC)
            f.write("\n")
            f.write("float cZEPETC[NPARZEPETC]={")
            write_vector(f, zepetc_fir)
            f.write("float cILC[NPARILC]={")
            write_vector(f, zepeq_fir)


class ILCOptimize(SuperILCDesign):

    def __init__(self, o, g, ts, cfb, z=None):
        """
        optimize evaluation function ||V||_p, where p=1, 2, or infinity
        :param o:
        :param g:
        :param ts:
        :param cfb:
        :param z:
        """
        super().__init__(o, g, ts, cfb, z=z)

        self.nominal_inverse(1)  # 1 since it is coeffcient
        self.phi = self.ginvn

    def optimize(self):
        """
        min |V|_infinity, where
        V = Q * S * (1 - P * Pninv)
        with parameter rho = [jn]
        :return:
        """
        P = np.ones((self.F, 1), dtype=complex)
        P[:, 0] = - self.Q * self.S * self.g * self.phi
        P.reshape((self.F, 1))
        q = np.ones((self.F, 1), dtype=complex)
        q[:, 0] = - self.Q * self.S
        P = np.block([[np.real(P)], [np.imag(P)]])
        q = np.block([[np.real(q)], [np.imag(q)]])
        _gamma, self.rho = mycvxopt.solve_min_infinity_norm(P, q)
        return self.rho
