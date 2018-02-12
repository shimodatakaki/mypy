"""
FRF Data Driven H Infinity Feed-Forward Shaping
"""

import myfbcd
import numpy as np


class PrefilterDesign(object):
    def __init__(self, o, T, ts, sinv=None):
        """
        Prefilter CFF makes T T*CFF
         since y = T * (CFF*r) = (T*CFF) * r
        :param o: frequnecy
        :param T: complementary sensitivity
        :param ts: sampling time
        :param sinv: s^(-1)
        """
        self.o = o
        self.T = T
        self.ts = ts
        self.l = [i for i in range(len(o))]
        self.F = len(o)
        if not sinv:
            sinv = myfbcd.calc_sinv(o, ts)
        self.sinv = sinv

    def specification(self, o_dcsc):
        """
        set bandwidth and other constraints
        :param o_dcsc: desired T crossover
        :return:
        """
        self.o_dcsc = o_dcsc

    def first_order(self):
        """
        First order prefilter whose bandwidth = o_dcsc:
        o_dcsc / (s + o_dcsc)
        :return:
        """
        t = self.o_dcsc * self.sinv
        self.cff = t / (1 + t)

    def second_order(self, zeta=1 / np.sqrt(2)):
        """
        Second order prefilter  whose bandwidth = o_dcsc:
        o_dcsc**2 / (s**2 + sqrt(2)*o_dcsc * s + o_dcsc**2)
        :return:
        """
        t1 = 2 * zeta * self.o_dcsc * self.sinv
        t2 = (self.o_dcsc * self.sinv) ** 2
        self.cff = t2 / (1 + t1 + t2)

    def pid_pzc(self, kp, ki, kd, taud):
        """
        CPID = (kd+taud*kp)*s**2 + (kp+taud*ki)*s + ki / DEN
        so this filter cancels num of CPID:
        cff = 1 / ( (kd+taud*kp)*s**2 + (kp+taud*ki)*s + ki )
        :param kp:
        :param ki:
        :param kd:
        :param taud:
        :return:
        """
        if kd > 0:
            t1 = (kp + ki * taud) / (kd + taud * kp) * self.sinv
            t2 = ki / (kd + taud * kp) * (self.sinv ** 2)
            self.cff = t2 / (1 + t1 + t2)
        else:
            t = (ki / kp) * self.sinv
            self.cff = t / (1 + t)

    def filter(self):
        """
        calculate modified T
        :return:
        """
        self.T_filtered = self.T * self.cff

    def check_T_infinity(self, gamma_max=1.1, T=None):
        """
        return True if |T|_infinity <= gamma_max
        :param gamma_max:
        :return:
        """
        if T is None:
            self.filter()
            T = self.T_filtered
        return all(abs(T) <= gamma_max)

    def desired_second_order_T(self, zeta, on):
        """
        Desired T
        :param zeta:
        :param on:
        :return:
        """
        t1 = 2 * zeta * on * self.sinv
        t2 = (on * self.sinv) ** 2
        self.Td = t2 / (1 + t1 + t2)

    def check_lowerer_Td(self, T=None, gamma_max=1):
        """
        check if T <= Td in bandwidth
        :param T:
        :param gamma_max:
        :return:
        """
        if T is None:
            self.filter()
            T = self.T_filtered
        l = [i for i in self.l if self.o[i] < self.o_dcsc]
        for i in l:
            if abs(T[i]) >= abs(self.Td[i]) * gamma_max:
                return False
        return True


def overshoot_to_Tmax(os_ratio):
    """
    calculate Tmax to satsfity given overshoot in a sense of 2nd order system:
    :param os_ratio: max over shoot in percent
    :return:
    """
    if os_ratio < 5 / 100:
        return 1 + 10 ** (-4)
    sigma = np.log(os_ratio)
    zeta = np.sqrt(sigma ** 2 / (np.pi ** 2 + sigma ** 2))
    Tmax = 1 / 2 / zeta / np.sqrt(1 - zeta ** 2)
    return Tmax
