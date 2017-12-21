from scipy import signal
from sympy import *
import numpy as np


class MyTransferFunction():
    def __init__(self, f, s=symbols("s")):
        self.f = f
        self.s = s

    def freqresp(self, w):
        return [complex(self.f.subs(self.s, 1j * o)) for o in w]


def zoh_compensation(tau, w):
    """
    ZOH compenstation: *= tau*s / (1 - exp(-ts*s)
    :return:
    """
    s = symbols('s')
    zoh = (1 - exp(-s * tau)) / (s * tau)
    p = MyTransferFunction(1 / zoh, s=s)
    return p.freqresp(w)


def zoh_w_delay(o, ts, td=0):
    return np.array([
                        (1 - np.exp(-1.j * _o * ts)) / (1.j * _o * ts) * np.exp(-1.j * _o * td)
         for _o in o])


def symbolic_to_tf(expr, symbol, ts=0.0):
    """

    :param expr:P('s')
    :param symbol: s = symbol('s')
    :return: P(s)
    """
    nd = fraction(cancel(expr))  # [numerator, denominator]
    degrees = []
    for x in nd:
        try:
            degrees.append(degree(x))
        except ComputationFailed:
            degrees.append(0)  # only coefficients
    a_nd = [[float(x.coeff(symbol, i)) for i in range(degrees[i] + 1)][::-1] for i, x in enumerate(nd)]
    if ts == 0:
        return signal.TransferFunction(*a_nd)
    else:
        n, d, ts = signal.cont2discrete(a_nd, dt=ts)
        p = signal.TransferFunction([x for x in n[0] if not x == 0], [x for x in d if not x == 0], dt=ts)
        return p


def array_to_eq(a, x):
    """
    calculate equation from array
    :param a: a = [an, ..., a1, a0] or [bm, ..., b1, b0]
    :param x: variable
    :return: an*x^n + ... + a1*x + a0 or ...
    """
    return sum(c * x ** i for i, c in enumerate(a[::-1]))


def concatenate(*systems, o="*"):
    """
    :param systems:
    :param o:
    :return: prod(systems) if o=="*", sum(systems)  elif o=="+"
    """

    def parallel(x, y):
        return x + y

    def series(x, y):
        return x * y

    s = symbols('s')
    operator = {"*": series, "+": parallel}
    system = 1
    for p in systems:
        system = operator[o](system, array_to_eq(p.num, s) / array_to_eq(p.den, s))
    return symbolic_to_tf(system, s)


def inv_tf(system):
    """
    :param system:  P(s)
    :return:  1/P(s)
    """
    return signal.TransferFunction(system.den, system.num)


def minus_tf(system):
    """
    :param system:  P(s)
    :return:  -P(s)
    """
    return signal.TransferFunction([-x for x in system.num], system.den)


def lpf():
    pass


def hpf():
    pass


def second_order():
    pass


"""
test
"""


def test():
    num = [1, 1]
    den = [1, 2, 1]
    sysA = signal.TransferFunction(num, den)
    num = [1, 1]
    den = [1, 4, 4]
    sysB = signal.TransferFunction(num, den)
    sysD = concatenate(sysA, sysB, o="*")
    print(sysD)
    print(minus_tf(sysA))
    print(inv_tf(sysA))

    s = symbols('s')
    P = (0.1 *
         (100 * 2 * np.pi) ** 2 / (s ** 2 + 2 * 0.001 * 100 * 2 * np.pi * s + (100 * 2 * np.pi) ** 2) *
         (s ** 2 + 2 * 0.1 * 50 * 2 * np.pi * s + (50 * 2 * np.pi) ** 2) / (50 * 2 * np.pi) ** 2 *
         1 / (0.1 * s + 1) *
         (400 * 2 * np.pi) ** 2 / (s ** 2 + 2 * 0.001 * 400 * 2 * np.pi * s + (400 * 2 * np.pi) ** 2) *
         (300 * 2 * np.pi) ** 2 / (s ** 2 + 2 * 0.01 * 300 * 2 * np.pi * s + (300 * 2 * np.pi) ** 2) *
         (200 * 2 * np.pi) ** 2 / (s ** 2 + 2 * 0.01 * 200 * 2 * np.pi * s + (200 * 2 * np.pi) ** 2))
    p = symbolic_to_tf(P, s)
    print(p)


def test2():
    s = symbols('s')
    P = s / (1 - exp(-s))
    P = MyTransferFunction(P)
    w = [0.1, 1, 10]
    print(P.freqresp(w))


if __name__ == "__main__":
    import numpy as np

    test2()
