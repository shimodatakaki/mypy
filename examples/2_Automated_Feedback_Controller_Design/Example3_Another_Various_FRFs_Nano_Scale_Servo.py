"""
Example1: Single_FRF_Nano_Scale_Servo
Reference: page 105, T. Yamaguchi, M. Hirata and H. Fujimoto, Nanoscale servo control, Tokyo Denki University Press, 2002.
"""

from myfbcd import *
import myplot
import mysignal
import mycsv
from scipy import signal

from Example1_Single_FRF_Nano_Scale_Servo import *

DATA = "data"
F = 1000  # number of FRF lines
TS = 50 * 10 ** (-6)  # sampring of FIRs
TD = 3 / 10 * TS  # Delay
PERT = (0, -0.1, 0.1, 0.2, -0.2)  # perturbation
NDATA = 2 * len(PERT) - 1


def plant_data(fig):
    """
    return FRF
    :return:
    """
    on = 2 * np.pi * np.array([0, 3950, 5400, 6100, 7100])
    kappa = [1, -1, 0.4, -1.2, 0.9]
    zeta = [0, 0.035, 0.015, 0.015, 0.06]
    Kp = 3.7 * 10 ** 7

    ol = np.array([])
    hl = np.array([])
    fig += 1
    for i in range(2):
        for l in PERT:
            if l == 0 and i > 0:
                continue
            P = 0
            for o, k, z in zip(on, kappa, zeta):
                o *= (1 + l * (i == 0))
                k *= (1 + l * (i == 1))
                z *= (1 + l * (i == 2))
                P += k / (s ** 2 + 2 * z * o * s + o ** 2)
            P *= Kp
            # calc continous
            p = mysignal.symbolic_to_tf(P, s, ts=0)
            o = 2 * np.pi * np.linspace(10 ** 0, 10 ** 4, num=F)
            o, mag, phase = signal.bode(p, w=o)
            theta = phase / 180 * np.pi
            a = 10 ** (mag / 20)
            h = a * np.exp(1.j * theta)
            # calc discrete with 3/10 delay
            h = h * mysignal.zoh_w_delay(o, TS, TD)
            myplot.bodeplot(fig, o / 2 / np.pi, 20 * np.log10(abs(h)), np.angle(h, deg=True), line_style="-",
                            xl=[10 ** 1, 10 ** 4])

            ol = np.append(ol, o)
            hl = np.append(hl, h)

    myplot.save(fig, save_name=DATA + "/" + str(fig) + "_plant", title="plant",
                leg=["plant" + str(i) for i in range(NDATA)])
    mycsv.save(ol, np.real(hl), np.imag(hl), save_name=DATA + "/example1_plant_frf.csv",
               header=("o (rad/s)", "real(FRF)", "imag(FRF)"))

    assert len(ol) == len(hl)
    return fig, np.array(ol), np.array(hl)


if __name__ == "__main__":
    from sympy import *
    import os

    try:
        os.mkdir(DATA)
    except:
        pass

    s = symbols('s')

    fig = -1
    fig, o, g = plant_data(fig)
    fig, fbc = optimize(fig, o, g, nofir=30)
    fig = plotall(fig, fbc, ndata=NDATA)

    myplot.show()
