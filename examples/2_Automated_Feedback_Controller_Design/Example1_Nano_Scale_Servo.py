"""
Example1: Nano Scale Servo Control
"""

from myfbcd import *

from scipy import signal
import myplot
import mycvxopt
import mysignal
import mycsv

DATA = "data"
F = 800  # number of FRF lines
PERT = (0, -0.1, 0.1)  # perturbation


def plant(fig):
    """
    return FRF
    :return:
    """
    N = 5

    on = 2 * np.pi * np.array([0, 3950, 5400, 6100, 7100])
    kappa = [1, -1, 0.4, -1.2, 0.9]
    zeta = [0, 0.035, 0.015, 0.015, 0.06]
    Kp = 3.7 * 10 ** 7

    _o = np.array([])
    _h = np.array([])
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
            p = mysignal.symbolic_to_tf(P, s)

            o = 2 * np.pi * np.logspace(1, 4, num=F)
            o, mag, phase = p.bode(w=o)
            theta = phase / 180 * np.pi
            a = 10 ** (mag / 20)
            h = a * np.exp(1.j * theta)
            myplot.bodeplot(fig, o / 2 / np.pi, mag, phase, line_style="-", xl=[10 ** 1, 10 ** 4])
            _o = np.append(_o, o)
            _h = np.append(_h, h)

    myplot.save(fig, save_name=DATA + "/" + str(fig) + "_plant", title="plant",
                leg=["plant" + str(i) for i in range(N)])

    assert len(_o) == len(_h)
    return fig, np.array(_o), np.array(_h)


def optimize(fig, o, g, nofir=5, f_desired_list=[50 + 150 * i for i in range(10)]):
    """
    calc pids and firs
    :param o:
    :param p:
    :return:
    """
    THETA_DPM = 30 / 180 * np.pi  # Phase Margin
    THETA_DPM2 = 30 / 180 * np.pi  # Second Phase Margin
    GDB_DGM = 5  # Gain Margin in (dB)
    # SIGMA = -1
    # RM = np.sqrt((1 + SIGMA) ** 2 - 2 * SIGMA * (1 - np.cos(THETA_DPM2)))
    gm = 10 ** (GDB_DGM / 20)
    tm = max(THETA_DPM, THETA_DPM2)
    RM = max(1 - 1 / gm, np.sqrt(2 - 2 * np.cos(tm)))
    SIGMA = -1
    NSTBITER = 5
    is_robust = False
    GAMMA = 0.5

    TAUD = 500 ** (-6)  # Pseudo Differential Cut-off for D Control
    NOFIR = nofir  # Number of FIRs
    NOPID = "pid"
    TS = 50 * 10 ** (-6)  # sampring of FIRs

    _f = 0
    _c = None
    tol = 1
    rho = None
    for f in f_desired_list:
        F_DGC = 2 * np.pi * f  # Desired Cross-over Frequency
        print("Try: ", f, " Hz")
        fbc = ControllerDesign(o, g, nopid=NOPID, taud=TAUD, nofir=NOFIR, ts=TS, rho0=rho)
        fbc.specification(F_DGC, THETA_DPM, GDB_DGM, theta_dpm2=THETA_DPM2)  # set constraints
        try:
            for i in range(NSTBITER):
                fbc.gaincond()  # append gain constraints
                fbc.stabilitycond(rm=RM, sigma=-1)  # append stability condition
                fbc.nominalcond()  # append nominal performance condition
                if is_robust:
                    fbc.robustcond(GAMMA)  # may need much time
                rho = fbc.optimize()
                _f = f
                _c = fbc
                fbc.lreset()
        except:
            tol -= 1
            if tol < 0:
                break

    for e in range(11, 0, -1):
        print((11 - e) * ' ' + e * '*')
    print('')
    for g in range(11, 0, -1):
        print(g * ' ' + (11 - g) * '*')

    assert _f > 0
    print("Best cross-over frequency:", _f, " Hz")
    print("PIDs:", rho[:3])
    print("FIRs:", rho[3:])
    fbc = _c
    assert fbc.rho[-1] == rho[-1] and fbc.rho[0] == rho[0]
    mycsv.save(rho, save_name=DATA + "/rho" + str(fig) + ".csv",
               header=("P,I,D, FIR(1+n) for n in range(10)", "taud (s):" + str(TAUD), "FIR sampling (s):" + str(TS)))
    N = 5  # number of data

    ##########Plot1##########
    fig += 1
    for j in range(N):
        L = np.dot(fbc.X[j * F:(j + 1) * F], fbc.rho)

        # Nyquist
        myplot.plot(fig, np.real(L), np.imag(L), lw=6, line_style="-")

    _theta = np.linspace(0, 2 * np.pi, 100)
    myplot.plot(fig, (np.cos(_theta)), (np.sin(_theta)), line_style="r--", lw=1)  # r=1
    myplot.plot(fig, (0,), (0,), line_style="r+")  # origin
    myplot.plot(fig, (-1,), (0,), line_style="r+")  # critical
    # Stanility Constraint
    myplot.plot(fig, (RM * np.cos(_theta)) + SIGMA, RM * np.sin(_theta), line_style="y:")  # r=1
    # Save
    myplot.save(fig, xl=[-1, 1], yl=[-1, 1],
                leg=(*["Nyquist" + str(k) for k in range(N)], "r=1", "Origin", "(-1,j0)", "Stb. Cond."),
                label=("Re", "Im"), save_name=DATA + "/" + str(fig) + "_nyquist_enlarged",
                title="Optimized Gain-Crossover Frequency (Hz): " + str(_f))

    myplot.save(fig, xl=[-3, 3], yl=[-3, 3],
                leg=(*["Nyquist" + str(k) for k in range(N)], "r=1", "Origin", "(-1,j0)", "Stb. Cond."),
                label=("Re", "Im"), save_name=DATA + "/" + str(fig) + "_nyquist",
                title="Optimized Gain-Crossover Frequency (Hz): " + str(_f))

    ##########Plot2##########
    fig += 1
    # L(s)
    for j in range(N):
        L = np.dot(fbc.X[j * F:(j + 1) * F], fbc.rho)
        myplot.bodeplot(fig, o[:F] / 2 / np.pi, 20 * np.log10(abs(L)), np.angle(L, deg=True), line_style='-')
    myplot.save(fig, save_name=DATA + "/" + str(fig) + "_bode", title="Openloop L(s)",
                leg=["L" + str(k) for k in range(N)])

    ##########Plot3##########
    fig += 1
    c = fbc.freqresp()[:F]
    myplot.bodeplot(fig, o[:F] / 2 / np.pi, 20 * np.log10(abs(c)), np.angle(c, deg=True), line_style='-')
    myplot.save(fig, save_name=DATA + "/" + str(fig) + "_controller", title="Controller C(s)")

    ##########Plot4##########
    fig += 1
    for j in range(N):
        L = np.dot(fbc.X[j * F:(j + 1) * F], fbc.rho)
        T = 1 / (1 + L)
        S = 1 - T
        myplot.plot(fig, o[:F] / 2 / np.pi, 20 * np.log10(abs(T)), line_style='b-', plotfunc=plt.semilogx)
        myplot.plot(fig, o[:F] / 2 / np.pi, 20 * np.log10(abs(S)), line_style='r-', plotfunc=plt.semilogx)
    myplot.save(fig, save_name=DATA + "/" + str(fig) + "_ST", title="S(s) (Blue) and T(s) (Red).")

    return fig


if __name__ == "__main__":
    from sympy import *
    import os
    import matplotlib.pyplot as plt

    try:
        os.mkdir(DATA)
    except:
        pass

    s = symbols('s')

    fig = -1

    fig, o, h = plant(fig)
    fig = optimize(fig, o, h, nofir=10)  # PID + 10 FIRs

    myplot.show()
