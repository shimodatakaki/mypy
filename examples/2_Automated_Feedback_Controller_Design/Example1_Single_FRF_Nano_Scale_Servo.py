"""
Example1: Single_FRF_Nano_Scale_Servo
Reference: page 105, T. Yamaguchi, M. Hirata and H. Fujimoto, Nanoscale servo control, Tokyo Denki University Press, 2002.
"""

from myfbcd import *
import myplot
import mysignal
import mycsv
from scipy import signal
import matplotlib.pyplot as plt
from sympy import *

DATA = "data/example1_result"
F = 1000  # number of FRF lines
TS = 50 * 10 ** (-6)  # sampring of FIRs
TD = 3 / 10 * TS  # Delay
NDATA = 1  # number of data


def plant(fig, datapath=DATA):
    """
    return FRF
    :return:
    """
    s = symbols('s')

    on = 2 * np.pi * np.array([0, 3950, 5400, 6100, 7100])
    kappa = [1, -1, 0.4, -1.2, 0.9]
    zeta = [0, 0.035, 0.015, 0.015, 0.06]
    Kp = 3.7 * 10 ** 7

    ol = np.array([])
    hl = np.array([])
    fig += 1
    P = 0
    for o, k, z in zip(on, kappa, zeta):
        P += k / (s ** 2 + 2 * z * o * s + o ** 2)
    P *= Kp
    # calc continous
    p = mysignal.symbolic_to_tf(P, s, ts=0)
    o = 2 * np.pi * np.linspace(10 ** 0, 10 ** 4, num=F)
    o, mag, phase = signal.bode(p, w=o)
    h = mysignal.magphase2resp(mag, phase)
    # calc discrete with 3/10 delay
    h = h * mysignal.zoh_w_delay(o, TS, TD)
    mag, angle = mysignal.resp2magphase(h, deg=True)
    myplot.bodeplot(fig, o / 2 / np.pi, mag, angle, line_style="-", xl=[10 ** 1, 10 ** 4])

    ol = np.append(ol, o)
    hl = np.append(hl, h)

    myplot.save(fig, save_name=datapath + "/" + str(fig) + "_plant", title="plant",
                leg=["plant" + str(i) for i in range(NDATA)])
    mycsv.save(ol, np.real(hl), np.imag(hl), save_name=datapath + "/example1_plant_frf.csv",
               header=("o (rad/s)", "real(FRF)", "imag(FRF)"))

    assert len(ol) == len(hl)
    return fig, np.array(ol), np.array(hl)


def optimize(fig, o, g, nofir=50, datapath=DATA):
    """
    calc pids and firs
    :param o:
    :param g:
    :return:
    """
    THETA_DPM = 30 / 180 * np.pi  # Phase Margin
    THETA_DPM2 = 30 / 180 * np.pi  # Second Phase Margin
    GDB_DGM = 5  # Gain Margin in (dB)
    gm = 10 ** (GDB_DGM / 20)
    tm = max(THETA_DPM, THETA_DPM2)

    NSTBITER = 3

    TAUD = 2 * TS  # Pseudo Differential Cut-off for D Control
    NOFIR = nofir  # Number of FIRs
    NOPID = "pid"

    f = 10
    _f = [0]
    _c = [None]
    rho = None
    R = 2
    LAMBDA = (1 + R) / R / 2
    tol = 10
    rho_best=rho
    while tol > 0:
        F_DGC = 2 * np.pi * f  # Desired Cross-over Frequency (rad/s)
        print("Try: ", f, " Hz")
        for i in range(NSTBITER):
            fbc = ControllerDesign(o, g, nopid=NOPID, taud=TAUD, nofir=NOFIR, ts=TS, tsfir=TS, rho0=rho)
            fbc.specification(F_DGC, THETA_DPM, GDB_DGM, theta_dpm2=THETA_DPM2)  # set constraints
            fbc.nominalcond(db=-40)  # append nominal performance condition
            fbc.stabilitycond()  # append stability condition
            fbc.gainpositivecond()  # append gain constraints
            fbc.picond(ti=50 / 1000)
            if i > 0 and nofir > 0:
                fbc.fircond()  # append gain constraints
            try:
                rho = fbc.optimize()
            except:
                tol -= 1
                f = f * LAMBDA
                rho = rho_best
                break
            if i >= NSTBITER // 2 and check_disk(np.dot(fbc.X, fbc.rho), fbc.rm, fbc.sigma):
                print("Solver found a local minima @ iteration", i)
                if f > max(_f):
                    print("Best @", f)
                    rho_best = rho
                print()
                _f.append(f)
                _c.append(fbc)
                f *= R
                break

    for e in range(11, 0, -1):
        print((11 - e) * ' ' + e * '*')
    print('')
    for g in range(11, 0, -1):
        print(g * ' ' + (11 - g) * '*')

    assert _f[-1] > 0
    i_max = [i for i, f in enumerate(_f) if f == max(_f)][0]
    print("Best nominal frequency:", _f[i_max], " Hz")
    fbc = _c[i_max]
    print("PIDs:", fbc.rho[:3])
    print("FIRs:", fbc.rho[3:])
    mycsv.save(fbc.rho, save_name=datapath + "/rho" + str(fig) + ".csv",
               header=("P,I,D, FIR(1+n) for n in range(" + str(nofir) + ")", "taud (s):" + str(TAUD),
                       "FIR sampling (s):" + str(TS)))

    return fig, fbc


def plotall(fig, fbc, ndata=NDATA, datapath=DATA):
    """

    :param fig:
    :return:
    """
    fbc.split(ndata)
    olist = fbc.olist
    o = olist[0]
    F = len(o)
    Llist = fbc.Llist
    gcf = fbc.calc_gcf()
    print("Gain Crossover Frequencies (Hz):", gcf)
    print("\tMin. (Hz):", min(gcf))
    print("\tAve. (Hz):", np.mean(gcf))
    print("\tMax. (Hz):", max(gcf))
    f_gc = min(gcf)

    ##########Plot1##########
    fig += 1
    for L in Llist:
        # Nyquist
        myplot.plot(fig, np.real(L), np.imag(L), lw=6, line_style="-")

    _theta = np.linspace(0, 2 * np.pi, 100)
    myplot.plot(fig, (np.cos(_theta)), (np.sin(_theta)), line_style="r--", lw=1)  # r=1
    myplot.plot(fig, (0,), (0,), line_style="r+")  # origin
    myplot.plot(fig, (-1,), (0,), line_style="r+")  # critical
    # Stanility Constraint
    myplot.plot(fig, (fbc.rm * np.cos(_theta)) + fbc.sigma, fbc.rm * np.sin(_theta), line_style="y:")  # r=1
    myplot.plot(fig, (-1 / fbc.g_dgm, - np.cos(fbc.theta_dpm), -np.cos(fbc.theta_dpm2)),
                (0, - np.sin(fbc.theta_dpm), np.sin(fbc.theta_dpm2)),
                line_style="yo")
    myplot.save(fig, xl=[-1, 1], yl=[-1, 1],
                leg=(*["Nyquist" + str(k) for k in range(ndata)], "r=1", "Origin", "(-1,j0)", "Stb. Cond.", "Margins"),
                label=("Re", "Im"), save_name=datapath + "/" + str(fig) + "_nyquist_enlarged",
                title="Optimized Gain-Crossover Frequency (Hz): " + str(f_gc))

    myplot.save(fig, xl=[-3, 3], yl=[-3, 3],
                leg=(*["Nyquist" + str(k) for k in range(ndata)], "r=1", "Origin", "(-1,j0)", "Stb. Cond.", "Margins"),
                label=("Re", "Im"), save_name=datapath + "/" + str(fig) + "_nyquist",
                title="Optimized Gain-Crossover Frequency (Hz): " + str(f_gc))

    ##########Plot2##########
    fig += 1
    # L(s)
    for L in Llist:
        myplot.bodeplot(fig, o[:F] / 2 / np.pi, 20 * np.log10(abs(L)), np.angle(L, deg=True),
                        line_style='-')
    myplot.save(fig, save_name=datapath + "/" + str(fig) + "_bode", title="Openloop L(s)",
                leg=["L" + str(k) for k in range(ndata)])

    ##########Plot3##########
    fig += 1
    try:
        c = fbc.freqresp()[:F]
    except:
        c = fbc.C[:F]
    myplot.bodeplot(fig, o[:F] / 2 / np.pi, 20 * np.log10(abs(c)), np.angle(c, deg=True), line_style='-')
    g = fbc.g[:F]
    myplot.bodeplot(fig, o[:F] / 2 / np.pi, 20 * np.log10(abs(g)), np.angle(g, deg=True), line_style='-')
    myplot.save(fig, title='C(s)', leg=("Controller", "Plant"), save_name=datapath + "/Controller")

    ##########Plot4##########
    fig += 1
    m = fbc.nominal_sensitivity / (-20)
    x = (fbc.o_dgc / fbc.o[:F]) ** m
    myplot.plot(fig, o[:F] / 2 / np.pi, - 20 * np.log10(abs(x)), line_style='k--', plotfunc=plt.semilogx)
    for L in Llist:
        T = 1 / (1 + L)
        S = 1 - T
        myplot.plot(fig, o[:F] / 2 / np.pi, 20 * np.log10(abs(T)), line_style='b-', plotfunc=plt.semilogx)
        myplot.plot(fig, o[:F] / 2 / np.pi, 20 * np.log10(abs(S)), line_style='r-', plotfunc=plt.semilogx)
    myplot.save(fig, save_name=datapath + "/" + str(fig) + "_ST", title="S(s) (Blue) and T(s) (Red).",
                leg=("Nominal:" + str(fbc.nominal_sensitivity) + " dB/dec",), yl=(-80, 10))

    ##########Plot5##########
    fig += 1
    try:
        c = fbc.freqresp(obj="pid")[:F]
        myplot.bodeplot(fig, o[:F] / 2 / np.pi, 20 * np.log10(abs(c)), np.angle(c, deg=True), line_style='-')
    except:
        pass
    try:
        c = fbc.freqresp(obj="fir")[:F]
        myplot.bodeplot(fig, o[:F] / 2 / np.pi, 20 * np.log10(abs(c)), np.angle(c, deg=True), line_style='-')
    except:
        pass
    myplot.save(fig, title='Controllers', leg=("PID", "FIR"))

    return fig


if __name__ == "__main__":
    import os

    try:
        os.mkdir(DATA)
    except:
        pass

    fig = -1

    fig, o, h = plant(fig)
    fig, fbc = optimize(fig, o, h)
    plotall(fig, fbc)

    myplot.show()
