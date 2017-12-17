"""
Example1: Example1_optmized_PID_and_FIR_Notch
"""

from myfbcd import *

from scipy import signal
import myplot
import mycvxopt
import mysignal

DATA = "data"


def plant(fig, n):
    """
    return FRF
    :return:
    """
    s = symbols('s')

    if n == 0:
        P = (50 * 2 * np.pi) ** 2 / (s ** 2 + 2 * 0.01 * 50 * 2 * np.pi * s + (50 * 2 * np.pi) ** 2)
    elif n == 1:
        """
        Rigid: 1/s/(1+0.1*s)
        Vibration: Resonances @ 100, 200, 300, 400, 600, 800 Hz; Anti-Resonance @ 250 Hz
        """
        P = (1000 / s / (1 + 0.1 * s)
             * (100 * 2 * np.pi) ** 2 / (s ** 2 + 2 * 0.01 * 100 * 2 * np.pi * s + (100 * 2 * np.pi) ** 2)
             * (s ** 2 + 2 * 0.01 * 250 * 2 * np.pi * s + (250 * 2 * np.pi) ** 2) / (250 * 2 * np.pi) ** 2 *
             (200 * 2 * np.pi) ** 2 / (s ** 2 + 2 * 0.01 * 200 * 2 * np.pi * s + (200 * 2 * np.pi) ** 2) *
             (300 * 2 * np.pi) ** 2 / (s ** 2 + 2 * 0.01 * 300 * 2 * np.pi * s + (300 * 2 * np.pi) ** 2) *
             (400 * 2 * np.pi) ** 2 / (s ** 2 + 2 * 0.01 * 400 * 2 * np.pi * s + (400 * 2 * np.pi) ** 2) *
             (600 * 2 * np.pi) ** 2 / (s ** 2 + 2 * 0.01 * 600 * 2 * np.pi * s + (600 * 2 * np.pi) ** 2) *
             (800 * 2 * np.pi) ** 2 / (s ** 2 + 2 * 0.01 * 800 * 2 * np.pi * s + (800 * 2 * np.pi) ** 2)
             )

    p = mysignal.symbolic_to_tf(P, s)

    o = 2 * np.pi * np.logspace(0, 3, num=1000)
    o, mag, phase = p.bode(w=o)
    theta = phase / 180 * np.pi
    a = 10 ** (mag / 20)
    h = a * np.exp(1.j * theta)
    fig += 1
    myplot.bode(p, fig, w=o, save_name=DATA + "/" + str(fig) + "_plant" + str(n), leg=("plant1",))

    return fig, o, h


def optimize(fig, o, g, nofir=10, f_desired_list=[30 + 4 * i for i in range(25)]):
    """
    calc pids and firs
    :param o:
    :param p:
    :return:
    """
    THETA_DPM = 30 / 180 * np.pi  # Phase Margin
    THETA_DPM2 = 45 / 180 * np.pi  # Second Phase Margin
    GDB_DGM = 10  # Gain Margin in (dB)
    SIGMA = -1
    RM = np.sqrt((1 + SIGMA) ** 2 - 2 * SIGMA * (1 - np.cos(THETA_DPM2)))
    NSTBITER = 5
    is_robust = False
    GAMMA = 0.5

    TAUD = 4 * 10 ** (-3)  # Pseudo Differential Cut-off for D Control
    NOFIR = nofir  # Number of FIRs
    NOPID = "pid"
    TS = 0.5 * 10 ** (-3)  # sampring of FIRs

    _f = 0
    _c = None
    tol = 3
    for f in f_desired_list:
        F_DGC = 2 * np.pi * f  # Desired Cross-over Frequency
        print("Try: ", f, " Hz")
        fbc = ControllerDesign(o, g, nopid=NOPID, taud=TAUD, nofir=NOFIR, ts=TS)
        fbc.specification(F_DGC, THETA_DPM, GDB_DGM, theta_dpm2=THETA_DPM2)  # set constraints
        try:
            for i in range(NSTBITER):
                fbc.gccond()  # append crossover-gain constraint
                fbc.pmcond()  # append phase margin constraint
                fbc.gmcond()  # append gain margin constraint, more robust if nupper = 1 / 20
                fbc.pm2cond()  # append second phase margin constraint
                fbc.gaincond()  # append gain constraints
                fbc.stabilitycond(rm=RM, sigma=SIGMA) #append stability condition
                if is_robust:
                    fbc.robustcond(GAMMA) #may need much time
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

    exit()

    ##########Plot1##########
    L = np.dot(fbc.X, rho)

    fig += 1

    # Nyquist
    myplot.time(np.real(L), np.imag(L), fig, lw=6)
    _theta = np.linspace(0, 2 * np.pi, 100)
    myplot.time((np.cos(_theta)), (np.sin(_theta)), fig, line_style="r--", lw=1)  # r=1
    myplot.time((0,), (0,), fig, line_style="r+")  # origin
    myplot.time((-1,), (0,), fig, line_style="r+")  # critical

    # Gain-Crossover
    _l = np.array([L[i] for i in fbc.l_gc])
    _x = np.real(_l)
    myplot.time(np.real(_l), np.imag(_l), fig, line_style="m-.")
    myplot.time(_x, - np.tan(fbc.phi_dgc) * _x - 1 / np.cos(fbc.phi_dgc), fig, line_style="m-.")

    # Phase Margin
    _l = np.array([L[i] for i in fbc.l_pm])
    _x = np.real(_l)
    myplot.time(_x, np.imag(_l), fig, line_style="g--")
    myplot.time(_x, np.tan(fbc.theta_dpm) * _x, fig, line_style="g--")

    # Gain Margin
    cx, cy = - np.cos(fbc.phi_dgm), -np.sin(fbc.phi_dgm)
    a = cy / (cx + 1 / fbc.g_dgm)
    b = a * (1 / fbc.g_dgm)
    _l = np.array([L[i] for i in fbc.l_gm])
    _x = np.real(_l)
    myplot.time(np.real(_l), np.imag(_l), fig, line_style="r:")  # gain margin
    myplot.time(_x, a * _x + b, fig, line_style="r:")  # gain margin

    # Second Phase Margin
    _l = np.array([L[i] for i in fbc.l_pm2])
    _y = np.imag(_l)
    myplot.time(np.real(_l), np.imag(_l), fig, line_style="y--")
    myplot.time(- np.cos(fbc.theta_dpm2) * np.ones(len(_y)), _y, fig, line_style="y--")

    # Stanility Constraint
    _l = np.array([L[i] for i in fbc.l_stb])
    myplot.time((RM * np.cos(_theta)) + SIGMA, RM * np.sin(_theta), fig, line_style="y:",
                xl=[-3, 3], yl=[-3, 3], leg=("Nyquist", "r=1", "origin", "critical", "G.C. Cond.", "G.C. Cond.",
                                             "P.M. Cond.", "P.M. Cond.", "G.M. Cond.", "G.M. Cond.",
                                             "P.M.2 Cond.", "P.M.2 Cond.", "Stb. Cond.", "Stb. Cond."),
                label=("Re", "Im"), save_name=DATA + "/" + str(fig) + "_nyquist",
                text=(-3, 3.1, "Optimized Gain-Crossover Frequency (Hz): " + str(_f)))  # r=1

    if is_robust:
        #Robust Constraint
        _l = np.array([L[i] for i in fbc.l_rbs])
        myplot.time((GAMMA * np.cos(_theta)), GAMMA * np.sin(_theta), fig, line_style="y-.",
                    xl=[-3, 3], yl=[-3, 3], leg=("Nyquist", "r=1", "origin", "critical", "G.C. Cond.", "G.C. Cond.",
                                                 "P.M. Cond.", "P.M. Cond.", "G.M. Cond.", "G.M. Cond.",
                                                 "P.M.2 Cond.", "P.M.2 Cond.", "Stb. Cond.", "Rbs. Cond."),
                    label=("Re", "Im"), save_name=DATA + "/" + str(fig) + "_nyquist",
                    text=(-3, 3.1, "Optimized Gain-Crossover Frequency (Hz): " + str(_f)))  # r=1

    plt.grid()

    ##########Plot2##########
    fig += 1

    # # Gain Margin
    # _l = np.array([L[i] for i in fbc.l_gm])
    # _x = np.real(_l)
    # _db =[10*np.log10(_x[i]**2 + (a * _x[i] + b)**2) for i in range(len(fbc.l_gm))]
    # myplot.bodeplot([o[i] / 2 / np.pi for i in fbc.l_gm], _db, None, fig, nos=1,
    #                 line_style="r:", lw=1)
    # Phase Margin
    myplot.bodeplot([o[i] / 2 / np.pi for i in fbc.l_pm], None, [-180 + THETA_DPM / np.pi * 180 for i in fbc.l_pm], fig,
                    nos=-2, line_style="g--", lw=1)
    # Gain-Crossover
    ldb = 20 * np.log10(abs(L))
    myplot.bodeplot([_f for l in L], np.linspace(min(ldb), max(ldb), num=len(L)),
                    np.linspace(-180, 180, num=len(L)), fig, nos=2, line_style="m-.", lw=1)
    # Second Phase Margin
    myplot.bodeplot([o[i] / 2 / np.pi for i in fbc.l_pm2], None, [180 - THETA_DPM2 / np.pi * 180 for i in fbc.l_pm2],
                    fig,
                    nos=-2, line_style="y-.", lw=1)
    myplot.bodeplot([o[i] / 2 / np.pi for i in fbc.l_pm2], None, [-180 + THETA_DPM2 / np.pi * 180 for i in fbc.l_pm2],
                    fig,
                    nos=-2, line_style="y-.", lw=1)
    # Plot
    myplot.bodeplot(o / 2 / np.pi, ldb, np.angle(L, deg=True), fig, line_style='b-',
                    save_name=DATA + "/" + str(fig) + "_bode",
                    leg=("P.M.", str(_f) + " Hz", "P.M.2", "P.M.2", "Bode"))
    plt.grid()
    ##########Plot3##########
    fig += 1
    c = fbc.freqresp()
    myplot.bodeplot(o / 2 / np.pi, 20 * np.log10(abs(c)), np.angle(c, deg=True), fig, line_style='b-',
                    save_name=DATA + "/" + str(fig) + "_controller")

    return fig


if __name__ == "__main__":
    from sympy import *
    import matplotlib.pyplot as plt
    import os

    try:
        os.mkdir(DATA)
    except:
        pass

    fig = -1

    fig, o, h = plant(fig, 1)

    fig = optimize(fig, o, h, nofir=0, f_desired_list=[1 + i for i in range(50)]) #PID Only
    fig = optimize(fig, o, h, nofir=10) #PID + 10 FIRs

    myplot.show()
