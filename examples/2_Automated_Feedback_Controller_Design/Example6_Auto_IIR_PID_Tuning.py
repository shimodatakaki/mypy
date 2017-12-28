"""

"""

from myfbcd import *
import myplot
import mycsv

DATA = "data/example6_result"
TS = 50 * 10 ** (-6)  # sampring of FIRs
TD = 3 / 10 * TS  # Delay
NDATA = 1  # number of data


def optimize(fig, o, g, datapath=DATA):
    """
    calc pids and firs
    :param o:
    :param g:
    :return:
    """
    THETA_DPM = 30 / 180 * np.pi  # Phase Margin
    THETA_DPM2 = 30 / 180 * np.pi  # Second Phase Margin
    GDB_DGM = 5  # Gain Margin in (dB)

    NSTBITER = 3

    TAUD = 2 * TS  # Pseudo Differential Cut-off for D Control
    NOFIR = 0  # Only PID
    NOPID = "pid"

    f = 150
    _f = [0]
    _c = [None]
    rho = None
    rhon = []
    rhod = []
    rho_best = rho
    R = 1.5
    LAMBDA = (1 + R) / R / 2
    tol = 15
    while tol > 0:
        F_DGC = 2 * np.pi * f  # Desired Cross-over Frequency (rad/s)
        print("Try: ", f, " Hz")
        for i in range(NSTBITER):
            fbc = IIRControllerDesign(o, g, ts=TS, rhon=rhon, rhod=rhod)
            fbc.specification(F_DGC, THETA_DPM, GDB_DGM, theta_dpm2=THETA_DPM2)  # set constraints
            fbc.nominalcond(db=-60)  # append nominal performance condition
            fbc.stabilitycond()  # append stability condition
            fbc.gainpositivecond()  # append gain constraints
            try:
                rho = fbc.optimize()
            except:
                tol -= 1
                f = f * LAMBDA
                rho = rho_best
                break

            rhon = fbc.rhon
            rhod = fbc.rhod
            fbc.controller()
            fbc.openloop()
            taud = fbc.rho[-1]
            pid = btaud2pid(fbc.rho[:3], taud)
            print("rho:", fbc.rho)
            print("PIDs:", pid)
            print("TAU:", taud)
            print()

            if i >= NSTBITER // 2 and check_disk(fbc.L, fbc.rm, fbc.sigma):
                print("Solver found a local minima @ iteration", i)
                if f > max(_f):
                    rho_best = rho
                    print("best @", f)
                    print()
                _f.append(f)
                _c.append(fbc)
                f *= R
                break
        # if i == NSTBITER-1:
        #     break
        if f > 500:
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
    taud = fbc.rho[-1]
    pid = btaud2pid(fbc.rho[:3], taud)
    print("rho:", fbc.rho)
    print("PIDs:", pid)
    print("TAU:", taud)
    mycsv.save([*pid, taud], save_name=DATA + "/rho" + str(fig) + ".csv",
               header=("P,I,D, FIR(1+n) for n in range(" + str(NOFIR) + ")", "taud (s):" + str(TAUD),
                       "FIR sampling (s):" + str(TS)))

    return fig, fbc


if __name__ == "__main__":
    from sympy import *
    import matplotlib.pyplot as plt

    s = symbols('s')

    import os

    try:
        os.makedirs(DATA)
    except:
        pass

    fig = -1

    import Example1_Single_FRF_Nano_Scale_Servo as ex1

    fig, o, h = ex1.plant(fig, datapath=DATA)
    fig, fbc = optimize(fig, o, h)
    ex1.plotall(fig, fbc, ndata=NDATA, datapath=DATA)

    myplot.show()
