"""

"""

from myfbcd import *
import myplot
import mycsv

DATA = "data/example4_result"
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

    NSTBITER = 4

    TAUD = 5 * TS  # Pseudo Differential Cut-off for D Control
    NOFIR = 0  # Only PID
    NOPID = "pid"

    f = 10
    _f = [0]
    _c = [None]
    rho = None
    rho_best = rho
    R = 1.5
    LAMBDA = (1 + R) / R / 2
    tol = 15
    while tol > 0:
        F_DGC = 2 * np.pi * f  # Desired Cross-over Frequency (rad/s)
        print("Try: ", f, " Hz")
        for i in range(NSTBITER):
            fbc = ControllerDesign(o, g, nopid=NOPID, taud=TAUD, nofir=NOFIR, ts=TS, tsfir=TS, rho0=rho)
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
            if i >= NSTBITER // 2 and check_disk(np.dot(fbc.X, fbc.rho), fbc.rm, fbc.sigma):
                print("Solver found a local minima @ iteration", i)
                if f > max(_f):
                    rho_best = rho
                    print("best @", f)
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
    mycsv.save(fbc.rho, save_name=DATA + "/rho" + str(fig) + ".csv",
               header=("P,I,D, FIR(1+n) for n in range(" + str(NOFIR) + ")", "taud (s):" + str(TAUD),
                       "FIR sampling (s):" + str(TS)))

    return fig, fbc


if __name__ == "__main__":
    from sympy import *

    s = symbols('s')

    import os

    try:
        os.mkdir(DATA)
    except:
        pass

    fig = -1

    import Example1_Single_FRF_Nano_Scale_Servo as ex1

    fig, o, h = ex1.plant(fig, datapath=DATA)
    fig, fbc = optimize(fig, o, h)
    ex1.plotall(fig, fbc, ndata=NDATA, datapath=DATA)

    myplot.show()
