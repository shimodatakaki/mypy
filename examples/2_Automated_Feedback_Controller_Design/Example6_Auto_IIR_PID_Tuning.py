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

    NSTBITER = 1

    if True:
        # PID Triple Poleassignment 400 Hz
        rhon = np.array([0.000101889491467777,
                         0.170717481532357,
                         107.264957164280])
        rhod = np.array([9.94718394324346 * 10 ** (-5), 1])
    elif False:
        # PID 400 Hz  + 5400 Hz Notch
        rhon = np.array([0.000178306610068610,
                         0.846420925630186,
                         368574.319724049,
                         1077291892.56301,
                         1183397129740.58])
        rhod = np.array([5.68410511042483e-05,
                         1.51578813420334,
                         126083.578322355,
                         2058536168.05419])
    elif False:
        rhon = np.array([0.000127361864334721,
                         0.787836113815149,
                         676560.592842885,
                         2760708571.14940,
                         853187331902783,
                         1.78089599036966e+18,
                         1.39697403340072e+21
                         ])
        rhod = np.array([7.95774715459477e-05,
                         4.43955832426420,
                         489464.466739361,
                         13540677611.4662,
                         630318001739791,
                         6.66807430879530e+18
                         ])
    else:
        # PID 300 Hz + Phase Compensator + Notch
        rhon = np.array([7.64171186008328e-05,
                         6.23379530477791,
                         175952.995822072,
                         12564320886.3040,
                         15618222242330.9,
                         7.31297798873055e+15
                         ])
        rhod = np.array([0.000132629119243246,
                         5.51210670192588,
                         337065.291819523,
                         9095778966.23193,
                         51352787209705.0
                         ])

    f = 300
    _f = [0]
    _c = [None]
    rho = None
    rho_best = rho
    R = 1.1
    LAMBDA = (1 + R) / R / 2
    tol = 10
    while tol > 0:
        print("Try: ", f, " Hz")

        fbc = IIRControllerDesign(o, g, ts=TS, rhon=rhon, rhod=rhod)
        F_DGC = 2 * np.pi * f  # Desired Cross-over Frequency (rad/s)
        fbc.specification(F_DGC, THETA_DPM, GDB_DGM, theta_dpm2=THETA_DPM2)  # set constraints

        for i in range(NSTBITER):
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
            fbc.reset()

            taud = fbc.rhod[0] / fbc.rhod[1]
            pid = btaud2pid([x / fbc.rhod[1] for x in fbc.rhon], taud)
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

    for e in range(11, 0, -1):
        print((11 - e) * ' ' + e * '*')
    print('')
    for g in range(11, 0, -1):
        print(g * ' ' + (11 - g) * '*')

    assert _f[-1] > 0
    i_max = [i for i, f in enumerate(_f) if f == max(_f)][0]
    print("Best nominal frequency:", _f[i_max], " Hz")
    fbc = _c[i_max]
    taud = fbc.rhod[0] / fbc.rhod[1]
    pid = btaud2pid([x / fbc.rhod[1] for x in fbc.rhon], taud)
    print("rho:", fbc.rho)
    print("PIDs:", pid)
    print("TAU:", taud)
    mycsv.save([*pid, taud], save_name=DATA + "/rho" + str(fig) + ".csv",
               header=("taud (s):" + str(taud),
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
