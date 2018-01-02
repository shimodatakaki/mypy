"""

"""

from myfbcd import *
import myplot
import mycsv

DATA = "data/example10_result"

F = 1000  # number of FRF lines
TS = 50 * 10 ** (-6)  # sampring of FIRs
TD = 3 / 10 * TS  # Delay
PERT = (0, -0.1, 0.1, 0.2, -0.2)  # perturbation
NDATA = 2 * len(PERT) - 1


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

    rho = [np.array([
        0.000101889491467777,
        0.170717481532357,
        107.264957164280,
        9.94718394324346e-05,
        1]),
        np.array([
            9.074218794043245e+03 * 1 / 1,
            2.058536168054191e+09,
            9.074218794043245e+03]),
        np.array([
            1.138284861945496e+04 * 1 / 1,
            3.239231067335693e+09,
            1.138284861945496e+04]),
        np.array([
            5.716884426916397e+03 * 1 / 1,
            8.170691887679803e+08,
            5.716884426916397e+03]),
        np.array([
            2 * np.pi * 3500,
            2 * np.pi * 3500])
    ]
    rho0 = rho
    f = 300
    _f = [0]
    _c = [None]
    rho_best = rho
    R = 1.1
    LAMBDA = (1 + R) / R / 2
    tol = 20
    while tol > 0:
        # rho = rho0

        print("Try: ", f, " Hz")
        fbc = LFControllerDesign(o, g, TS, rho, ctype=["pid", 3, 1])
        F_DGC = 2 * np.pi * f  # Desired Cross-over Frequency (rad/s)
        fbc.specification(F_DGC, THETA_DPM, GDB_DGM, theta_dpm2=THETA_DPM2)  # set constraints
        fbc.controller()

        for i in range(NSTBITER):

            fbc.nominalcond(db=-60)  # append nominal performance condition
            fbc.stabilitycond()  # append stability condition
            fbc.gainpositivecond()  # append gain constraints
            fbc.gaincond()
            try:
                rho = fbc.optimize()
            except:
                tol -= 1
                f = f * LAMBDA
                rho = rho_best
                break
            fbc.reset()
            fbc.controller()

            taud = fbc.rho[0][-2] / fbc.rho[0][-1]
            pid = btaud2pid([x / fbc.rho[0][-1] for x in fbc.rho[0][:3]], taud)
            print("rho:", fbc.rho[0])
            print("PIDs:", pid)
            print("TAU:", taud)
            print()
            for ino in range(fbc.nonotch):
                d1, d2, c1 = fbc.rho[1 + ino]
                on = np.sqrt(d2)
                zeta = c1 / 2 / on
                d = d1 / c1
                print("rho:", fbc.rho[1 + ino])
                print("on: ", on)
                print("zeta:", zeta)
                print("d:", d)
                print()
            if fbc.nopc:
                print("rho:", fbc.rho[fbc.nonotch + 1])
                print("2*pi*fnum ", fbc.rho[fbc.nonotch + 1][0])
                print("2*pi*fden:", fbc.rho[fbc.nonotch + 1][1])
                print()

            if i >= NSTBITER // 2:
                if check_disk(fbc.L, fbc.rm, fbc.sigma):
                    print("Solver found a local minima @ iteration", i)
                    if f > max(_f):
                        rho_best = rho
                        print("best @", f)
                        print()
                        _f.append(f)
                        _c.append(fbc)
                    f *= R
                    break
                else:
                    # rho = rho0
                    print("stability condition violation")
                    tol -= 1

            print("-" * 50)
            print()

    for e in range(11, 0, -1):
        print((11 - e) * ' ' + e * '*')
    print('')
    for g in range(11, 0, -1):
        print(g * ' ' + (11 - g) * '*')

    assert _f[-1] > 0
    i_max = [i for i, f in enumerate(_f) if f == max(_f)][0]
    print("Best nominal frequency:", _f[i_max], " Hz")
    fbc = _c[i_max]

    print("rho:", fbc.rho)

    taud = fbc.rho[0][-2] / fbc.rho[0][-1]
    pid = btaud2pid([x / fbc.rho[0][-1] for x in fbc.rho[0][:3]], taud)
    print("PIDs:", pid)
    print("TAU:", taud)
    print()
    for ino in range(fbc.nonotch):
        d1, d2, c1 = fbc.rho[1 + ino]
        on = np.sqrt(d2)
        zeta = c1 / 2 / on
        d = d1 / c1
        print("rho:", fbc.rho[1 + ino])
        print("on: ", on)
        print("zeta:", zeta)
        print("d:", d)
        print()
    if fbc.nopc:
        print("rho:", fbc.rho[fbc.nonotch + 1])
        print("2*pi*fnum ", fbc.rho[fbc.nonotch + 1][0])
        print("2*pi*fden:", fbc.rho[fbc.nonotch + 1][1])
        print()

    mycsv.save([y for x in rho for y in x], save_name=DATA + "/rho" + str(fig) + ".csv",
               header=())

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
    import Example3_Another_Various_FRFs_Nano_Scale_Servo as ex3

    fig, o, h = ex3.plant_data(fig, datapath=DATA)
    fig, fbc = optimize(fig, o, h)
    ex1.plotall(fig, fbc, ndata=NDATA, datapath=DATA)

    myplot.show()
