"""
TEST
"""
from mytrajectory import *
import mycvxopt
import myplot

POS = 0
VEL = 1
ACC = 2
JER = 3
SNA = 4
STATES = (POS, VEL, ACC, JER, SNA)
LABELS = ("Position (m)", "Velocity (m/s)", "Acceleration (m/s/s)", "Jerk (m/s/s/s)", "Snap (m/s/s/s/s)")


def min_infinity_norm(fig=0):
    """
    Infinity Norm Minimization
    :return:
    """

    """
    CONDITION AND CONSTRAINTS
    """
    MIN = JER  # What to be minimized

    TS = 0.001  # Sampling Period
    TINIT = (0, 1)
    START, END = 0, 1
    TSAMPLE = np.linspace(0, TINIT[END], TINIT[END] / TS + 1)

    QC = (0, 1.0)
    VC = (0, 0)
    AC = (0, 0)
    JC = (0, 0)
    CONSTRAINTS = (*QC, *VC, *AC, *JC)
    LABEL_CONSTRAINTS = (POS, VEL, ACC, JER)

    NCONSTRAINTS = len(CONSTRAINTS)

    assert NCONSTRAINTS == len(LABEL_CONSTRAINTS) * 2

    nc = NCONSTRAINTS + 60  # number of control points == number of theta

    p = 4  # jerk continous
    u = [TINIT[END] * i / (nc - 1) for i in range(nc)]  # knots
    bspl = Bspline(u, nc, p, verbose=True)

    """
    Optimization
    """
    F = np.array(bspl.basis(TSAMPLE, der=MIN))
    f = np.zeros(len(TSAMPLE)).reshape((len(TSAMPLE), 1))
    A = None
    for d in LABEL_CONSTRAINTS:
        for x in bspl.basis(TINIT, der=d):
            if A is None:
                A = np.array(x)
            else:
                A = np.block([[A], [np.array(x)]])
    b = np.array(CONSTRAINTS).reshape((NCONSTRAINTS, 1))

    tau, theta = mycvxopt.solve_min_infinity_norm(F, f, A=A, b=b)
    print("theta:")
    print(theta)

    """
    Plot
    """
    fig = 0
    for d in STATES:
        fig += 1
        text = None
        if d == MIN:
            text = (0, tau + 5, "Minimized Infinity Norm: " + str(tau))
            myplot.time(TSAMPLE, bspl.bspline(theta, TSAMPLE, der=d), fig, line_style="r-")
            myplot.time(TSAMPLE, [tau for _ in range(len(TSAMPLE))], fig, line_style="r--", lw=1)
            myplot.time(TSAMPLE, [-tau for _ in range(len(TSAMPLE))], fig, line_style="r--", lw=1, text=text,
                        save_name="data/i_norm_" + LABELS[d].split()[0], label=("Time (s)", LABELS[d]))
        else:
            myplot.time(TSAMPLE, bspl.bspline(theta, TSAMPLE, der=d), fig, text=text,
                        save_name="data/i_norm_" + LABELS[d][:-1].split()[0], label=("Time (s)", LABELS[d]),
                        line_style="r-")


def min_2norm(fig=0):
    """
    2 norm optimization
    :return:
    """

    """
    CONDITION AND CONSTRAINTS
    """
    MIN = JER  # What to be minimized

    TS = 0.001  # Sampling Period
    TINIT = (0, 1)
    START, END = 0, 1
    TSAMPLE = np.linspace(0, TINIT[END], TINIT[END] / TS + 1)

    QC = (0, 1.0)
    VC = (0, 0)
    AC = (0, 0)
    JC = (0, 0)
    CONSTRAINTS = (*QC, *VC, *AC, *JC)
    LABEL_CONSTRAINTS = (POS, VEL, ACC, JER)

    NCONSTRAINTS = len(CONSTRAINTS)

    assert NCONSTRAINTS == len(LABEL_CONSTRAINTS) * 2

    VMAX = 10000
    AMAX = 10000

    nc = NCONSTRAINTS + 60  # number of control points == number of theta

    p = 4  # jerk continous
    u = [TINIT[END] * i / (nc - 1) for i in range(nc)]  # knots
    bspl = Bspline(u, nc, p, verbose=True)

    """
    QUDRATIC PROGRAMMING
    """
    M = np.array(bspl.basis(TSAMPLE, der=MIN))
    P = np.dot(M.T, M)
    q = np.zeros(nc)
    A = None
    for d in LABEL_CONSTRAINTS:
        for x in bspl.basis(TINIT, der=d):
            if A is None:
                A = np.array(x)
            else:
                A = np.block([[A], [np.array(x)]])
    b = np.matrix(CONSTRAINTS).reshape((NCONSTRAINTS, 1))
    G = np.array([*bspl.basis(TSAMPLE, der=VEL),
                  *bspl.basis(TSAMPLE, der=ACC)])
    hv = np.ones((len(TSAMPLE), 1)) * VMAX
    ha = np.ones((len(TSAMPLE), 1)) * AMAX
    h = np.append(hv, ha, axis=0)
    G, h = mycvxopt.constraints(G, h, -h)

    theta = mycvxopt.solve("qp", [P, q], G=G, h=h, A=A, b=b)
    print(theta)

    """
    Plot
    """
    for d in STATES:
        fig += 1
        text = None
        myplot.time(TSAMPLE, bspl.bspline(theta, TSAMPLE, der=d), fig, text=text,
                    save_name="data/" + LABELS[d][:-1].split()[0], label=("Time (s)", LABELS[d]))

    myplot.show()


def test():
    import sys
    print(sys.version)

    M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
    P = np.dot(M.T, M)
    q = np.dot(np.array([3., 2., 3.]), M).reshape((3,))
    G = np.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
    h = np.array([3., 2., -2.]).reshape((3,))

    opt = {'abstol': 10 ** -5}

    print(cvxopt_solve_qp(P, q, G, h, opt=opt))

    print(cvxopt_solve("qp", [P, q], G=G, h=h))


def test3():
    c = matrix([-4., -5.])
    G = matrix([[2., 1., -1., 0.], [1., 2., 0., -1.]])
    h = matrix([3., 3., 0., 0.])
    sol = solvers.lp(c, G, h)
    print(sol['x'])


if __name__ == "__main__":
    import os

    try:
        os.mkdir("data")
    except:
        pass

    min_infinity_norm()
    min_2norm()

    myplot.show()
