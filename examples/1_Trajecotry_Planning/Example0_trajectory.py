"""
TEST
"""
from mycvxopt import *
from mytrajectory import *


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


"""
2 NORM MINIMAZATION
"""
def test2():
    import sys
    print(sys.version)

    """
    CONDITION AND CONSTRAINTS
    """
    TS = 0.001
    TINIT = (0, 1)
    START, END = 0, 1
    TSAMPLE = np.linspace(0, TINIT[END], TINIT[END] / TS + 1)
    F = len(TSAMPLE)
    AMAX = 600
    VMAX = 100
    QC = (0, 1.0)
    VC = (0, 0)
    AC = (0, 0)

    POS = 0
    VEL = 1
    ACC = 2
    JER = 3
    SNA = 4
    STATES = (POS, VEL, ACC, JER, SNA)
    LABELS = ("Pos.", "Vel.", "Acc.", "Jer.", "Sna.")

    MIN = JER

    nc = 9 + 32  # number of control points == number of theta

    p = 4  # jerk continous
    u = [TINIT[END] * i / (nc - 1) for i in range(nc)]  # knots

    bspl = Bspline(u, nc, p, verbose=True)

    """
    QUDRATIC PROGRAMMING
    """
    M = np.array(bspl.basis(TSAMPLE, der=MIN))
    P = np.dot(M.T, M)
    q = np.zeros(nc)
    A = np.array([*bspl.basis(TINIT, der=POS),
                  *bspl.basis(TINIT, der=VEL),
                  *bspl.basis(TINIT, der=ACC)])
    b = np.matrix([*QC, *VC, *AC]).reshape((6, 1))
    G = np.array([*bspl.basis(TSAMPLE, der=VEL),
                  *bspl.basis(TSAMPLE, der=ACC)])
    hv = np.ones(F) * VMAX
    ha = np.ones(F) * AMAX
    h = np.append(hv, ha, axis=0)
    G = np.append(G, -G, axis=0)
    h = np.append(h, h, axis=0)

    print("P:")
    print(P.shape)
    print(P)
    print("q:")
    print(q.shape)
    print(q)
    print("G:")
    print(G.shape)
    print(G)
    print("h:")
    print(h.shape)
    print(h)
    print("A:")
    print(A.shape)
    print(A)
    print("b:")
    print(b.shape)
    print(b)

    theta = cvxopt_solve_qp(P, q, G, h, A, b)
    theta = cvxopt_solve("qp", [P, q], G=G, h=h, A=A, b=b)
    print(theta)

    for d in STATES:
        fig, ax = plt.subplots()
        # TS_UP = 10 ** -4
        # TSAMPLE = np.linspace(0, TINIT[END], TINIT[END] / TS + 1)
        ax.plot(TSAMPLE, bspl.bspline(theta, TSAMPLE, der=d), 'r-', lw=3, label=LABELS[d])
        ax.grid(True)
        ax.legend(loc='best')

    plt.show()


def test3():
    c = matrix([-4., -5.])
    G = matrix([[2., 1., -1., 0.], [1., 2., 0., -1.]])
    h = matrix([3., 3., 0., 0.])
    sol = solvers.lp(c, G, h)
    print(sol['x'])

"""
INFINITY NORM MINIMAZATION
"""
def test4():
    import sys
    print(sys.version)

    """
    CONDITION AND CONSTRAINTS
    """
    TS = 0.01
    TINIT = (0, 1)
    START, END = 0, 1
    TSAMPLE = np.linspace(0, TINIT[END], TINIT[END] / TS + 1)
    F = len(TSAMPLE)
    QC = (0, 1.0)
    VC = (0, 0)
    AC = (0, 0)
    JC = (0, 0)
    NCONSTRAINTS = sum(len(x) for x in (QC, VC, AC, JC))

    POS = 0
    VEL = 1
    ACC = 2
    JER = 3
    SNA = 4
    STATES = (POS, VEL, ACC, JER, SNA)
    LABELS = ("Pos.", "Vel.", "Acc.", "Jer.", "Sna.")

    MIN = JER

    nc = NCONSTRAINTS + 48  # number of control points == number of theta

    p = 4  # jerk continous
    u = [TINIT[END] * i / (nc - 1) for i in range(nc)]  # knots

    bspl = Bspline(u, nc, p, verbose=True)
    """
    INFINITY NORM OPTIMIZATION
    """
    c = np.array([1])
    c = np.append(c, np.zeros(nc), axis=0)

    G = np.array([])
    for x in bspl.basis(TSAMPLE, der=MIN):
        xx = [_x * -1 for _x in x]
        G = np.append(G, np.array([1, *xx]))
    for xx in bspl.basis(TSAMPLE, der=MIN):
        G = np.append(G, np.array([1, *xx]))
    G = - G
    G = G.reshape(2 * F, 1 + nc)

    h = np.zeros(2 * F)

    A = np.array([])
    for d in (POS, VEL, ACC, JER):
        for x in bspl.basis(TINIT, der=d):
            A = np.append(A, np.array([0, *x]))
    A = A.reshape(NCONSTRAINTS, 1 + nc)

    b = np.matrix([*QC, *VC, *AC, *JC]).reshape((NCONSTRAINTS, 1))

    print("c:")
    print(c)
    print(c.shape)
    print("G:")
    print(G)
    print(G.shape)
    print("h:")
    print(h)
    print(h.shape)
    print("A:")
    print(A)
    print(A.shape)
    print("b:")
    print(b)
    print(b.shape)

    # c, G, h, A, b = matrix(c), matrix(G), matrix(h), matrix(A), matrix(b)
    # sol = solvers.lp(c, G, h, A, b)
    # theta_ex = np.array(sol['x']).reshape(1+nc,)

    # MAX_ITER_SOL = 10
    # for i in range(MAX_ITER_SOL):
    #     new_opt = {}
    #     for k, v in opt.items():
    #         new_opt[k] = v *  10 ** i
    #
    #     theta_ex = cvxopt_solve_lp(c, G, h, A=A, b=b, opt=new_opt)
    #
    #     if theta_ex is not None:
    #         break
    #     print(new_opt)
    #     print("This condition is ill. Retrieving the conditon by 1000%...")
    # if theta_ex is None:
    #     print("solver not feasible!!!")
    #     exit()


    theta_ex = cvxopt_solve("lp", [c], G=G, h=h, A=A, b=b)
    print(theta_ex)

    theta = theta_ex[1:]
    print("theta:")
    print(theta)
    print("tau")
    print(theta_ex[0])

    for d in STATES:
        fig, ax = plt.subplots()
        # TS_UP = 10 ** -4
        # TSAMPLE = np.linspace(0, TINIT[END], TINIT[END] / TS + 1)
        ax.plot(TSAMPLE, bspl.bspline(theta, TSAMPLE, der=d), 'r-', lw=3, label=LABELS[d])
        ax.grid(True)
        ax.legend(loc='best')

    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test4()
