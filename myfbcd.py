"""

"""
import mycvxopt
import numpy as np
from scipy import signal
import mysignal
import mynum


class ControllerDesign():
    """
    Find a linear FB controller that satisfies desired (given) linear constraints for FB,
    i.e. (1) Gain-Crossover Linear Inequalities (self.o_dgc, self.phi_dgc),
         (2) Phase Margin Linear Inequalities (self.theta_dpm),
         (3) Gain Margin Linear Inequalities (self.g_dgm, self.phi_dgm),
         (4) Second Phase Margin Linear Inequalities (self.theta_dpm2),
         (5) Gain Minimum/Maximum Linear Equalities/Inequalities,
         (6) Stability Margin (Disk) Concave Inequalities via CCCP method,
         (7) Robust Stability Quadratic Inequalities (using socp or sdp),
         (8) Nominal Performance (Disk) Concave Inequalities via CCCP method.
    Default Controller: PIDs + 50 FIRs (53 variables)
    """

    def __init__(self, o, g, nopid="pid", taud=0.01, nofir=10, ts=0.001, tsfir=0.001, rho0=None, is_notch=False):
        """
        L = P*C = X*rho, where X = g*phi.T. phi:Basis FRF of controller, rho:Controller parameters.
        x, y = real(L), imag(L)
        :param o: frequnecy (rad/s)
        :param g: FRF of plant (1)
        :param nopid:
        :param taud:
        :param nofir:
        :param ts:
        """
        self.o = o
        self.g = g
        self.l = [i for i in range(len(o))]
        self.F = len(o)
        self.nopid = nopid
        self.nofir = nofir
        self.NOC = len(nopid) + nofir
        self.phi = np.array([
                                np.array([*[c(_o) for c in pid(nopid, taud=taud, ts=ts)],
                                          *[c(_o) for c in fir(nofir=nofir, ts=tsfir, is_notch=is_notch)]])
                                for _o in o])
        self.X = np.array([_g * _phi for _g, _phi in zip(g, self.phi)])
        self.phi.reshape((self.F, self.NOC))
        self.X.reshape((self.F, self.NOC))
        self.x, self.y = np.real(self.X), np.imag(self.X)
        # Objective function and linear inequalties for optimization
        self.reset()
        # Initial solution
        if rho0 is None:
            rho0 = np.array([10 ** (-6) if i < len(nopid) else 0 for i in range(self.NOC)])
            # rho0 = np.zeros(self.NOC)
        self.rho = rho0

    def specification(self, o_dgc, theta_dpm, gdb_dgm, phi_dgc=np.pi / 6, theta_dpm2=np.pi / 4, phi_dgm=np.pi / 6):
        """

        :param o_dgc: Desired Gain-Crossover Frequency (rad/s)
        :param theta_dpm: Desired Phase Margin (rad)
        :param gdb_dgm: Desired Gain Margin (dB)
        :param phi_dgc: Constraints parameter for dgc
        :param theta_dpm2: Desired Second Phase Margin (rad)
        :param phi_dgm: Constraints parameter for dgm
        :return:
        """
        g_dgm = 10 ** (gdb_dgm / 20)
        self.o_dgc = o_dgc
        self.phi_dgc = phi_dgc
        self.theta_dpm = theta_dpm
        self.g_dgm = g_dgm
        self.phi_dgm = phi_dgm
        self.theta_dpm2 = theta_dpm2

        # A circle that satisfies Gain Margin, Phase Margin, and Second Phase Margin.
        tm = max(theta_dpm, theta_dpm2)
        xg = -1 / g_dgm
        xp, yp = -np.cos(tm), -np.sin(tm)
        self.sigma = 1 / 2 * (xg ** 2 - 1) / (xg - xp)
        self.rm = xg - self.sigma

    def linecond(self, l, a, b, c=1):
        """
        c * y <= a*x + b for l
        :param l:
        :param a:
        :param b:
        :param c:
        :return:
        """
        if not l:
            return None
        Gl = np.array([c * self.y - a * self.x for i in l])
        Gl.reshape((len(l), self.NOC))
        hl = np.ones((len(Gl), 1)) * b
        self.lcond_append(Gl, hl)

    def gccond(self, nlower=10):
        """
        (1) Gain Crossover Frequency Constraints:
        y <= -tan(self.phi_dgc) * x - 1/cos(self.phi_dgc)
        for 0 << o < self.o_dgc
        :param nlower:
        :return:
        """
        self.l_gc = [i for i, _o in enumerate(self.o) if self.o_dgc / nlower <= _o < self.o_dgc]
        self.linecond(self.l_gc, -np.tan(self.phi_dgc), -1 / np.cos(self.phi_dgc))

    def pmcond(self, nlower=2, nupper=2):
        """
        (2) Phase Margin Constraints:
        y <= -tan(self.phi_dpm) * x
         for o ~ self.o_dgc
        :param nlower:
        :param nupper:
        :return:
        """
        olower = self.o_dgc / nlower
        oupper = self.o_dgc / nupper
        self.l_pm = [i for i, _o in enumerate(self.o) if (self.o_dgc - olower <= _o < self.o_dgc + oupper)]
        self.linecond(self.l_pm, -np.tan(self.phi_dgm), 0)

    def gmcond(self, nupper=1 / 5):
        """
        (3) Gain Margin Constraints:
        y >= a*x + b, where a = cy / (cx + 1/gm), b = a / gm,
        for inf >> o > self.o_dgc
        :param nupper:
        :return:
        """
        oupper = self.o_dgc / nupper
        self.l_gm = [i for i, _o in enumerate(self.o) if oupper > _o >= self.o_dgc]
        cx, cy = - np.cos(self.phi_dgm), -np.sin(self.phi_dgm)
        a = cy / (cx + 1 / self.g_dgm)
        b = a * (1 / self.g_dgm)
        self.linecond(self.l_gm, -a, -b, c=-1)

    def pm2cond(self):
        """
        (4) Second Phase Margin Constraints:
        - cos(self.theta_pm2) < x
        for o >> o_dgc
        :return:
        """
        self.l_pm2 = [i for i, _o in enumerate(self.o) if (max(self.o[j] for j in self.l_gm) <= _o)]
        self.linecond(self.l_pm2, 1, np.cos(self.theta_dpm2), c=0)

    def gainpositivecond(self, glower=0.):
        """
        (5) any(rho[:len(nopid)] > 0) == True, and sum(rho[nofir:]) > 0 == True
        :param glower:
        :param nocon:
        :return:
        """
        nocon = len(self.nopid)
        Gl = np.zeros((nocon, self.NOC))
        for i in range(len(Gl)):
            if i < nocon:
                Gl[i][i] = -1.
        hl = - glower * np.ones((nocon, 1))
        self.lcond_append(Gl, hl)

    def picond(self, ti=50 * 10 ** (-3)):
        """
        (5) Kp - taui * Ki < 0
        :param ti:Integral Time
        :return:
        """
        Gl = np.zeros((1, self.NOC))
        Gl[0][0] = 1
        Gl[0][1] = - ti
        hl = 0 * np.ones((1, 1))
        self.lcond_append(Gl, hl)

    def outofdiskcond(self, r, sigma, l=None):
        """
        (x-sigma)**2 + y**2 >= rm**2
        for all o
        This condition is CONCAVE, so convex solvers cannot deal with it.
        Here, CCCP (Concave-Convex Procedure) is applied to make it convex via Taylor expansion, i.e.
          f(xt) - g(xt) >>> f(xt) - g(xt-1) - dg(xt-1)/dxt-1.T * (xt - xt-1),
          where f is convex and g is concave.
          No GURANTEES for convergence (it may converge to saddle points or local minima),
          yet now it becomes convex constraints.
        :param r:
        :param sigma:
        :param l:
        :return:
        """
        if l is None:
            l = self.l
        L0 = np.dot(self.X, self.rho)
        n = L0 - sigma
        Gl = np.array([- np.real(np.conj(n[i]) * self.X[i]) / abs(n[i]) for i in l])
        Gl.reshape((len(l), self.NOC))
        hl = np.array([-r[i] - np.real(n[i]) / abs(n[i]) * sigma
                       for i in l]).reshape((len(l), 1)) * np.ones((len(l), 1))
        self.lcond_append(Gl, hl)

    def stabilitycond(self, rm=None, sigma=None):
        """
        (6) Stability Constaints:
        (x-sigma)**2 + y**2 >= rm**2
        for all o
        :param rm: radius of stability disk
        :param sigma: center of stability disk
        :return:
        """
        if rm is None:
            rm = self.rm
        if sigma is None:
            sigma = self.sigma
        assert - sigma >= rm
        r = rm * np.ones(len(self.l))
        self.outofdiskcond(r, sigma)

    def nominalcond(self, db=-40):
        """
        (8) Nominal Performance Constraints:
        (x-1)**2 + y**2 >= |W1(s)|**2, where W1(s) = (self.o_dgc/s) ** m
        for all o
        :param db:
        :return:
        """
        m = db / (-20)
        r = [(self.o_dgc / self.o[i]) ** m for i in self.l]
        self.outofdiskcond(r, -1)

    def set_obj(self, c):
        """
        Optimization objective
        :param c:
        :return:
        """
        self.c = c

    def lcond_append(self, Gl, hl):
        """
        Append condtion to Gl and hl
        :param Gl:
        :param hl:
        :return:
        """
        if self.Gl is None:
            self.Gl = Gl
            self.hl = hl
        else:
            self.Gl = np.block([[self.Gl], [Gl]])
            self.hl = np.block([[self.hl], [hl]])

    def robustcond(self, gamma):
        """
        (7) Robust Stability Constratins:
        x**2 + y**2 < gamma**2
        for o >> o_dgc
        Solver must be socp or sdp, that usually takes more time.
        :param gamma:
        :return:
        """
        if not self.l_gm:
            return False
        self.l_rbs = [i for i, _o in enumerate(self.o) if (max(self.l_gm) <= _o)]
        for i in self.l_rbs:
            are, aim = np.real(self.X[i]), np.imag(self.X[i])
            A0 = np.block([[are], [aim]])
            A0.reshape((2, self.NOC))
            b0 = np.zeros((len(A0), 1))
            c0 = np.zeros((self.NOC, 1))
            d0 = gamma
            gq0, hq0 = mycvxopt.qc2socp(A0, b0, c0, d0)
            self.Gql.append(gq0)
            self.hql.append(hq0)

    def econd_append(self, A, b):
        """
        Append condtion to Gl and hl
        :param Gl:
        :param hl:
        :return:
        """
        if self.A is None:
            self.A = A
            self.b = b
        else:
            self.A = np.block([[self.A], [A]])
            self.b = np.block([[self.b], [b]])

    def fircond(self, term=0):
        """
        (5) FIRs = term @ o=0 and o=pi/ts
        EQUIV sum(a[i] for i in even) == term and sum(a[i] for i in odd) == term
        :return:
        """
        # sum( (-1)**i * ai ) == term and sum( ai ) == term
        EVEN = 0
        ODD = 1
        for j in (EVEN, ODD):
            A = np.zeros((1, self.NOC))
            for i in range(self.NOC):
                k = i - len(self.nopid)
                if k >= 0 and (k + j) % 2:
                    A[0][i] = 1
            b = term * np.ones(1)
            self.econd_append(A, b)

    def optimize(self, solver="socp"):
        """
        Available Solvers: Linear Programming (of course including min infinity-norm/1-norm), Quadratic Prgramming,
        Second-Order Cone Prgramming, Semi-Definite Progamming (or LMI)
        :param solver:
        :return:
        """
        self.rho = mycvxopt.solve(solver, [self.c], G=self.Gl, h=self.hl, Gql=self.Gql, hql=self.hql, Gsl=self.Gsl,
                                  hsl=self.hsl, A=self.A, b=self.b, MAX_ITER_SOL=1)
        return self.rho

    def freqresp(self, obj="c"):
        """
        return FRF of C(s), CPID(s), or CFIR(s)
        """
        if obj == "c":
            noc = self.NOC
            phi = np.array(self.phi)
            rho = np.array(self.rho)
        elif obj == self.nopid:
            noc = len(self.nopid)
            phi = np.array([
                               self.phi[i][:noc] for i in range(self.F)
                               ])
            rho = np.array(self.rho[:noc])
        elif obj == "fir":
            noc = self.nofir
            phi = np.array([
                               self.phi[i][-self.nofir:] for i in range(self.F)
                               ])
            rho = np.array(self.rho[-self.nofir:])

        phi.reshape((self.F, noc))
        rho.reshape((noc, 1))
        ret = np.dot(phi, rho)
        assert ret.shape == (self.F,)
        return ret

    def reset(self):
        """
        Reset linear Inequalities Condtions
        :return:
        """
        self.c = np.zeros((self.NOC, 1))  # default FIND
        self.c.reshape((self.NOC, 1))
        self.Gl, self.hl = None, None
        self.Gql, self.hql = [], []
        self.Gsl, self.hsl = [], []
        self.A, self.b = None, None

    def split(self, n):
        """
        split data to n parts (into each FRF)
        :param n:
        :return:
        """
        self.olist = mynum.nsplit(self.o, n)
        Xlist = mynum.nsplit(self.X, n)
        self.Llist = [np.dot(X, self.rho) for X in Xlist]

    def calc_gcf(self):
        """
        calculate acutual gain crossover frequency
        :return:
        """
        gain_crossover_o = []
        for o, l in zip(self.olist, self.Llist):
            for _o, _l in zip(o, l):
                if not check_disk((_l,), 1, 0):
                    gain_crossover_o.append(temp)
                    break
                temp = _o
        return np.array(gain_crossover_o) / 2 / np.pi


class Simulation():
    def __init__(self, s, plant, controller):
        l = plant * controller
        t = l / (1 + l)
        self.T = mysignal.symbolic_to_tf(t, s)
        self.S = 1 - self.T

    def step(self):
        t, yout = signal.step2(self.T)
        return t, yout


def pid(nopid, taud, ts=0.0):
    """
    return [P(o), I(o), D(o)]
    :param nopid: "p" if P, "pi" if pi, "pid" if pid, "pd" if pd
    :param taud: period of pseudo differential
    :return:
    """

    def p(o):
        return 1

    def i(o):
        s = 1.j * o
        if ts == 0:
            return 1 / s
        else:
            zi = np.exp(- ts * s)
            return -(ts * (zi + 1)) / (2 * (zi - 1))

    def d(o):
        s = 1.j * o
        if ts == 0:
            return s / (1 + taud * s)
        else:
            zi = np.exp(-ts * s)
            return -(2 * (zi - 1)) / (2 * taud + ts - 2 * taud * zi + ts * zi)

    pid_controller = {"p": p, "i": i, "d": d}
    return [pid_controller[t] for t in nopid.lower()]


def fir(nofir, ts, is_notch=False):
    """
    return [z^-1 (o), z^-2 (o), ..., z^-nofir (o)] if not is_notch (PARALLEL TYPE)
    else return [CFIR0(z), ..., CFIRnofr(z) ], where CFIRi(z) = (1+z**(-i))/2
    where z = exp(ts*s)
    :param nofir: number of FIR filters
    :param ts:
    :return:
    """

    if not is_notch:
        def nfir(n):
            def zinv(o):
                s = 1.j * o
                zi = np.exp(- ts * s)
                return zi ** (n)

            return zinv

        return [nfir(i) for i in range(1, 1 + nofir)]
    else:
        def nfirnotch(n):
            def firnotch(o):
                s = 1.j * o
                zi = np.exp(- ts * s)
                return (zi ** (n) + 1) / 2

            return firnotch

        return [nfirnotch(i) for i in range(nofir)]


def check_disk(l, r, sigma):
    """
    return if all of L(s) is out of the disk (radius:r, center:(sigma, 0))
    :param l:
    :param r:
    :param sigma:
    :return:
    """

    x, y = np.real(l), np.imag(l)
    return all((xp - sigma) ** 2 + yp ** 2 >= r ** 2 for xp, yp in zip(x, y))
