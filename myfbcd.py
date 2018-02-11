"""

"""
import mycvxopt
import mynum
import numpy as np


class SuperControllerDesin(object):

    def __init__(self, o, g, ts):
        """
        Super initialization.
        :param o: Frequency Lines (rad/s)
        :param g: Gain Lines
        :param ts: Samping period (s)
        """
        self.o = o
        self.g = g.reshape((len(g), 1))
        self.ts = ts
        self.l = [i for i in range(len(o))]
        self.F = len(o)
        self.reset()  # Objective function and linear inequalties for optimization

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
        self.sigma = self.sigma * np.ones((self.F, 1))
        self.rm = xg - self.sigma

    def gainpositivecond(self, glower=0):
        """
        All of parameters must be > 0
        :param glower:
        :param nocon:
        :return:
        """
        Gl = -1 * np.eye(self.NOP)
        hl = - glower * np.ones((len(Gl), 1))
        self.lcond_append(Gl, hl)

    def outofdiskcond(self, rm, sigmam, l=None):
        """
        (x-sigma)**2 + y**2 >= rm**2 for all o
        Overwrite this method.
        """
        pass

    def nominalcond(self, db=-40, l=None):
        """
        Nominal Performance Constraints:
        (x-1)**2 + y**2 >= |W1(s)|**2, where W1(s) = (self.o_dgc/s) ** m
        for all o
        :param db:
        :return:
        """
        self.nominal_sensitivity = db
        m = db / (-20)
        if l is None:
            l = self.l
        r = np.array([(self.o_dgc / self.o[i]) ** m for i in self.l]).reshape((self.F, 1))
        self.outofdiskcond(r, -1 * np.ones((self.F, 1)), l)

    def stabilitycond(self, rm=None, sigma=None):
        """
        Stability Constaints:
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
        self.outofdiskcond(rm, sigma)

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

    def reset(self):
        """
        Reset linear Inequalities Condtions
        :param nop:
        :return:
        """
        self.c = np.zeros((self.NOP, 1))  # default FIND
        self.c.reshape((self.NOP, 1))
        self.Gl, self.hl = None, None
        self.Gql, self.hql = [], []
        self.Gsl, self.hsl = [], []
        self.A, self.b = None, None

    def controller(self):
        """
        Calculate controller self.C and openloop FRF self.L
        Overwrite this method.
        """
        self.C = None
        self.L = None

    def sensitivity(self):
        """
        Calculate sensitivity S and complementary sensitivity T
        :return:
        """
        self.S = 1 / (1 + self.L)
        self.T = 1 - self.S

    def split(self, n):
        """
        split data to n parts (into each FRF)
        :param n:
        :return:
        """
        self.olist = mynum.nsplit(self.o, n)
        self.controller()
        self.Clist = mynum.nsplit(self.C, n)
        self.Llist = mynum.nsplit(self.L, n)

    def check_stability(self, rm=None, sigma=None):
        """
        check if stability condition is satisfied
        :param rm:
        :param sigma:
        :return:
        """
        if rm is None:
            rm = self.rm
        if sigma is None:
            sigma = self.sigma
        return check_disk(self.L, rm, sigma)

    def check_nominal(self, db, l=None):
        """
        check if nominal sensitivity condition is satisfied
        :param db:
        :param l:
        :return:
        """
        if l is None:
            l = self.l
        L = [self.L[i] for i in l]
        F = len(l)
        m = db / (-20)
        r = np.array([(self.o_dgc / self.o[i]) ** m for i in l]).reshape((F, 1))
        return check_disk(L, r, -1 * np.ones((F, 1)))

    def calc_gcf(self, mode="g"):
        """
        calculate X-crossover frequency
        :param mode: if "g" then X:gain, elif "s" then X:sensitivity
        :return:
        """
        gain_crossover_o = []
        for o, l in zip(self.olist, self.Llist):
            for _o, _l in zip(o, l):
                if mode == "g":
                    x = _l  # open loop
                elif mode == "s":
                    x = _l + 1  # sensitivity
                if not check_disk((x,), np.ones(1), np.zeros(1)):
                    gain_crossover_o.append(temp)
                    break
                temp = _o
        return np.array(gain_crossover_o) / 2 / np.pi


class ControllerDesign(SuperControllerDesin):
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

    def __init__(self, o, g, nopid="pid", taud=0.01, nofir=10, is_notch=False, notch_offset=0, ts=0.001, tsfir=0.001,
                 rho0=None):
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
        self.nopid = nopid
        self.nofir = nofir
        self.NOP = len(nopid) + nofir

        super().__init__(o, g, ts)

        self.phi = np.array([
            np.array([*[c(_o) for c in pid(nopid, taud=taud, ts=ts)],
                      *[c(_o) for c in
                        fir(nofir=nofir, ts=tsfir, is_notch=is_notch, noffset=notch_offset)]])
            for _o in o])
        self.X = np.array([_g * _phi for _g, _phi in zip(g, self.phi)])
        self.phi.reshape((self.F, self.NOP))
        self.X.reshape((self.F, self.NOP))
        self.x, self.y = np.real(self.X), np.imag(self.X)
        # Initial solution
        if rho0 is None:
            if not is_notch:
                rho0 = np.array([10 ** (-6) if i < len(nopid) else 0 for i in range(self.NOP)])
            else:
                rho0 = np.array([1 if i == 0 else 0 for i in range(self.NOP)])
        self.rho = rho0

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
        Gl.reshape((len(l), self.NOP))
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

    def gainpositivecond(self, glower=0., is_fir=False):
        """
        (5) any(rho[:len(nopid)] > 0) == True, or any(rho[-nofir:]) > 0 == True if is_notch
        :param glower:
        :param nocon:
        :return:
        """
        if not is_fir:
            noconl = 0
            noconu = len(self.nopid)
        else:
            noconl = len(self.nopid)
            noconu = self.NOP
        Gl = np.zeros((noconu - noconl, self.NOP))
        for i in range(len(Gl)):
            Gl[i][i + noconl] = -1.
        hl = - glower * np.ones((len(Gl), 1))
        self.lcond_append(Gl, hl)

    def picond(self, ti=50 * 10 ** (-3)):
        """
        (5) Kp - taui * Ki < 0
        :param ti:Integral Time
        :return:
        """
        Gl = np.zeros((1, self.NOP))
        Gl[0][0] = 1
        Gl[0][1] = - ti
        hl = 0 * np.ones((1, 1))
        self.lcond_append(Gl, hl)

    def pdcond(self, td=50 * 10 ** (-3)):
        """
        (5) Kd - taud * Kp < 0
        :param td:Differenciation Time
        :return:
        """
        Gl = np.zeros((1, self.NOP))
        Gl[0][2] = 1
        Gl[0][0] = - td
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
        F = len(l)
        L0 = np.dot(self.X, self.rho).reshape((self.F, 1))
        n = L0 - sigma
        Gl = np.array([- np.real(np.conj(n[i]) * self.X[i]) / abs(n[i]) for i in l])
        Gl.reshape((F, self.NOP))
        hl = np.array([-r[i] - np.real(n[i]) / abs(n[i]) * sigma[i] for i in l]).reshape((F, 1))
        self.lcond_append(Gl, hl)

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
            A0.reshape((2, self.NOP))
            b0 = np.zeros((len(A0), 1))
            c0 = np.zeros((self.NOP, 1))
            d0 = gamma
            gq0, hq0 = mycvxopt.qc2socp(A0, b0, c0, d0)
            self.Gql.append(gq0)
            self.hql.append(hq0)

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
            A = np.zeros((1, self.NOP))
            for i in range(self.NOP):
                k = i - len(self.nopid)
                if k >= 0 and (k + j) % 2:
                    A[0][i] = 1
            b = term * np.ones(1)
            self.econd_append(A, b)

    def firnotchcond(self, term=1):
        """
        (5) FIR Notch constraint:
        CFIR(1)=1
        as notch filter wokrs
        :param term:
        :return:
        """
        A = np.zeros((1, self.NOP))
        for i in range(self.NOP):
            k = i - len(self.nopid)
            if k >= 0:
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

    def controller(self):
        """
        Calculate controller and openloop frequency response
        :return:
        """
        self.C = np.dot(self.phi, self.rho)
        self.L = np.dot(self.X, self.rho)

    def freqresp(self, obj="c"):
        """
        return FRF of C(s), CPID(s), or CFIR(s)
        """
        if obj == "c":
            noc = self.NOP
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

    def split(self, n):
        """
        split data to n parts (into each FRF)
        :param n:
        :return:
        """
        self.olist = mynum.nsplit(self.o, n)
        Xlist = mynum.nsplit(self.X, n)
        self.Llist = [np.dot(X, self.rho) for X in Xlist]


class IIRControllerDesign(SuperControllerDesin):

    def __init__(self, o, g, ts=50 * 10 ** (-6), rhon=[], rhod=[], blist=[], dlist=[]):
        """
        k = 0, 1, ..., F-1
        Controller: Ck = ( a0k.T * rhon + b0k ) / ( c0k.T * rhod + d0k )
        Open Loop: Lk = gk * Ck
        Parameter vector rho = [rhom; rhod]
        :param o: Frequency ok
        :param g: Plant FRF gk
        :param ts: Sampling period
        :param rhon: numerator parameter
        :param rhod: denominator parameter
        """
        if rhod == []:
            rhod = np.array([5 * ts, 1])
        if rhon == []:
            rhon = np.array(pidtaud2b(0.2, 200, 10 ** (-4), rhod[0]))
        self.nonum = len(rhon)
        self.noden = len(rhod)
        self.NOP = self.nonum + self.noden

        super().__init__(o, g, ts)

        self.rhon = rhon
        self.rhod = rhod
        self.rho = np.block([self.rhon, self.rhod])

        self.a0T, self.b0, self.c0T, self.d0 = self.basis(self.o, self.ts, blist, dlist)  # set basis

    def basis(self, o, ts, blist=[], dlist=[]):
        """
        return basis a0.T, b0, c0T, d0 of controller
        :param o:
        :param ts:
        :param blist:
        :param dlist:
        :return:
        """
        a0T, b0, c0T, d0 = np.array([]), np.array([]), np.array([]), np.array([])
        for k, ok in enumerate(o):
            s = 1.j * ok
            zi = np.exp(- ts * s)
            # Inverse bilinear transform is used to avoid ill condition @ z=-1
            sinv = (ts * (1 + zi)) / (2 * (1 - zi))
            aT = np.array([sinv ** i for i in range(self.nonum)])
            if not blist:
                b = 0
            else:
                b = blist[k]
            cT = np.array([sinv ** i for i in range(self.noden)])
            if not dlist:
                d = 0
            else:
                d = dlist[k]
            a0T = np.append(a0T, aT)
            b0 = np.append(b0, b)
            c0T = np.append(c0T, cT)
            d0 = np.append(d0, d)
        return a0T.reshape((len(o), self.nonum)), b0, c0T.reshape((len(o), self.noden)), d0

    def outofdiskcond(self, rm, sigma, l=None):
        """
        Add SOCP constarints:
        rk - |Lk - simgak| <=0 <--->
        ||Ak rho - Bk||2 <= Ck.T rho + Dk for k in self.l
        :param rm:
        :param sigma:
        :return:
        """
        if l is None:
            l = self.l
        for i in l:
            E = np.block([self.g[i] * self.a0T[i], -sigma[i] * self.c0T[i]])
            F = self.g[i] * self.b0[i] - sigma[i] * self.d0[i]
            G = np.block([np.zeros(self.nonum), self.c0T[i]])
            H = self.d0[i]
            E.reshape((1, self.NOP))
            G.reshape((1, self.NOP))
            n0 = np.dot(E, self.rho) + F
            I = np.real(np.conj(n0) * E) / abs(n0)
            J = np.real(np.conj(n0) * F) / abs(n0)
            A = np.block([[np.real(G)], [np.imag(G)]])
            B = np.block([[np.real(H)], [np.imag(H)]])
            C = I / rm[i]
            D = J / rm[i]
            gq0, hq0 = mycvxopt.qc2socp(A, B, C, D)
            self.Gql.append(gq0)
            self.hql.append(hq0)

    def gainpositivecond(self, glower=np.sqrt(np.finfo(float).eps)):
        """
        (5) any(rho[:len(nopid)] > 0) == True, or any(rho[-nofir:]) > 0 == True if is_notch
        :param glower:
        :param nocon:
        :return:
        """
        super().gainpositivecond(glower=glower)

    def optimize(self, solver="socp"):
        """
        Available Solvers: Linear Programming (of course including min infinity-norm/1-norm), Quadratic Prgramming,
        Second-Order Cone Prgramming, Semi-Definite Progamming (or LMI)
        :param solver:
        :return:
        """
        self.rho = mycvxopt.solve(solver, [self.c], G=self.Gl, h=self.hl, Gql=self.Gql, hql=self.hql, Gsl=self.Gsl,
                                  hsl=self.hsl, A=self.A, b=self.b, MAX_ITER_SOL=1)
        self.rhon = self.rho[:self.nonum]
        self.rhod = self.rho[self.nonum:]
        return self.rho

    def controller(self):
        """
        calculate controller response
        :return:
        """
        self.C = np.array([
            (np.dot(self.a0T[i], self.rhon) + self.b0[i]) / (np.dot(self.c0T[i], self.rhod) + self.d0[i])
            for i in self.l])
        self.L = self.g.reshape((self.F, 1)) * self.C.reshape((self.F, 1))


class LFControllerDesign(SuperControllerDesin):

    def __init__(self, o, g, ts, rho, ctype=["pid", 3, 1]):
        """
        Linear Fractional Controller Design:
        Ck = prod( (aiTk*xi + bik)/(ciTk*xi+dik) for i in (0, 1, 2 ...))
        :param o:
        :param g:
        :param ts:
        :param rho:
        :param ctype: [pid, number of notch filters, number of phase compensator filters]
        """

        self.nopid = (3, 2)  # num, den
        self.nonotch = ctype[1]
        self.nopc = ctype[2]
        B2, B1, B0, A2, A1 = 0, 1, 2, 3, 4
        LENPID = 5
        D1, D2, C1 = 5, 6, 7
        # PIDtaud
        self.numidx = [B2, B1, B0]
        self.denidx = [A2, A1]
        self.sepidx = [0, LENPID]
        # Notch
        for i in range(self.nonotch):
            self.numidx.append(D1 + 3 * i)
            self.denidx.append(D2 + 3 * i)
            self.denidx.append(C1 + 3 * i)
            self.sepidx.append(self.sepidx[-1] + 3)
        # Phase Compensator
        for i in range(self.nopc):
            F1 = self.sepidx[-1]
            E1 = F1 + 1
            self.numidx.append(F1 + 2 * i)
            self.denidx.append(E1 + 2 * i)
            self.sepidx.append(self.sepidx[-1] + 2)

        self.NOP = len(self.numidx) + len(self.denidx)
        super().__init__(o, g, ts)
        self.g = self.g.reshape((len(g), 1))

        self.set_rho0(rho)

        self.sinv = calc_sinv(o, ts)

        self.NOC = int(ctype[0] != "") + sum(int(x > 0) for x in ctype[1:])
        # for reusing phi
        self.phia = []
        self.phic = []
        for i in range(self.NOC):
            self.phia.append(None)
            self.phic.append(None)

    def set_rho0(self, rho):
        self.rho = rho  # [rho[i] for i in range(NOSC)]
        self.NOSC = len(rho)  # number of series controllers
        self.rho0 = np.array([item for sublist in rho for item in sublist]).reshape((self.NOP, 1))

    def pidbasis(self, i):
        """
        basis of pids:
        Cpid = N*rhopid/D*rhopid, where
        rhopid = [b2, b1, b0, a1, a0]
        numerator N: [1, sinv, sinv**2, 0, 0]
        denominator D: [0, 0, 0, 1, sinv]
        :return:
        """
        if self.phia[i] is None:
            self.phia[i] = phi(self.l, self.sinv, self.nopid[0])
            self.phic[i] = phi(self.l, self.sinv, self.nopid[1])
        phia = self.phia[i]
        phic = self.phic[i]
        zeroa = np.zeros(self.nopid[1])
        aT = np.array([np.array([*phia[k], *zeroa]) for k in self.l])
        zeroc = np.zeros(self.nopid[0])
        cT = np.array([np.array([*zeroc, *phic[k]]) for k in self.l])
        b = np.zeros(self.F)
        d = np.zeros(self.F)
        return aT, b, cT, d

    def notchbasis(self, i):
        """
        basis of notch filter:
        Cnotch = (N * rhonotch + 1) / (D*notch + 1), where
        rhonotch = [d1, d2, c1]
        N: [sinv, siinv**2, 0]
        D: [0, sinv**2, sinv]
        :return:
        """
        if self.phia[i] is None:
            self.phia[i] = phi(self.l, self.sinv, 3, ioffset=1)  # [sinv, sinv**2]
        phia = self.phia[i]
        zeroa = np.zeros(1)
        aT = np.array([np.array([*phia[k], *zeroa]) for k in self.l])
        cT = np.array([aT[k][::-1] for k in self.l])
        b = np.ones(self.F)
        d = np.ones(self.F)
        return aT, b, cT, d

    def pcbasis(self, i):
        """
        Basis of phase compensator filter
        Cp = (N * rhocp + 1) / (D * rhon + 1)
        rhonotch = [f1, e1]
        N: [sinv, 0]
        D: [0, sinv]
        :return:
        """
        if self.phia[i] is None:
            self.phia[i] = phi(self.l, self.sinv, 2, ioffset=1)  # [sinv]
        phia = self.phia[i]
        zeroa = np.zeros(1)
        aT = np.array([np.array([*phia[k], *zeroa]) for k in self.l])
        cT = np.array([aT[k][::-1] for k in self.l])
        b = np.ones(self.F)
        d = np.ones(self.F)
        return aT, b, cT, d

    def setbasis(self):
        """
        set each basis function
        :return:
        """
        # PID
        self.aT = []
        self.b = []
        self.cT = []
        self.d = []

        i = 0

        aT, b, cT, d = self.pidbasis(i)
        if self.nopid:
            self.aT.append(aT)
            self.b.append(b)
            self.cT.append(cT)
            self.d.append(d)
        if self.nonotch:
            i += 1
            aT, b, cT, d = self.notchbasis(i)
            for n in range(self.nonotch):
                self.aT.append(aT)
                self.b.append(b)
                self.cT.append(cT)
                self.d.append(d)
        if self.nopc:
            i += 1
            aT, b, cT, d = self.pcbasis(i)
            for n in range(self.nopc):
                self.aT.append(aT)
                self.b.append(b)
                self.cT.append(cT)
                self.d.append(d)

    def controller(self):
        """
        set controller
        :return:
        """

        def frac_numden(rho, aT, b):
            return [np.array([affine(rho[i], aT[i][k], b[i][k])
                              for k in self.l]) for i in range(self.NOSC)]

        self.setbasis()
        self.alpha = frac_numden(self.rho, self.aT, self.b)
        self.gamma = frac_numden(self.rho, self.cT, self.d)
        self.C = lf_controller(self.l, self.NOSC, self.alpha, self.gamma)
        self.L = self.g * self.C

    def gaincond(self, denlower=0.8, denupper=1.2):
        """
        denominator parameters should not be largely changed because of linearization
        :return:
        """
        # parameter change limitted within -20 % ~ 20 %
        Gl = np.zeros((len(self.denidx), self.NOP))
        for j, i in enumerate(self.denidx):
            Gl[j][i] = -1. / self.rho0[i]
        hl = - denlower * np.ones((len(self.denidx), 1))
        self.lcond_append(Gl, hl)
        Gl = np.zeros((len(self.denidx), self.NOP))
        for j, i in enumerate(self.denidx):
            Gl[j][i] = 1. / self.rho0[i]
        hl = denupper * np.ones((len(self.denidx), 1))
        self.lcond_append(Gl, hl)

        if self.nopc == 1:
            Gl = np.zeros((1, self.NOP))
            Gl[0][14] = -1.
            Gl[0][15] = 1.
            hl = np.zeros(1)
            self.lcond_append(Gl, hl)

        if self.nonotch == 3:
            A = np.zeros((1, self.NOP))
            A[0][6] = 1  # const freq0
            b = 2.058536168054191e+09 * np.ones(1)
            self.econd_append(A, b)

            A = np.zeros((1, self.NOP))
            A[0][6 + 3] = 1  # const freq
            b = 3.239231067335693e+09 * np.ones(1)
            self.econd_append(A, b)

            A = np.zeros((1, self.NOP))
            A[0][6 + 6] = 1  # const freq
            b = 8.170691887679803e+08 * np.ones(1)
            self.econd_append(A, b)

    def outofdiskcond(self, rm, sigmam, l=None):
        """
        out of disk
        :param rm:
        :param sigmam:
        :param l:
        :return:
        """
        if l is None:
            l = self.l
        F = len(l)
        L = np.array([self.L[i] for i in l]).reshape((F, 1))
        A = np.ones((F, self.NOP), dtype=np.complex_)
        for j, k in enumerate(l):
            a = [self.aT[i][k] / self.alpha[i][k] - self.cT[i][k] / self.gamma[i][k]
                 for i in range(self.NOSC)]
            A[j] = np.array([item for sublist in a for item in sublist])
        A *= L
        R = L - sigmam
        Gl = - np.real(np.conj(R) * A) / abs(R)
        hl = -rm + abs(R) + np.dot(Gl, self.rho0)
        self.lcond_append(Gl, hl)

    def optimize(self, solver="lp"):
        """
        Available Solvers: Linear Programming (of course including min infinity-norm/1-norm), Quadratic Prgramming,
        Second-Order Cone Prgramming, Semi-Definite Progamming (or LMI)
        :param solver:
        :return:
        """
        rhovec = mycvxopt.solve(solver, [self.c], G=self.Gl, h=self.hl, Gql=self.Gql, hql=self.hql, Gsl=self.Gsl,
                                hsl=self.hsl, A=self.A, b=self.b, MAX_ITER_SOL=1)
        self.rho0 = rhovec.reshape((self.NOP, 1))
        self.rho = [rhovec[self.sepidx[i]:self.sepidx[i + 1]] for i in range(len(self.sepidx) - 1)]
        return self.rho


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


def fir(nofir, ts, is_notch=False, noffset=0):
    """
    return [z^-1 (o), z^-2 (o), ..., z^-nofir (o)] if not is_notch (PARALLEL TYPE)
    else return [CFIR0(z), ..., CFIRnofir-1(z) ], where CFIRi(z) = (1+z**(-i))/2
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
                return zi ** n

            return zinv

        return [nfir(i) for i in range(1 + noffset, 1 + noffset + nofir)]
    else:
        def nfirnotch(n):
            def firnotch(o):
                s = 1.j * o
                zi = np.exp(- ts * s)
                return (zi ** n + 1) / 2

            return firnotch

        return [nfirnotch(i) for i in (0, *range(noffset, noffset + nofir - 1))]


def check_disk(L, r, sigma):
    """
    return if all of L(s) is out of the disk (radius:r, center:(sigma, 0))
    :param l:
    :param r:
    :param sigma:
    :return:
    """
    if len(r) == 1:
        r = r * np.ones(len(L))
        simga = sigma * np.ones(len(L))
    x, y = np.real(L), np.imag(L)
    return all((x[i] - sigma[i]) ** 2 + y[i] ** 2 >= r[i] ** 2 for i in range(len(L)))


def nominalcond(db=-40, l=None):
    """
    Nominal Performance Constraints:
    (x-1)**2 + y**2 >= |W1(s)|**2, where W1(s) = (self.o_dgc/s) ** m
    for all o
    :param db:
    :return:
    """
    self.nominal_sensitivity = db
    m = db / (-20)
    r = np.array([(self.o_dgc / self.o[i]) ** m for i in l])
    self.outofdiskcond(r.reshape((l, 1)), -1 * np.ones((l, 1)), l)


def pidtaud2b(kp, ki, kd, taud):
    """
    PIDtauD to b vector
    :param kp:
    :param ki:
    :param kd:
    :param taud:
    :return:
    """
    b0 = ki
    b1 = kp + ki * taud
    b2 = kd + kp * taud
    return [b2, b1, b0]


def btaud2pid(b, taud):
    """
    b vector to PIDtauD
    :param b:
    :param taud:
    :return:
    """
    ki = b[2]
    kp = b[1] - ki * taud
    kd = b[0] - kp * taud
    return kp, ki, kd


def calc_sinv(o, ts):
    """
    return [sinv[0], sinv[1], ..., sinv[F]] for o[0], ..., o[F}
    :param o:
    :param ts:
    :return:
    """
    sinv = np.zeros(len(o), dtype=complex)
    for k, ok in enumerate(o):
        s = 1.j * ok
        zi = np.exp(- ts * s)
        # Inverse bilinear transform is used to avoid ill condition @ z=-1
        sinv[k] = (ts * (1 + zi)) / (2 * (1 - zi))
    return sinv


def phi(l, sinv, n, ioffset=0):
    """
    return [sinv**i for i=(ioffset, ..., n-1)] for all k
    :param l:
    :param sinv:
    :param n:
    :param ioffset:
    :return:
    """
    return np.array([np.array([sinv[k] ** i for i in range(ioffset, n)]) for k in l])


def affine(x, aT, b):
    """
    return aT * x + b
    :param x:
    :param aT:
    :param b:
    :return:
    """
    return np.dot(aT, x) + b


def lf_controller(l, nosc, alpha, gamma):
    """
    Linear Fractional Controller:
    C[k] = prod(alpha[k][i]/gamma[k][i] for i in range(nosc))
    :param l: frequency lines
    :param nosc: number of series controllers
    :param alpha: k-th frequency response of numerator of i-th controller
    :param gamma: k-th frequency response of denominator of i-th controller
    :return: np.array(C[k] for k in l])
    """
    C = np.array([np.prod([alpha[i][k] / gamma[i][k] for i in range(nosc)]) for k in l])
    return C.reshape((len(l), 1))
