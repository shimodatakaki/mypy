"""

"""
import mycvxopt
import numpy as np


class ControllerDesign():
    """
    Find a linear FB controller that satisfies desired (given) linear constraints for FB,
    i.e. (1) Gain-Crossover Linear Inequalities (self.o_dgc, self.phi_dgc),
         (2) Phase Margin Linear Inequalities (self.theta_dpm),
         (3) Gain Margin Linear Inequalities (self.g_dgm, self.phi_dgm),
         (4) Second Phase Margin Linear Inequalities (self.theta_dpm2),
         (5) Gain Minimum/Maximum Linear Inequalities,
         (6) Other Linear Equalities/Inequalities.
    Default Controller: PIDs + 10 FIRs (13 variables)
    Default Optimization: solver=="lp", min c.T * rho, s.t. Gl*rho <= hl, A*rho = b
    """

    def __init__(self, o, g, nopid="pid", taud=0.001, nofir=10, ts=0.001):
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
        self.F = len(o)
        self.nopid = nopid
        self.nofir = nofir
        self.NOC = len(nopid) + nofir
        self.phi = [np.array([*[c(_o) for c in pid(nopid, taud=taud)], *[c(_o) for c in fir(nofir=nofir, ts=ts)]])
               for _o in o]
        self.X = np.array([_g * _phi for _g, _phi in zip(g, self.phi)])
        self.X.reshape((self.F, self.NOC))
        # Objective function and linear inequalties for optimization
        self.c = np.zeros((self.NOC, 1))  # default FIND
        self.c.reshape((self.NOC, 1))
        self.Gl = None
        self.hl = None

    def specification(self, o_dgc, theta_dpm, gdb_dgm, phi_dgc=np.pi / 3, theta_dpm2=np.pi / 4, phi_dgm=np.pi / 6):
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

    def gccond(self, nlower=10):
        """
        (1) Gain Crossover Frequency Constraints:
        y <= -tan(self.phi_dgc) * x - 1/cos(self.phi_dgc)
        for 0 << o < self.o_dgc
        :param nlower:
        :return:
        """
        self.l_gc = [i for i, _o in enumerate(self.o) if self.o_dgc / nlower <= _o < self.o_dgc]
        if not self.l_gc:
            return False
        Gl = np.array(
            [np.imag(self.X[i]) * np.cos(self.phi_dgc) + np.real(self.X[i]) * np.sin(self.phi_dgc)
             for i in self.l_gc])
        Gl.reshape((len(self.l_gc), self.NOC))
        hl = np.ones((len(Gl), 1)) * (-1.)
        self.lcond_append(Gl, hl)

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
        if not self.l_pm:
            return False
        Gl = np.array([np.imag(self.X[i]) - np.tan(self.theta_dpm) * np.real(self.X[i]) for i in self.l_pm])
        Gl.reshape((len(self.l_pm), self.NOC))
        hl = np.zeros((len(Gl), 1))
        self.lcond_append(Gl, hl)

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
        if not self.l_gm:
            return False
        cx, cy = - np.cos(self.phi_dgm), -np.sin(self.phi_dgm)
        a = cy / (cx + 1 / self.g_dgm)
        b = a * (1 / self.g_dgm)
        Gl = np.array([np.imag(self.X[i]) - np.real(self.X[i]) * a for i in self.l_gm])
        Gl.reshape((len(self.l_gm), self.NOC))
        hl = np.ones((len(Gl), 1)) * b
        self.lcond_append(Gl, hl)

    def pm2cond(self):
        """
        (4) Second Phase Margin Constraints:
        - cos(self.theta_pm2) < x
        for o >> o_dgc
        :return:
        """
        if not self.l_gm:
            return False
        self.l_pm2 = [i for i, _o in enumerate(self.o) if (max(self.l_gm) < _o)]
        Gl = np.array([- np.real(self.X[i]) for i in self.l_pm2])
        Gl.reshape((len(self.l_pm2), self.NOC))
        hl = np.ones((len(Gl), 1)) * np.cos(self.theta_dpm2)
        self.lcond_append(Gl, hl)

    def gaincond(self, glower=0., nocon=3):
        """
        (5) Kp, Ki, Kd > 0
        :param glower:
        :param nocon:
        :return:
        """
        Gl = np.zeros((nocon, self.NOC))
        for i in range(nocon):
            Gl[i][i] = -1.
        hl = - glower * np.ones((nocon, 1))
        self.lcond_append(Gl, hl)

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

    def optimize(self, solver="lp"):
        """
        Available Solvers: Linear Programming (of course including min infinity-norm/1-norm), Quadratic Prgramming,
        Second-Order Conce Prgramming, Semi-Definite Progamming (or LMI)
        :param solver:
        :return:
        """
        self.rho = mycvxopt.solve(solver, [self.c], G=self.Gl, h=self.hl, MAX_ITER_SOL=1)
        return self.rho

    def freqresp(self):
        phi = np.array(self.phi)
        phi.reshape((self.F, self.NOC))
        self.rho.reshape((self.NOC, 1))
        ret = np.dot(self.phi, self.rho)
        assert ret.shape == (self.F, )
        return ret

def pid(nopid, taud):
    """
    return [P(o), I(o), D(o)]
    :param nopid: "p" if P, "pi" if pi, "pid" if pid, "pd" if pd
    :param taud: period of pseudo differential
    :return:
    """

    def p(o):
        return 1

    def i(o):
        return 1 / (1.j * o)

    def d(o):
        return 1.j * o / (1 + taud * 1.j * o)

    pid_controller = {"p": p, "i": i, "d": d}
    return [pid_controller[t] for t in nopid.lower()]


def fir(nofir, ts):
    """
    return [z^-1 (o), z^-2 (o), ..., z^-nofir (o)]
    where z = exp(ts*s)
    :param nofir: number of FIR filters
    :param ts:
    :return:
    """

    def nfir(n):
        def zinv(o):
            return np.exp(- ts * 1.j * o * n)

        return zinv

    return [nfir(i) for i in range(1, 1 + nofir)]