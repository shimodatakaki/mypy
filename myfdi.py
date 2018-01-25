from scipy import signal
import numpy as np
from scipy.fftpack import fft
import os
import mycsv
import myplot
import myoptimization as myopt
import mysignal

INPUT_CSV = "t_u.csv"
OUTPUT_CSV = "t_u_y.csv"
LINES_CSV = "lines.csv"
DATA_CSV = "data.csv"
INPUT_H = "input.h"
DATA_PATH = "./data/"


class Excitation():
    """
    # Multisine Excitation
    :method multisine: make multisine, given amplitude and phase
    :method optimize: optimize phase of multisine by non-linear least square optimization using my optimization toolbox
    """

    def __init__(self, path=DATA_PATH, fs=4000, f_min=10, f_max=500, df=1, r=1.05):
        """
        path: parent directory
        fs: sampling frequency
        m: number of iterations
        trs: transient period
        f_min: min frequency
        f_max: max frequency
        df: initial frequency difference
        r: ratio of quasi-logarithmic
        :return None
        """
        self.path = path
        self.fs = fs
        self.df = df
        self.r = r
        t0 = 1 / df  # excitation period
        dt = 1 / fs  # sampling period
        n = t0 * fs  # number of points
        self.t = np.linspace(0, n - 1, n) * dt  # time series
        ###quasi-log frequency lines###
        f = f_min
        l_lines = [f // df]  # f_lines = l_lines * df
        while f < f_max:
            f *= r
            if f // df > l_lines[-1]:
                l_lines.append(f // df)
        l_lines = l_lines[:-1]  # remove f>f_max
        self.l_lines = l_lines
        self.f_lines = np.array(l_lines) * df  # frequency line
        self.o_lines = self.f_lines * 2 * np.pi  # rad/s line
        ###initial multisine###
        self.a = np.ones(len(self.f_lines))  # amplitude
        self.phi = np.random.rand(len(self.f_lines)) * 2 * np.pi  # phase

    def amplitude(self, system):
        """
        Modify Amplitude Spectrum
        :param system:Transfer function of desired spectrum, usually 1 or 1/LPF(s)
        :return:
        """
        w, h = signal.freqresp(system, w=self.o_lines)
        self.a *= abs(h)

    def multisine(self):
        """
        Make multisine
        :return None
        """
        self.u = sum(a * np.cos(2 * np.pi * f * self.t + phi) for f, phi, a in zip(self.f_lines, self.phi, self.a))
        self.u /= max(self.u)
        self.crest = 1 / np.sqrt(np.mean(self.u ** 2))  # crest factor

    def optimize(self, PP_MAX=8, MIN_CREST=1.4):
        """
        Optimization of multisine by INFINITY NORM ALGORITHM
        :param PP_MAX:
        :param MIN_CREST:
        :return: None
        """
        N = len(self.t)  # number of data points
        M = len(self.f_lines)  # number of parameter
        for p in [2 ** x for x in range(2, PP_MAX)]:
            def update(beta):
                r = np.zeros((N, 1))
                jacob = np.matrix(np.zeros((N, M)))
                for i, t in enumerate(self.t):
                    c_temp = np.dot(self.a, np.cos(2 * np.pi * t * self.f_lines + beta))
                    r[i] = (c_temp ** p) / N
                    jacob[i] = p * (c_temp ** (p - 1)) * (
                            - self.a * np.sin(2 * np.pi * t * self.f_lines + beta)) / N
                return r, jacob, sum(r ** 2)[0] ** (1 / 2 / p)

            self.phi = myopt.nonlinear_least_square(self.phi, update)
            self.multisine()
            if self.crest < MIN_CREST:  # enough
                return True
            print("p, crest=", p, self.crest)

    def save_to_csv(self, noi, input_csv=INPUT_CSV, lines_csv=LINES_CSV, data_csv=DATA_CSV):
        """
        save data to csv file (.csv)
        :param noi:
        :param data_path:
        :param input_csv:
        :param lines_csv:
        :param data_csv:
        :return:
        """
        mycsv.save(self.t, self.u, save_name=self.path + input_csv, header=("time t [s]", "excitation input u []"))
        mycsv.save(self.l_lines, self.f_lines, save_name=self.path + lines_csv,
                   header=("lines []", "frequency lines [Hz]"))
        mycsv.save((self.fs,), (noi,), (self.df,), (self.r,), save_name=self.path + data_csv,
                   header=("Sampling Frequency [Hz]", "Number of iterations []",
                           "Minimum Frequency Resolution [Hz] = 1 / (Excitation Time [s])", "rlog []"))

    def save_to_h(self, input_h=INPUT_H, max_line_number=10):
        """
        save data to header file (.h)
        :param data_path:
        :param input_h:
        :return:
        """
        with open(self.path + input_h, "w") as f:
            for v in (self.fs, self.crest, self.df, self.r):
                f.write("//" + var_name(v, self.__dict__.items()) + ":" + str(v) + "\n")
            f.write("\n")
            f.write("#define NROFS " + str(len(self.u)) + "\n")
            f.write("far float refvec[NROFS]={\n")
            write_str = ""
            for i, u in enumerate(self.u):
                if i == len(self.u) - 1:
                    write_str += str(u)
                    break
                write_str += str(u) + ", "
                if i and not i % max_line_number:
                    write_str += "\n"
            f.write(write_str + "};")


class Simulation():
    def __init__(self, system, noi, path=DATA_PATH, input_csv=INPUT_CSV):
        """

        :param path:
        :param system:
        :param noi:
        """
        self.path = path
        self.system = system
        t, u = mycsv.read_float(path + input_csv)
        dt = t[1] - t[0]
        self.t = [i * dt for i in range(len(t) * noi)]
        self.u = [u[i % len(t)] for i in range(len(t) * noi)]

    def run(self, fig_num, is_noise=True, noise_level=10 ** (-2), output_csv=OUTPUT_CSV):
        """
        run simulation and save result
        :return:
        """
        tout, self.y, x = signal.lsim2(self.system, self.u, self.t)
        noise = np.zeros(len(self.t))
        if is_noise:
            noise = np.random.rand(len(self.t)) * noise_level * max(self.y)
        self.y += noise
        mycsv.save(self.t, self.u, self.y, save_name=self.path + output_csv, header=("t", "u", "y"))
        myplot.plot(fig_num, self.t, self.u)
        myplot.plot(fig_num, self.t, self.y)
        myplot.save(fig_num, label=("time [s]", "u/y []"),
                    save_name=self.path + "Simulation_Result", leg=("u", "y"))


class SystemIdentification():
    U, Y = 0, 1

    def __init__(self, n_num, n_den, transient=2, path=DATA_PATH, output_csv=OUTPUT_CSV, lines_csv=LINES_CSV,
                 data_csv=DATA_CSV, is_zoh_compensation=False):
        """
        System LTI transfer function P(s), where s = 2*pi*f:
        P(s) = (b0 + b1*s + ... + bm*s^m) / (a0 + a1*s + a2*s^2 + ... + an * s^n)
        Paramter vector theta:
        theta = [a0, a1, a2, ..., aden-1, b0, b1, b2, ..., bnum-1]
        :param n_num: number of numerators = m + 1
        :param n_den: number of denominators = n + 1
        :param transient:
        """
        self.path = path
        self.n_num, self.n_den = n_num, n_den
        t, u, y = mycsv.read_float(path + output_csv)  # TIME, INPUT, OUTPUT
        self.fs, noi, df, r = mycsv.read_float(path + data_csv)  # Sampling Frequency, Number of Iteration
        l_lines, self.f_lines = mycsv.read_float(path + lines_csv)  # Lines, Frequency Lines [Hz]
        self.o_lines = 2 * np.pi * self.f_lines  # Angular Frequency Lines [rad/s]
        self.nof, self.nop = len(l_lines), n_den + n_num  # Number of frequency lines, Number of parameter
        noi = int(noi)
        n = len(t) // noi
        # remove DC and transient, and split by iteration
        # [xi-mean(xi)] for i in iterations] for x in (u,y)
        uy_data = [[x[n * i:n * (i + 1)] - np.mean(x[n * i:n * (i + 1)]) for i in range(transient, noi)]
                   for x in (u, y)]
        # [Xdata(f) for f in all(f)] for X in (U,Y), where Xdata = [Xtry(0), Xtry(1), ..., Xtry(noi-transient)]
        uy_f_data = [[fft(x) * 2 / n for x in x_data] for x_data in uy_data]
        # [Udata(f),Ydata(f)] for f in f_lines, removing index not in l_lines, hence no need to remove [N//2:N] of fft(x)
        uy_data_f = [[x_f for i, x_f in enumerate(zip(*x_f_data)) if i in l_lines] for x_f_data in uy_f_data]
        comp = np.ones(self.nof)
        if is_zoh_compensation:
            comp = mysignal.zoh_compensation(tau=1 / self.fs, w=self.o_lines)
        uy_data_f = [[np.array(uy_data_f[x][k]) * comp[k] if x == self.Y else np.array(uy_data_f[x][k])
                      for x in (self.U, self.Y)] for k in range(self.nof)]
        # [[average(Udata(f)), average(Ydata(f))] for f in f_lines]
        # for example, self.uy_av_f[k][self.U] means G_U(Omega_k)
        self.uy_av_f = [[np.mean(x_f) for x_f in data] for data in uy_data_f]
        # [[cov(Udata(f),Udata(f)), cov(Udata(f),Ydata(f)); cov(Ydata(f), Udata(f)), cov(Ydata(f),Ydata(f))] for f in f_lines]
        # for example, self.uy_cov_f[k][self.Y][self.Y] mean Cov(Y,Y)(Omega_k)
        self.uy_cov_f = [np.cov(uy_f) for uy_f in uy_data_f]

        # noi-transient times longer period
        non = n * (noi - transient)
        uy_noise_data = [x[-non:] for x in (u, y)]
        # [Xdata(f) for f in all(f)] for X in (U,Y), where Xdata = [Xtry(0), Xtry(1), ..., Xtry(noi-transient)]
        uy_f_noise_data = [fft(x_data) * 2 / non for x_data in uy_noise_data]
        # [Udata(f),Ydata(f)] for f in f_lines, removing index not in l_lines, hence no need to remove [N//2:N] of fft(x)
        uy_f_noise_data = [[x_f for i, x_f in enumerate(x_f_data)
                            if (not i / (noi - transient) in l_lines) and
                            (l_lines[0] <= i / (noi - transient) <= l_lines[-1])]
                           for x_f_data in uy_f_noise_data]
        self.uy_noise_f = [x for x in zip(*uy_f_noise_data)]
        freq_noise = self.fs / non * np.linspace(0, non // 2 - 1, non // 2)
        self.freq_noise = [f for f in freq_noise
                           if (not f in self.f_lines) and (self.f_lines[0] < f < self.f_lines[-1])]
        self.non = len(self.freq_noise)
        assert len(self.uy_noise_f) == len(self.freq_noise)

    def fap_data(self, var=None):
        """
        return frequecny, amplitude in dB, phase in degrees of signal FRF data or noise
        :param var: if default: return FRF, elif var=="noise": return noise
        :return:
        """
        freq = None
        gyu = None
        if var is None:
            # [average(G(f)) for f in f_lines]
            freq = self.f_lines
            gyu = [g[self.Y] / g[self.U] for g in self.uy_av_f]
        elif var == "noise":
            freq = self.freq_noise
            gyu = [g[self.Y] / 1 for g in self.uy_noise_f]
        gain = 20 * np.log10(np.abs(gyu))
        phi = np.angle(gyu, True)  # in degree
        return freq, gain, phi

    def concatenate(self, old):
        """
        concatenate old system identifcation class into new one
        :param old:
        :return:
        """
        assert old.f_lines[-1] <= self.f_lines[0]
        # signal
        self.f_lines = np.append(old.f_lines, self.f_lines)
        self.o_lines = np.append(old.o_lines, self.o_lines)
        self.nof += old.nof
        old.uy_av_f.extend(self.uy_av_f)
        self.uy_av_f = old.uy_av_f
        old.uy_cov_f.extend(self.uy_cov_f)
        self.uy_cov_f = old.uy_cov_f
        # noise
        self.non += old.non
        old.uy_noise_f.extend(self.uy_noise_f)
        self.uy_noise_f = old.uy_noise_f
        self.freq_noise = np.append(old.freq_noise, self.freq_noise)

    def linear_least_squares(self, w=None):
        """
        https://en.wikipedia.org/wiki/Linear_least_squares_(mathematics)
        Linear Least Square Solution
        :return: <1-D numpy array> of parameter vector theta = [1, a1, a2, ..., an, b0, b1, b2, ..., bm]
        """
        if w is None:
            w = np.ones(self.nof)
        r = np.zeros((self.nof, 1), dtype=complex)
        jacob = np.zeros((self.nof, self.nop - 1), dtype=complex)
        for k in range(self.nof):
            jacob[k] = np.array([- self.uy_av_f[k][self.Y] * (1.j * self.o_lines[k]) ** i if i < self.n_den
                                 else self.uy_av_f[k][self.U] * (1.j * self.o_lines[k]) ** (i - self.n_den)
                                 for i in range(1, self.nop)])
            r[k] = self.uy_av_f[k][self.Y]
            jacob[k] *= w[k]
            r[k] *= w[k]
        j_re = np.matrix([line for mat in (np.real(jacob), np.imag(jacob)) for line in mat])
        r_re = np.array(
            [-line for mat in (np.real(r), np.imag(r)) for line in mat])
        r_re = r_re.reshape(len(r_re), 1)
        theta = - np.linalg.inv(j_re.T * j_re) * j_re.T * np.matrix(r_re)
        theta = np.insert(np.array(theta.T)[0], 0, 1)  # [1, theta]
        theta = stablize_theta(theta, self.n_den)
        return theta

    def iterative_weighted_linear_least_squares(self, MAX_ITER=100):
        """
        Iterative Weighted Linear Least Square
        :param MAX_ITER:
        :return:
        """
        theta = self.linear_least_squares()
        for i in range(MAX_ITER):
            w = []
            for k in range(self.nof):
                w.append(1 / abs(sum(theta[i] * (1.j * self.o_lines[k]) ** i for i in range(self.n_den))))
            theta = self.linear_least_squares(w=w)
        return theta

    def nonlinear_least_squares(self, is_MLE=True, verbose=True, is_stable=True, weight=None):
        """
        Maximum Likelihood Estimation if is_MLE = true,
        for more detail, see https://en.wikipedia.org/wiki/Non-linear_least_squares
        Nonlinear Least Square Solution if is_MLE=false
        :param is_MLE:
        :param verbose:
        :param is_stable: always return stable denominator polynominals if True
        :param weight: frequency weighting factor, len(weigh) == self.nof
        :return: <1-D numpy array> of parameter vector theta = [a0, a1, a2, ..., aden, b0, b1, b2, ..., bnum]
        """
        theta_nls_0 = self.iterative_weighted_linear_least_squares()
        c_mle = np.ones(self.nof)
        if is_MLE:
            # r \to r/\sqrt(2)/\sigma
            c_mle = [np.sqrt(1 / 2 / self.uy_cov_f[k][self.Y][self.Y])
                     for k in range(self.nof)]
        if weight is not None:
            assert len(weight) == self.nof
            for i, w in enumerate(weight):
                c_mle[i] *= w

        def update(theta):
            if is_stable:
                theta = stablize_theta(theta, self.n_den)
            r = np.zeros((self.nof, 1), dtype=complex)
            jacob = np.zeros((self.nof, self.nop), dtype=complex)
            for k in range(self.nof):
                sum_den = sum(theta[i] * (1.j * self.o_lines[k]) ** i for i in range(self.n_den))
                sum_num = sum(theta[i] * (1.j * self.o_lines[k]) ** (i - self.n_den)
                              for i in range(self.n_den, self.nop))
                jacob[k] = [(1.j * self.o_lines[k]) ** i * sum_num / (sum_den ** 2) if i < self.n_den
                            else - (1.j * self.o_lines[k]) ** (i - self.n_den) / sum_den
                            for i in range(self.nop)]
                jacob[k] *= self.uy_av_f[k][self.U]
                r[k] = self.uy_av_f[k][self.Y] - self.uy_av_f[k][self.U] * sum_num / sum_den
                jacob[k] *= c_mle[k]
                r[k] *= c_mle[k]
            j_re = np.matrix([line for mat in (np.real(jacob), np.imag(jacob)) for line in mat])
            r_re = np.array([line for mat in (np.real(r), np.imag(r)) for line in mat])
            r_re = r_re.reshape(len(r_re), 1)
            return r_re, j_re, sum(abs(r) ** 2)[0] / self.nof

        theta = myopt.nonlinear_least_square(theta_nls_0, update, MAX_ITER=1000, verbose=verbose)
        theta = theta / theta[0]
        return np.array(theta) if theta[0] > 0 else np.array(theta) * -1


class SystemIdentificationMIMO():
    U, Y = 0, 1

    def __init__(self, siso_list):
        """

        :param siso_list: [ [G11, G12, ..., G1nin], ..., [Gnin,1, ..., Gnout,nin] ]
        transfer function matrix (nin inputs and nout outputs)
        """

        self.siso_list = siso_list

        self.nin = len(self.siso_list[0])  # number of input
        self.nout = len(self.siso_list)  # number of output
        self.n_den = self.siso_list[0][0].n_den
        self.n_num = []

        den_list, num_list = [], np.array([])
        self.nof = 0
        for x in self.siso_list:
            for y in x:
                theta_mle = y.nonlinear_least_squares(verbose=True)  # MLE Estimation
                den_list.append(theta_mle[:self.n_den])
                num_list = np.append(num_list, theta_mle[y.n_den:])
                self.nof += y.nof
                self.n_num.append(y.n_num)

        a = np.mean(den_list, axis=0)  # initial value is avearage of SISO identification
        b = num_list
        # theta = [a_0, a_1, a_2, ..., a_n,theta, b_0, ..., b_m,theta,11, b_0, ..., b_m,theta,nout,nin]
        self.theta_0 = np.array([x if x > 0 else 0 for x in np.append(a, b)])
        self.nop = len(self.theta_0)

    def nonlinear_least_squares(self, is_MLE=True, verbose=True, weight=None):
        """
        Maximum Likelihood Estimation if is_MLE = true,
        Nonlinear Least Square Solution if is_MLE=false
        :return: <1-D numpy array> of parameter vector theta
         = [a_0, a_1, a_2, ..., a_n,theta, b_0, ..., b_m,theta,11, b_0, ..., b_m,theta,nout,nin]
        """
        theta_nls_0 = self.theta_0

        def update(theta, weight=weight):
            r = np.zeros((self.nof, 1), dtype=complex)
            jacob = np.zeros((self.nof, self.nop), dtype=complex)

            k_offset = 0
            num_offset = self.n_den

            for ix, x in enumerate(self.siso_list):
                for iy, y in enumerate(x):
                    c_mle = np.ones(y.nof)
                    if is_MLE:
                        c_mle = [np.sqrt(1 / 2 / y.uy_cov_f[k][self.Y][self.Y])
                                 for k in range(y.nof)]
                    if weight is not None:
                        assert len(weight) == y.nof
                        for i, w in enumerate(weight):
                            c_mle[i] *= w

                    for k in range(y.nof):
                        sum_den = sum(theta[i] * (1.j * y.o_lines[k]) ** i for i in range(self.n_den))
                        sum_num = sum(theta[num_offset + i] * (1.j * y.o_lines[k]) ** i for i in range(y.n_num))
                        jacob[k_offset + k] += [(1.j * y.o_lines[k]) ** i * sum_num / (sum_den ** 2)
                                                if i < self.n_den else 0 for i in range(self.nop)]
                        jacob[k_offset + k] += [- (1.j * y.o_lines[k]) ** (i - num_offset) / sum_den
                                                if num_offset <= i < num_offset + y.n_num else 0
                                                for i in range(self.nop)]

                        r[k_offset + k] += y.uy_av_f[k][self.Y] - y.uy_av_f[k][self.U] * sum_num / sum_den
                        r[k_offset + k] *= c_mle[k]
                        jacob[k_offset + k] *= y.uy_av_f[k][self.U] * c_mle[k]

                    num_offset += y.n_num
                    k_offset += y.nof

            j_re = np.matrix([line for mat in (np.real(jacob), np.imag(jacob)) for line in mat])
            r_re = np.array([line for mat in (np.real(r), np.imag(r)) for line in mat])
            r_re = r_re.reshape(len(r_re), 1)
            return r_re, j_re, sum(abs(r) ** 2)[0] / self.nof

        theta = myopt.nonlinear_least_square(theta_nls_0, update, MAX_ITER=1000, verbose=verbose)
        theta = theta / theta[0]
        return np.array(theta) if theta[0] > 0 else np.array(theta) * -1

    def get_each(self, theta):
        """
        obtain each parameters
        :param theta: MIMO paramter
        :return: SISO parameters of each transfer function
        """
        theta_list = []
        num_offset = self.n_den
        den = theta[:self.n_den]
        for n in self.n_num:
            theta_list.append(np.append(den, theta[num_offset:num_offset + n]))
            num_offset += n
        return theta_list


def stablize_theta(theta, n_den):
    """
    Stabilize theta denominator: theta[:n_den]
    :param theta: [a0, a1, ..., an_den-1, ...]
    :param n_den:
    :return:
    """
    a, b = theta[:n_den], theta[n_den:]  # denominator, numerator
    a_stab = apolystab(a)
    theta = np.array([*a_stab, *b])  # stable plant
    return theta


def apolystab(p):
    """
    if p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n] has unstable poles,
    stablize all of them, then return stablized cofficients
    :param p: Rank-1 array of polynomial coefficients.
    :return:
    """
    r = np.roots(p)
    r_stab = np.array([-x if np.real(x) > 0 else x for x in r])
    a_stab = np.poly(r_stab)
    return a_stab


def get_path(f_min, f_max, headname="data_frequency_from"):
    """
    return folder path, if not folder exists then create it
    :param min_f:
    :param max_f:
    :return:
    """
    path = DATA_PATH + headname + "_" + str(f_min) + "_Hz_to_" + str(f_max) + "_Hz/"
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def var_name(val, items):
    """
    get the name of variable in class
    :param val:
    :param items:
    :return:
    """
    return [k for k, v in items if id(v) == id(val)][0]


def theta_to_tf(n_den, theta):
    """
    make transfer function from theta vector
    System LTI transfer function P(s), where s = 2*pi*f:
    P(s) = (b0 + b1*s + ... + bm*s^m) / (a0 + a1*s + a2*s^2 + ... + an * s^n)
    Paramter vector theta:
    :param n_den: number of denominators = n+1
    :param theta: [a0, a1, a2, ..., an, b0, b1, b2, ..., bm]
    :return: P(s)
    """
    num, den = theta[n_den:][::-1], theta[:n_den][::-1]
    return signal.TransferFunction(num, den)


def save_theta_to_csv(n_den, theta, path):
    """
    save parameter to csv
    :param n_den:
    :param theta:
    :param path:
    :return:
    """
    num, den = theta[n_den:][::-1], theta[:n_den][::-1]
    num = np.insert(num, 0, np.zeros(len(den) - len(num)))  # padding
    mycsv.save(num, den, save_name=path, header=("num", "den",
                                                 "P(s) = (s^m * num[0] + ... + s * num[-2] + num[-1]) / (s^n * den[0] + ... + s * den[-2] + den[-1])"))
