"""
Plot tools
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft


def show():
    """
    show all of figures
    :return:
    """
    plt.show()


def save(fig, save_name="", leg=None, text=None, formats=("png", "pdf"), title="", xl=None, yl=None, label=None):
    """
    save figure to file if leg and save_name given
    :param leg:
    :param save_name:
    :param text:
    :return:
    """
    plt.figure(fig)

    if label:
        plt.xlabel(label[0])
        plt.ylabel(label[1])
    if xl:
        plt.xlim(xl)
    if yl:
        plt.ylim(yl)
    if text:
        plt.text(*text)
    if leg:
        plt.legend(leg)
    if title:
        plt.title(title)
    if save_name:
        for f in formats:
            plt.savefig(save_name + "." + f)


def phase_bind(phase):
    """
    limit phase to -180-180 degrees
    :param phase:
    :return:
    """
    ret = []
    MAX, MIN = 180, -180
    EQ = 360
    for d in phase:
        while d > MAX:
            d -= EQ
        while d < MIN:
            d += EQ
        ret.append(d)
    return ret


def bode(fig_num, sys, w=np.array([])):
    """
    plot bode diagram
    :param sys:
    :param fig_num:
    :param w:
    :param save_name:
    :param leg:
    :return:
    """
    if not w.any():
        w, mag, phase = signal.bode(sys)
    else:
        w, mag, phase = signal.bode(sys, w=w, n=len(w))
    phase = phase_bind(phase)
    plt.figure(fig_num)
    plt.subplot(211)
    plt.semilogx(w / 2 / np.pi, mag, lw=3)
    plt.ylabel("Gain [dB]")
    plt.grid(True)
    # plt.axis('tight')
    plt.subplot(212)
    plt.semilogx(w / 2 / np.pi, phase, lw=3)
    plt.ylabel("Phase [deg]")
    plt.xlabel("Frequency [Hz]")
    plt.axis('tight')
    plt.grid(True)


def plot(fig, x, y, lw=3, line_style="-", plotfunc=plt.plot):
    plt.figure(fig)
    plotfunc(x, y, line_style, lw=lw)
    plt.grid(True)


def scale_fft(xf, fs, N=None, yaxis="dB"):
    """
    calculate amplitude and phase of FFT result
    :param xf: fft(x) where x = x(t)
    :param fs: sampling frequency
    :param N: N = len(xf)
    :param yaxis: absolute or dB
    :return: frequency (Hz), gain, phase (degree)
    """
    if N is None:
        N = len(xf)
    if yaxis.lower() == "db":
        gain = 20 * np.log10(np.abs(xf[:N // 2]) / N * 2)
    elif yaxis.lower() == "abs":
        gain = np.abs(xf[:N // 2]) / N * 2
    phi = np.angle(xf[:N // 2], True)
    freq = fs / N * np.linspace(0, N // 2 - 1, N // 2)
    return freq, gain, phi


def bodeplot(fig, freq, gain, phi, line_style='b-', nos=2, yaxis="dB", xl=None):
    """
    plot bode plot, given frequency, gain, and phase
    :param freq:
    :param gain:
    :param phi:
    :param fig_num:
    :param line_style:
    :param text:
    :param save_name:
    :param leg:
    :param nos:
    :return:
    """
    plt.figure(fig)
    if nos < -1 or nos > 1:
        plt.subplot(212)
        plt.semilogx(freq, phi, line_style, lw=3)
        plt.ylim(-180, 180)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Phase [deg]")
        plt.grid(True)
        if xl:
            plt.xlim(xl)
    if nos >= 1:
        plt.subplot(211)
        plt.semilogx(freq, gain, line_style, lw=3)
        plt.ylabel("Gain [" + yaxis + "]")
        plt.grid(True)
        if xl:
            plt.xlim(xl)


def FFT(fig_num, x, fs, line_style="r+"):
    """
    plot FFT results of signal
    :param x:
    :param fs:
    :param fig_num:
    :param text:
    :param save_name:
    :param leg:
    :return:
    """
    xf = fft(x)
    freq, gain, phi = scale_fft(xf, fs)
    bodeplot(fig_num, freq, gain, phi, line_style="r+")
