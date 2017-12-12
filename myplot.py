import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from scipy.fftpack import fft


def save(leg, save_name, is_show, text=None, formats=('jpg', "pdf")):
    """
    save figure to file if leg and save_name given
    :param leg:
    :param save_name:
    :param is_show:
    :param text:
    :return:
    """
    if text:
        plt.text(*text)
    if leg:
        plt.legend(leg)
    if save_name:
        for f in formats:
            plt.savefig(save_name + "." + f)
    if is_show:
        plt.draw()


def bode(sys, fig_num, w=np.array([]), save_name=None, leg=None, is_show=True):
    """
    plot bode diagram
    :param sys:
    :param fig_num:
    :param w:
    :param save_name:
    :param leg:
    :param is_show:
    :return:
    """
    if not w.any():
        w, mag, phase = signal.bode(sys)
    else:
        w, mag, phase = signal.bode(sys, w=w, n=len(w))
    plt.figure(fig_num)
    plt.subplot(211)
    plt.semilogx(w / 2 / np.pi, mag)
    plt.ylabel("Gain [dB]")
    plt.axis('tight')
    plt.subplot(212)
    plt.semilogx(w / 2 / np.pi, phase)
    plt.ylabel("Phase [deg]")
    plt.xlabel("Frequency [Hz]")
    plt.axis('tight')
    save(leg, save_name, is_show)


def time(t, x, fig_num, text=None, label=("time [s]", "x []"), save_name=None, leg=None, is_show=True):
    """
    plot time scale
    :param t:
    :param x:
    :param fig_num:
    :param text:
    :param label:
    :param save_name:
    :param leg:
    :param is_show:
    :return:
    """
    plt.figure(fig_num)
    plt.plot(t, x)
    plt.axis('tight')
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    save(leg, save_name, is_show, text=text)


def scale_fft(xf, fs, N=None, yaxis="dB"):
    if N is None:
        N = len(xf)
    if yaxis.lower() == "db":
        gain = 20 * np.log10(np.abs(xf[:N // 2]) / N * 2)
    elif yaxis.lower() == "abs":
        gain = np.abs(xf[:N // 2]) / N * 2
    phi = np.angle(xf[:N // 2], True)
    freq = fs / N * np.linspace(0, N // 2 - 1, N // 2)
    return freq, gain, phi


def bodeplot(freq, gain, phi, fig_num, line_style='b+', text=None, save_name=None, leg=None, is_show=True, nos=2):
    plt.figure(fig_num)
    plt.subplot(211)
    plt.semilogx(freq, gain, line_style)
    # plt.xlim(min(self.f_lines), max(freq))
    plt.ylabel("Gain [dB]")
    if nos > 1:
        plt.subplot(212)
        plt.semilogx(freq, phi, line_style)
        # plt.xlim(min(self.f_lines), max(freq))
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Phase [deg]")
    save(leg, save_name, is_show, text=text)


def FFT(x, fs, fig_num, text=None, save_name=None, leg=None, is_show=True):
    """
    plot FFT results of signal
    :param x:
    :param fs:
    :param fig_num:
    :param text:
    :param save_name:
    :param leg:
    :param is_show:
    :return:
    """
    xf = fft(x)
    freq, gain, phi = scale_fft(xf, fs)
    bodeplot(freq, gain, phi, fig_num, text=None, save_name=None, leg=None, is_show=True)
