"""
Example2
The amplitude of the input spectrum is modified by inverse of LPF(s) for high frequency excitation.
"""

from myfdi import *

FS = 4000  # sampling frequency
R = 1.05  # ratio of frequency evolution
F_RANGE = (100, 1000)  # split frequency range into 2 parts to avoid input saturation


def main():
    f_min, f_max = F_RANGE  # excitation frequency range
    df = 10
    path = get_path(f_min, f_max)  # parent directory
    print("path=", path)
    fig = 0

    ext = Excitation(path=path, fs=FS, f_min=f_min, f_max=f_max, df=df, r=R)

    s = symbols('s')
    a_hpf = (s + 2 * np.pi * 300) / (2 * np.pi * 300)  # Transfer Function of Desired Amplitude Spectrum
    p_hpf = mysignal.symbolic_to_tf(a_hpf, s)
    ext.amplitude(p_hpf)
    ext.multisine()

    fig += 1
    myplot.plot(fig, ext.t, ext.u)
    myplot.save(fig, label=("time [s]", "x []"), save_name=path + "non_optimized_multisine", leg=("multisine",),
                text=(ext.t[-1] * 0.25, 1 * 1.1, "CREST FACTOR = " + str(ext.crest)))
    fig += 1
    myplot.FFT(fig, ext.u, ext.fs)
    myplot.save(fig, save_name=path + "FFT_of_non_optimized_multisine", leg=("FFT",))

    print("Optimizing random phase multisine")
    ext.optimize(PP_MAX=7)

    fig += 1
    myplot.plot(fig, ext.t, ext.u)
    myplot.save(fig, label=("time [s]", "x []"), save_name=path + "optimized_multisine", leg=("multisine",),
                text=(ext.t[-1] * 0.25, 1 * 1.1, "CREST FACTOR = " + str(ext.crest)))
    fig += 1
    myplot.FFT(fig, ext.u, ext.fs)
    myplot.save(fig, save_name=path + "FFT_of_optimized_multisine", leg=("FFT",))


if __name__ == "__main__":
    from sympy import *

    main()
