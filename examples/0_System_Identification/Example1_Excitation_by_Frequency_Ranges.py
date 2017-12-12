"""
Example 1.
Design Optimized Multisine for 2 Frequency Ranges.
Run Simulation with output noise.
Identifiy System Transfer Function, given len(num) and len(den).
"""

from myfdi import *

FS = 4000  # sampling frequency
R = 1.05  # ratio of frequency evolution
F_RANGE = ((10, 100), (100, 500))  # split frequency range into 2 parts to avoid input saturation
NOI = 10  # Number of iteration


def excitation_design(fig):
    """
    Example of excitation signal design
    :param fig:
    :return: fig
    """
    print("\nEntering test of class Excitation\n")

    for i, x in enumerate(F_RANGE):
        f_min, f_max = x  # excitation frequency range
        df = (1, 1)[i]  # frequency resolution
        path = get_path(f_min, f_max)  # parent directory
        print("path=", path)

        ext = Excitation(path=path, fs=FS, f_min=f_min, f_max=f_max, df=df, r=R)

        ext.multisine()
        fig += 1
        myplot.time(ext.t, ext.u, fig,
                    label=("time [s]", "x []"), save_name=path + "non_optimized_multisine", leg=("multisine",),
                    text=(ext.t[-1] * 0.25, 1 * 1.1, "CREST FACTOR = " + str(ext.crest)))
        fig += 1
        myplot.FFT(ext.u, ext.fs, fig,
                   save_name=path + "FFT_of_non_optimized_multisine", leg=("FFT",))

        print("Optimizing random phase multisine")
        ext.optimize()
        ext.save_to_csv(noi=NOI)
        ext.save_to_h()

        fig += 1
        myplot.time(ext.t, ext.u, fig,
                    label=("time [s]", "x []"), save_name=path + "optimized_multisine", leg=("multisine",),
                    text=(ext.t[-1] * 0.25, 1 * 1.1, "CREST FACTOR = " + str(ext.crest)))
        fig += 1
        myplot.FFT(ext.u, ext.fs, fig,
                   save_name=path + "FFT_of_optimized_multisine", leg=("FFT",))
    return fig


def simulation_with_output_noise(fig, plant):
    """

    :return: fig
    """
    print("\nEntering test of class Simulation\n")

    P_true = plant
    print("System:")
    print(P_true)
    for i, x in enumerate(F_RANGE):
        path = get_path(*x)
        print("path=", path)
        sim = Simulation(P_true, noi=NOI, path=path)
        fig += 1
        sim.run(fig_num=fig)
    return fig


def system_identification(fig, plant):
    """
    exmaple of system identification
    :return: fig
    """
    print("\nEntering test class method concatenate\n")

    n_num, n_den = len(plant.num), len(plant.den)  # in real use, you choose correct values for them

    path = get_path(*F_RANGE[0])
    sys_id_old = SystemIdentification(n_num=n_num, n_den=n_den, path=path, transient=3)
    path = get_path(*F_RANGE[1])
    sys_id = SystemIdentification(n_num=n_num, n_den=n_den, path=path, transient=5)
    sys_id.concatenate(sys_id_old)

    fig += 1
    freq, gain, phi = sys_id.fap_data()
    myplot.bodeplot(freq, gain, phi, fig, line_style = 'b-')
    freq, gain, phi = sys_id.fap_data("noise")
    myplot.bodeplot(freq, gain, phi, fig, line_style = 'r+', nos=1)

    print("f_lines:")
    print(sys_id.f_lines)
    theta_lls = sys_id.linear_least_squares()  # LLS Estimation
    print("theta_lls = ")
    print(theta_lls)
    theta_iwls = sys_id.iterative_weighted_linear_least_squares()  # LLS Estimation
    print("theta_iwls = ")
    print(theta_iwls)
    theta_nls = sys_id.nonlinear_least_squares(is_MLE=False)  # NLS Estimation
    print("theta_nls = ")
    print(theta_nls)
    theta_mle = sys_id.nonlinear_least_squares()  # MLE Estimation
    print("theta_mle = ")
    print(theta_mle)
    save_theta_to_csv(n_den, theta_mle, path=DATA_PATH + "theta_mle.csv")

    P_true = plant
    P_lls = theta_to_tf(n_den, theta_lls)
    P_iwls = theta_to_tf(n_den, theta_iwls)
    P_nls = theta_to_tf(n_den, theta_nls)
    P_mle = theta_to_tf(n_den, theta_mle)

    myplot.bode(P_true, fig, w=sys_id.o_lines)
    myplot.bode(P_lls, fig, w=sys_id.o_lines)
    myplot.bode(P_iwls, fig, w=sys_id.o_lines)
    myplot.bode(P_nls, fig, w=sys_id.o_lines)
    myplot.bode(P_mle, fig, w=sys_id.o_lines,
                save_name=DATA_PATH + "System_Identification_from_"
                          + str(min(F_RANGE[0])) + "_Hz_to_" + str(max(F_RANGE[1])) + "_Hz",
                leg=("Signal", "Noise", "P_true", "P_lls", "P_iwls", "P_nls", "P_mle"))
    return fig


def main(plant="r"):
    print("*" * 5 + "Entering test mode !" + "*" * 5 + "\n")

    # Transfer Function of Simulated Transfer Function
    s = symbols('s')
    if plant == "r":
        P = (0.1 *
             1 / (0.1 * s + 1) *
             (100 * 2 * np.pi) ** 2 / (s ** 2 + 2 * 0.001 * 100 * 2 * np.pi * s + (100 * 2 * np.pi) ** 2) *
             (s ** 2 + 2 * 0.1 * 50 * 2 * np.pi * s + (50 * 2 * np.pi) ** 2) / (50 * 2 * np.pi) ** 2 *
             (400 * 2 * np.pi) ** 2 / (s ** 2 + 2 * 0.001 * 400 * 2 * np.pi * s + (400 * 2 * np.pi) ** 2) *
             (300 * 2 * np.pi) ** 2 / (s ** 2 + 2 * 0.01 * 300 * 2 * np.pi * s + (300 * 2 * np.pi) ** 2) *
             (200 * 2 * np.pi) ** 2 / (s ** 2 + 2 * 0.01 * 200 * 2 * np.pi * s + (200 * 2 * np.pi) ** 2))
    else:
        P = 1 / (1 + s / 2 / np.pi / 100)
    P_TRUE = mysignal.symbolic_to_tf(P, s)

    fig = 0
    #fig = excitation_design(fig)
    #fig = simulation_with_output_noise(fig, plant=P_TRUE)
    fig = system_identification(fig, plant=P_TRUE)


if __name__ == "__main__":
    from sympy import *

    main()
