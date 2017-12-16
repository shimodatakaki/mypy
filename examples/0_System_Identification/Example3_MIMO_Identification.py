"""
Example3_MIMO_Identification:
MIMO Identification of G=[[G11, G12],[G21,G22]]
"""

from myfdi import *

FS = 4000  # sampling frequency
R = 1.02  # ratio of frequency evolution
F_RANGE = (10, 200)
NOI = 10  # Number of iteration

def sys_id():
    path = get_path(*F_RANGE, "data_1")
    sys_id_11 = SystemIdentification(n_num=3, n_den=4, path=path, transient=3)

    path = get_path(*F_RANGE, "data_2")
    sys_id_12 = SystemIdentification(n_num=1, n_den=4, path=path, transient=3)

    path = get_path(*F_RANGE, "data_3")
    sys_id_21 = SystemIdentification(n_num=1, n_den=4, path=path, transient=3)

    path = get_path(*F_RANGE, "data_4")
    sys_id_22 = SystemIdentification(n_num=3, n_den=4, path=path, transient=3)

    mimo_id = SystemIdentificationMIMO([[sys_id_11, sys_id_12], [sys_id_21, sys_id_22]])

    print("nin, nout: ", mimo_id.nin, mimo_id.nout)
    print("theta_0:\n", mimo_id.theta_0)
    print("nof: ", mimo_id.nof)
    print("n_den, n_num: ",mimo_id.n_den, mimo_id.n_num)
    print("nop: ", mimo_id.nop)

    theta_mle = mimo_id.nonlinear_least_squares()
    print("theta_mle = ")
    print(theta_mle)

    # Transfer Function of Simulated Transfer Function
    s = symbols('s')
    J = 0.01
    oz = 100 * 2 * np.pi
    oa = 50 * 2 * np.pi
    D = (J * s + 0.01) * (s ** 2 + 0.1 * s + oz ** 2)
    G11 = (s ** 2 + 0.1 * s + oa ** 2) / D
    G12 = oa ** 2 / D

    G11 = mysignal.symbolic_to_tf(G11, s)
    G12 = mysignal.symbolic_to_tf(G12, s)

    G21 = G12
    G22 = G11

    G = [G11, G12, G21, G22]

    o_lines = sys_id_11.o_lines

    fig = 0
    for i, theta in enumerate(mimo_id.get_each(theta_mle)):
        print(theta)
        save_theta_to_csv(mimo_id.n_den, theta, 'data/mimo_theta_mle_'+str(i+1)+'.csv')
        G_mle = theta_to_tf(mimo_id.n_den, theta)
        print("MLE", str(i),  G_mle)
        fig += 1
        myplot.bode(G[i], fig, w=o_lines)
        myplot.bode(G_mle, fig, w=o_lines,
                    save_name=DATA_PATH + str(i) + "_MIMO_System_Identification",
                leg=("P_true"+str(i), "P_mle"+str(i)))


def main():
    sys_id()

if __name__ == "__main__":
    from sympy import *

    main()