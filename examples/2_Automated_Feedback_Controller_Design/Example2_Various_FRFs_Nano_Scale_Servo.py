"""
example2_plant_frf.csv is from
page 258, T. Yamaguchi, M. Hirata and H. Fujimoto, Nanoscale servo control, Tokyo Denki University Press, 2002.
"""

import myplot
import mycsv
import mynum
import numpy as np

from Example1_Single_FRF_Nano_Scale_Servo import *

DATAPATH = "data"
NDATA = 18


def plant_data(fig):
    """

    :param fig:
    :return:
    """

    o, reg, img = mycsv.read_float(DATAPATH + "/example2_plant_frf.csv")
    g = reg + 1.j * img
    o_list = mynum.nsplit(o, NDATA)
    g_list = mynum.nsplit(g, NDATA)

    fig += 1
    for oi, gi in zip(o_list, g_list):
        myplot.bodeplot(fig, oi / 2 / np.pi, 20 * np.log10(abs(gi)), np.angle(gi, deg=True), line_style="-",
                        xl=[10 ** 0, 10 ** 4])
    myplot.save(fig, save_name=DATAPATH + "/" + str(fig) + "_plant", title="plant")

    return fig, o, g


if __name__ == "__main__":
    import os

    try:
        os.mkdir(DATAPATH)
    except:
        pass

    fig = -1
    fig, o, g = plant_data(fig)
    fig, fbc = optimize(fig, o, g, nofir=10)
    fig = plotall(fig, fbc, ndata=NDATA)

    myplot.show()
