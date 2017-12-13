"""
read/write CSV files
"""
import csv
import numpy as np


def save(*args, save_name="save_name.csv", header=("",)):
    """
    save data to csv
    :param args: pack of (iterable) variables; for example, (x0,x1,x2,...), (y0,y1,y2...)
    :param save_name:
    :param header:
    """
    with open(save_name, "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(header)
        for x in zip(*args):
            writer.writerow((x))


def read_float(path, is_header=True):
    """
    read data of float from csv
    :param path:
    :param is_header:
    :return:
    """
    with open(path, "r") as f:
        reader = csv.reader(f)
        if is_header:
            header = next(reader)
        return [np.array([float(x) for x in row]) for row in zip(*reader)]
