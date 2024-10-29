from numba import njit
from time import perf_counter
import numpy as np
import coefficient_eval as c_eval
import tqdm
from ast import literal_eval
import configparser

config = configparser.ConfigParser()
config.read("config.ini")
path = config.get("General", "path")
ion = config.get("General", "ion_name")
WoodsSaxon = literal_eval(config.get("Ion", "WoodsSaxon"))
R = WoodsSaxon[ion][0] / 0.197
a = WoodsSaxon[ion][1] / 0.197
Z = WoodsSaxon[ion][3]

Npointsx1 = config.getint("grid", "Npointsx1")
Npointsx2 = config.getint("grid", "Npointsx2")
Npointsqt = config.getint("grid", "Npointsqt")
logx1min = config.getfloat("grid", "logx1min")
logx1max = config.getfloat("grid", "logx1max")
logx2min = config.getfloat("grid", "logx2min")
logx2max = config.getfloat("grid", "logx2max")
logqtmin = config.getfloat("grid", "logqtmin")
logqtmax = config.getfloat("grid", "logqtmax")

X1 = np.logspace(logx1min, logx1max, Npointsx1)
X2 = np.logspace(logx2min, logx2max, Npointsx2)
QT = np.logspace(logqtmin, logqtmax, Npointsqt)


# t1 = perf_counter()
# I = c_eval.I1_eval([QT[1], X1[1], X2[1]], R, a, Z)
# t2 = perf_counter()
# print("new method")
# print(I)
# print(t2 - t1)
# print(ion)


def main():
    integratetime = np.zeros_like(QT)
    for i, q in enumerate(QT):
        # t1 = perf_counter()
        I = c_eval.I3_eval([q, X1[-1], X2[-1]], R, a, Z)
        # t2 = perf_counter()
        # integratetime[i] = t2 - t1
        print(q, I)
    # print(f"avg time: {np.mean(integratetime)}")


if __name__ == "__main__":
    main()
