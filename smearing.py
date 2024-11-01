#! /usr/bin/env python

import sys
import os
import random
from ast import literal_eval
from scipy import optimize
from scipy import interpolate
import numpy as np
import copy
import tqdm
import matplotlib.pyplot as plt
import configparser
import coefficient as c


Q2max = 1.0  # 1 GeV^2 as the maximally allowed Q2
ion_Form = 1  # Form1: Q**2=kT**2+(mn*x)**2, Qmin**2=(mn*x)**2;
# Form2: Q**2=kT**2/(1-x)+(mn*x)**2/(1-x), Qmin**2=(mn*x)**2/(1-x)


files = [arg for arg in sys.argv[1:] if arg.startswith("--file=")]
nuclei = [arg for arg in sys.argv[1:] if arg.startswith("--beams=")]
azimuthalSmearing = [arg for arg in sys.argv[1:] if arg.startswith("--azimuthal_smearing=")]

if not files or not nuclei:
    raise Exception("The usage of it should be e.g., ./smearing.py --beams='Pb208 Pb208' --file='/PATH/TO/file.lhe' --out='ktsmearing.lhe4upc' --azimuthal_smearing='True' ")
files = files[0]
files = files.replace("--file=", "")
# files=[file.lower() for file in files.split(' ')]
files = [file for file in files.split(" ")]
files = [files[0]]
nuclei = nuclei[0]
nuclei = nuclei.replace("--beams=", "")
nuclei = [nucleus.rstrip().lstrip() for nucleus in nuclei.split(" ")]

if nuclei[0] == "p":
    A1 = 1
else:
    A1 = float("".join([a for a in nuclei[0] if a.isdigit()]))
if nuclei[1] == "p":
    A2 = 1
else:
    A2 = float("".join([a for a in nuclei[1] if a.isdigit()]))

if not azimuthalSmearing:
    azimuthalSmearing = False
else:
    azimuthalSmearing = azimuthalSmearing[0]
    azimuthalSmearing = azimuthalSmearing.replace("--azimuthal_smearing=", "")
    azimuthalSmearing = literal_eval(azimuthalSmearing)

# name:(RA,aA,wA), RA and aA are in fm, need divide by GeVm12fm to get GeV-1
GeVm12fm = 0.1973
path = os.path.dirname(os.path.abspath(__file__)) + "/"
config = configparser.ConfigParser()
config.read(path + "config.ini")
WoodsSaxon = eval(config.get("Ion", "WoodsSaxon"))

if azimuthalSmearing:
    if nuclei[0] != nuclei[1]:
        tqdm.tqdm.write("Azimuthal distribution is only implemented for symmetric collisions, azimuthal smearing is turned off")
        azimuthalSmearing = False


if nuclei[0] != "p" and nuclei[0] not in WoodsSaxon.keys():
    raise ValueError("do not know the first beam type = %s" % nuclei[0])

if nuclei[1] != "p" and nuclei[1] not in WoodsSaxon.keys():
    raise ValueError("do not know the second beam type = %s" % nuclei[1])

outfile = [arg for arg in sys.argv[1:] if arg.startswith("--out=")]
if not outfile:
    outfile = ["ktsmearing.lhe4upc"]

outfile = outfile[0]
outfile = outfile.replace("--out=", "")

currentdir = os.getcwd()

p_Q2max_save = 1
p_x_array = None  # an array of log10(1/x)
p_xmax_array = None  # an array of maximal function value at logQ2/Q02, where Q02=0.71
p_fmax_array = None  # an array of maximal function value
p_xmax_interp = None
p_fmax_interp = None

offset = 100


def generate_Q2_epa_proton(x, Q2max):
    if x >= 1.0 or x <= 0:
        raise ValueError("x >= 1 or x <= 0")
    mp = 0.938272081  # proton mass in unit of GeV
    mupomuN = 2.793
    Q02 = 0.71  # in unit of GeV**2
    mp2 = mp**2
    Q2min = mp2 * x**2 / (1 - x)

    def xmaxvalue(Q2MAX):
        val = (np.sqrt(Q2MAX * (4 * mp2 + Q2MAX)) - Q2MAX) / (2 * mp2)
        return val

    global p_x_array
    global p_Q2max_save
    global p_xmax_array
    global p_fmax_array
    global p_xmax_interp
    global p_fmax_interp

    if Q2max <= Q2min or x >= xmaxvalue(Q2max):
        return Q2max

    logQ2oQ02max = np.log(Q2max / Q02)
    logQ2oQ02min = np.log(Q2min / Q02)

    def distfun(xx, logQ2oQ02):
        exp = np.exp(logQ2oQ02)
        funvalue = (-8 * mp2**2 * xx**2 + exp**2 * mupomuN**2 * Q02**2 * (2 - 2 * xx + xx**2) + 2 * exp * mp2 * Q02 * (4 - 4 * xx + mupomuN**2 * xx**2)) / (
            2 * exp * (1 + exp) ** 4 * Q02 * (4 * mp2 + exp * Q02)
        )
        return funvalue

    if p_x_array is None or (p_Q2max_save != Q2max):
        # we need to generate the grid first
        p_Q2max_save = Q2max
        xmaxQ2max = xmaxvalue(Q2max)
        log10xmaxQ2maxm1 = np.log10(1 / xmaxQ2max)
        p_x_array = []
        p_xmax_array = []
        p_fmax_array = []
        for log10xm1 in range(10):
            for j in range(10):
                tlog10xm1 = log10xmaxQ2maxm1 + 0.1 * j + log10xm1
                p_x_array.append(tlog10xm1)
                xx = 10 ** (-tlog10xm1)
                if log10xm1 == 0 and j == 0:
                    max_Q2 = logQ2oQ02max
                    max_fun = distfun(xx, max_Q2)
                    p_xmax_array.append(max_Q2)
                    p_fmax_array.append(max_fun)
                else:
                    max_Q2 = optimize.fminbound(lambda x0: -distfun(xx, x0), logQ2oQ02min, logQ2oQ02max, full_output=False, disp=False)
                    max_fun = distfun(xx, max_Q2)
                    p_xmax_array.append(max_Q2)
                    p_fmax_array.append(max_fun)
        p_x_array = np.array(p_x_array)
        p_xmax_array = np.array(p_xmax_array)
        p_fmax_array = np.array(p_fmax_array)
        c1 = interpolate.splrep(p_x_array, p_xmax_array)
        c2 = interpolate.splrep(p_x_array, p_fmax_array)
        p_xmax_interp = lambda x: interpolate.splev(x, c1)
        p_fmax_interp = lambda x: interpolate.splev(x, c2)
    log10xm1 = np.log10(1 / x)
    max_x = p_xmax_interp(log10xm1)
    max_fun = p_fmax_interp(log10xm1)
    logQ2oQ02now = logQ2oQ02min
    while True:
        r1 = random.random()  # a random float number between 0 and 1
        logQ2oQ02now = (logQ2oQ02max - logQ2oQ02min) * r1 + logQ2oQ02min
        w = distfun(x, logQ2oQ02now) / max_fun
        r2 = random.random()  # a random float number between 0 and 1
        if r2 <= w:
            break
    Q2v = np.exp(logQ2oQ02now) * Q02
    return Q2v


A_Q2max_save = [1, 1]
A_x_array = [None, None]  # an array of log10(1/x)
A_xmax_array = [None, None]  # an array of maximal function value at logQ2/Q02, where Q02=0.71
A_fmax_array = [None, None]  # an array of maximal function value
A_xmax_interp = [None, None]
A_fmax_interp = [None, None]


# first beam: ibeam=0; second beam: ibeam=1
def generate_Q2_epa_ion(ibeam, x, Q2max, RA, aA, wA):
    if x >= 1.0 or x <= 0:
        raise ValueError("x >= 1 or x <= 0")
    if ibeam not in [0, 1]:
        raise ValueError("ibeam != 0,1")
    mn = 0.9315  # averaged nucleon mass in unit of GeV
    Q02 = 0.71
    mn2 = mn**2
    if ion_Form == 2:
        Q2min = mn2 * x**2 / (1 - x)
    else:
        Q2min = mn2 * x**2
    RAA = RA / GeVm12fm  # from fm to GeV-1
    aAA = aA / GeVm12fm  # from fm to GeV-1

    def xmaxvalue(Q2MAX):
        val = (np.sqrt(Q2MAX * (4 * mn2 + Q2MAX)) - Q2MAX) / (2 * mn2)
        return val

    global A_x_array
    global A_Q2max_save
    global A_xmax_array
    global A_fmax_array
    global A_xmax_interp
    global A_fmax_interp

    if Q2max <= Q2min or x >= xmaxvalue(Q2max):
        return Q2max

    logQ2oQ02max = np.log(Q2max / Q02)
    logQ2oQ02min = np.log(Q2min / Q02)

    # set rhoA0=1 (irrelvant for this global factor)
    def FchA1(q):
        piqaA = np.pi * q * aAA
        funval = (
            4
            * np.pi**4
            * aAA**3
            / (piqaA**2 * np.sinh(piqaA) ** 2)
            * (
                piqaA * np.cosh(piqaA) * np.sin(q * RAA) * (1 - wA * aAA**2 / RAA**2 * (6 * np.pi**2 / np.sinh(piqaA) ** 2 + np.pi**2 - 3 * RAA**2 / aAA**2))
                - q * RAA * np.sinh(piqaA) * np.cos(q * RAA) * (1 - wA * aAA**2 / RAA**2 * (6 * np.pi**2 / np.sinh(piqaA) ** 2 + 3 * np.pi**2 - RAA**2 / aAA**2))
            )
        )
        return funval

    # set rhoA0=1 (irrelvant for this global factor
    def FchA2(q):
        funval = 0
        # only keep the first two terms
        for n in range(1, 3):
            funval = funval + (-1) ** (n - 1) * n * np.exp(-n * RAA / aAA) / (n**2 + q**2 * aAA**2) ** 2 * (1 + 12 * wA * aAA**2 / RAA**2 * (n**2 - q**2 * aAA**2) / (n**2 + q**2 * aAA**2) ** 2)
        funval = funval * 8 * np.pi * aAA**3
        return funval

    def distfun(xx, logQ2oQ02):
        exp = np.exp(logQ2oQ02) * Q02
        if ion_Form == 2:
            FchA = FchA1(np.sqrt((1 - xx) * exp)) + FchA2(np.sqrt((1 - xx) * exp))
        else:
            FchA = FchA1(np.sqrt(exp)) + FchA2(np.sqrt(exp))
        funvalue = (1 - Q2min / exp) * FchA**2
        return funvalue

    if A_x_array[ibeam] is None or (A_Q2max_save[ibeam] != Q2max):
        # we need to generate the grid first
        tqdm.tqdm.write("INFO: Generate the grid")
        A_Q2max_save[ibeam] = Q2max
        xmaxQ2max = xmaxvalue(Q2max)
        log10xmaxQ2maxm1 = np.log10(1 / xmaxQ2max)
        A_x_array[ibeam] = []
        A_xmax_array[ibeam] = []
        A_fmax_array[ibeam] = []
        for log10xm1 in range(10):
            for j in range(10):
                tlog10xm1 = log10xmaxQ2maxm1 + 0.1 * j + log10xm1
                A_x_array[ibeam].append(tlog10xm1)
                xx = 10 ** (-tlog10xm1)
                if log10xm1 == 0 and j == 0:
                    max_Q2 = logQ2oQ02max
                    max_fun = distfun(xx, max_Q2)
                    A_xmax_array[ibeam].append(max_Q2)
                    A_fmax_array[ibeam].append(max_fun)
                else:
                    max_Q2 = optimize.fminbound(lambda x0: -distfun(xx, x0), logQ2oQ02min, logQ2oQ02max, full_output=False, disp=False)
                    max_fun = distfun(xx, max_Q2)
                    A_xmax_array[ibeam].append(max_Q2)
                    A_fmax_array[ibeam].append(max_fun)
        A_x_array[ibeam] = np.array(A_x_array[ibeam])
        A_xmax_array[ibeam] = np.array(A_xmax_array[ibeam])
        A_fmax_array[ibeam] = np.array(A_fmax_array[ibeam])
        c1 = interpolate.splrep(A_x_array[ibeam], A_xmax_array[ibeam], k=1)
        c2 = interpolate.splrep(A_x_array[ibeam], A_fmax_array[ibeam], k=1)
        A_xmax_interp[ibeam] = lambda x: interpolate.splev(x, c1)
        A_fmax_interp[ibeam] = lambda x: interpolate.splev(x, c2)
        tqdm.tqdm.write("INFO: Grid generated")
    log10xm1 = np.log10(1 / x)
    max_x = A_xmax_interp[ibeam](log10xm1)
    max_fun = A_fmax_interp[ibeam](log10xm1)
    logQ2oQ02now = logQ2oQ02min
    n_rand = 0
    while True:
        r1 = random.random()  # a random float number between 0 and 1
        logQ2oQ02now = (logQ2oQ02max - logQ2oQ02min) * r1 + logQ2oQ02min
        w = distfun(x, logQ2oQ02now) / max_fun
        r2 = random.random()  # a random float number between 0 and 1
        if r2 <= w:
            break
        n_rand += 1
        if n_rand == int(1e6):
            tqdm.tqdm.write("WARNING: It's maybe impossible to find a correct answer")
            tqdm.tqdm.write(f"logQ2oQ02max,logQ2oQ02min,w ={logQ2oQ02max},{logQ2oQ02min},{w}")
        if n_rand == int(1e7):
            tqdm.tqdm.write("ERROR: It's impossible")
    Q2v = np.exp(logQ2oQ02now) * Q02
    return Q2v


# stream=open("Q2.dat",'w')
# for i in range(100000):
#    Q2v=generate_Q2_epa_ion(1,1e-1,1.0,WoodsSaxon['Pb208'][0],\
#                                WoodsSaxon['Pb208'][1],WoodsSaxon['Pb208'][2])
#    stream.write('%12.7e\n'%Q2v)
# stream.close()


def boostl(Q, PBOO, P):
    """Boost P via PBOO with PBOO^2=Q^2 to PLB"""
    # it boosts P from (Q,0,0,0) to PBOO
    # if P=(PBOO[0],-PBOO[1],-PBOO[2],-PBOO[3])
    # it will boost P to (Q,0,0,0)
    PLB = [0, 0, 0, 0]  # energy, px, py, pz in unit of GeV
    PLB[0] = (PBOO[0] * P[0] + PBOO[3] * P[3] + PBOO[2] * P[2] + PBOO[1] * P[1]) / Q
    FACT = (PLB[0] + P[0]) / (Q + PBOO[0])
    for j in range(1, 4):
        PLB[j] = P[j] + FACT * PBOO[j]
    return PLB


def boostl2(Q, PBOO1, PBOO2, P):
    """Boost P from PBOO1 (PBOO1^2=Q^2) to PBOO2 (PBOO2^2=Q^2) frame"""
    PBOO10 = [PBOO1[0], -PBOO1[1], -PBOO1[2], -PBOO1[3]]
    PRES = boostl(Q, PBOO10, P)  # PRES is in (Q,0,0,0) frame
    PLB = boostl(Q, PBOO2, PRES)  # PLB is in PBOO2 frame
    return PLB


def boostToEcm(E1, E2, pext):
    Ecm = 2 * np.sqrt(E1 * E2)
    PBOO = [E1 + E2, 0, 0, E2 - E1]
    pext2 = copy.deepcopy(pext)
    for j in range(len(pext)):
        pext2[j] = boostl(Ecm, PBOO, pext[j])
    return pext2


def boostFromEcm(E1, E2, pext):
    Ecm = 2 * np.sqrt(E1 * E2)
    PBOO = [E1 + E2, 0, 0, E1 - E2]
    pext2 = copy.deepcopy(pext)
    for j in range(len(pext)):
        pext2[j] = boostl(Ecm, PBOO, pext[j])
    return pext2


def deltaPhi(pt1, pt2, phi1, phi_diff):
    phi2 = phi1 + phi_diff
    d = np.abs(
        np.arctan2(pt1 * np.sin(phi1) + pt2 * np.sin(phi2), pt1 * np.cos(phi1) + pt2 * np.cos(phi2)) - np.arctan2(pt1 * np.sin(phi1) - pt2 * np.sin(phi2), pt1 * np.cos(phi1) - pt2 * np.cos(phi2))
    )
    if d > np.pi:
        d = 2 * np.pi - d
    return d


""" L_aim = []
L_real = []
err = [] """


def sufflePhi(pext2, X, w):
    g1, g2, l1, l2 = np.array(pext2)
    N = len(X)
    qt1 = pt(l1)
    qt2 = pt(l2)
    phi1 = phi(l1)
    phi2 = phi(l2)
    phi_diff = phi2 - phi1

    if phi_diff < 0:
        phi_diffmin = 1.05 * phi_diff
        phi_diffmax = 0.95 * phi_diff
    else:
        phi_diffmin = 0.95 * phi_diff
        phi_diffmax = 1.05 * phi_diff
    qt1min = qt1 * 0.9
    qt1max = qt1 * 1.1
    qt2min = qt2 * 0.9
    qt2max = qt2 * 1.1
    dphi = deltaPhi(qt1, qt2, phi1, phi_diff)

    if dphi < np.pi / 4:
        X, w = X[: N // 4], w[: N // 4]
    elif dphi < np.pi / 2:
        X, w = X[N // 4 : N // 2], w[N // 4 : N // 2]
    elif dphi < 3 * np.pi / 4:
        X, w = X[N // 2 : 3 * N // 4], w[N // 2 : 3 * N // 4]
    else:
        X, w = X[3 * N // 4 :], w[3 * N // 4 :]

    w = w / np.sum(w)

    dphi_choosen = np.random.choice(X, p=w)
    qt_pair2 = qt1**2 + qt2**2 + 2 * qt1 * qt2 * np.cos(phi_diff)

    qt_pair2min = qt_pair2 - 1e-10
    qt_pair2max = qt_pair2 + 1e-10

    constrain = optimize.NonlinearConstraint(lambda X: (X[0] ** 2 + X[1] ** 2 + 2 * X[0] * X[1] * np.cos(X[2])), qt_pair2min, qt_pair2max)
    res = optimize.minimize(
        lambda X: np.abs(deltaPhi(X[0], X[1], phi1, X[2]) - dphi_choosen),
        [qt1, qt2, phi_diff],
        bounds=[(qt1min, qt1max), (qt2min, qt2max), (phi_diffmin, phi_diffmax)],
        tol=1e-8,
        constraints=constrain,
    )

    qt1, qt2, phi_diff = res.x
    phi2 = phi1 + phi_diff
    dphi = deltaPhi(qt1, qt2, phi1, phi_diff)
    """ L_aim.append(dphi_choosen)
    L_real.append(dphi)
    err.append(np.abs(dphi-dphi_choosen)) """
    return phi1, phi2, qt1, qt2


def rapidity(p):
    return 0.5 * np.log((p[0] + p[-1]) / (p[0] - p[-1]))


def pt(p):
    return np.sqrt(p[1] ** 2 + p[2] ** 2)


def phi(p):
    return np.arctan2(p[2], p[1])


def M(p):
    M2 = p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2
    return np.sqrt(np.abs(M2))


def phi_distribution(X, pext2, sqrt_s, PID_lepton, RA, aA, wA, Z, ion):
    dict_mass = {11: 0.000511, 13: 0.105658}
    g1, g2, l1, l2 = np.array(pext2)
    pair = g1 + g2
    kt = 0.5 * (l1 - l2)
    Kt = pt(kt)
    ml = dict_mass[PID_lepton]
    mt = np.sqrt(ml**2 + Kt**2)
    y1 = rapidity(l1)
    y2 = rapidity(l2)
    m_pair = M(pair)
    x1 = mt / sqrt_s * (np.exp(y1) + np.exp(y2))
    x2 = mt / sqrt_s * (np.exp(-y1) + np.exp(-y2))
    qt = pt(pair)
    A = c.A_gammagamma(x1, x2, qt, Kt, ml, m_pair, RA, aA, Z, ion)
    B = c.B_gammagamma(x1, x2, qt, Kt, ml, m_pair, RA, aA, Z, ion)
    C = c.C_gammagamma(x1, x2, qt, Kt, ml, m_pair, RA, aA, Z, ion)
    if np.abs(B > A) > 1 or np.abs(C / A) > 1:
        tqdm.tqdm.write(f"WARNING: B or C is larger than A for x1={x1},x2={x2},qt={qt},Kt={Kt},ml={ml},m_pair={m_pair},RA={RA},aA={aA},Z={Z}")
        C = C / 10
        B = B / 10
    w = A + B * np.cos(2 * X) + C * np.cos(4 * X)
    return w


def SetQ(Ecm, x1, x2, Q1, Q2, ph1, ph2, pext):
    Kperp2 = Q1**2 + Q2**2 + 2 * Q1 * Q2 * np.cos(ph1 - ph2)
    Kperp2max = Ecm**2 * (min(1, x1 / x2, x2 / x1) - x1 * x2)
    if Kperp2 >= Kperp2max:
        return None
    x1bar = np.sqrt(x1 / x2 * Kperp2 / Ecm**2 + x1**2)
    x2bar = np.sqrt(x2 / x1 * Kperp2 / Ecm**2 + x2**2)
    if x1bar >= 1.0 or x2bar >= 1.0:
        return None
    pext2 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    # new initial state
    pext2[0][0] = Ecm / 2 * x1bar
    pext2[0][1] = Q1 * np.cos(ph1)
    pext2[0][2] = Q1 * np.sin(ph1)
    pext2[0][3] = Ecm / 2 * x1bar
    pext2[1][0] = Ecm / 2 * x2bar
    pext2[1][1] = Q2 * np.cos(ph2)
    pext2[1][2] = Q2 * np.sin(ph2)
    pext2[1][3] = -Ecm / 2 * x2bar
    # new final state
    PBOO1 = [0, 0, 0, 0]
    PBOO2 = [0, 0, 0, 0]
    for j in range(4):
        PBOO1[j] = pext[0][j] + pext[1][j]
        PBOO2[j] = pext2[0][j] + pext2[1][j]
    Q = np.sqrt(x1 * x2) * Ecm
    for j in range(2, len(pext)):
        pext2[j] = boostl2(Q, PBOO1, PBOO2, pext[j])
    return pext2


def MinimizeFunction(Ecm, x1, x2, Q1, Q2, ph1, ph2, pext, dphiaim):
    pext2 = SetQ(Ecm, x1, x2, Q1, Q2, ph1, ph2, pext)
    g1, g2, l1, l2 = np.array(pext2)
    pt1 = pt(l1)
    pt2 = pt(l2)
    phi1 = phi(l1)
    phi2 = phi(l2)
    phidiff = phi2 - phi1
    dPhi = deltaPhi(pt1, pt2, phi1, phidiff)
    return dPhi - dphiaim


L = []


def LinearCoefficients(x1, x2, y1, y2):
    return (y1 - y2) / (x1 - x2), (x1 * y2 - x2 * y1) / (x1 - x2)


def FindRoot(initialGuess, f):
    off = [0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4]
    for o in off:
        x2 = initialGuess + o
        y1 = f(initialGuess)
        y2 = f(x2)
        a, b = LinearCoefficients(initialGuess, x2, y1, y2)

        if np.abs(f(-b / a)) < 1e-2:
            return -b / a
    return None


def DefineBound(initialGuess, f):
    off = [0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4]
    for o in off:
        x2 = initialGuess + o
        y1 = f(initialGuess)
        y2 = f(x2)
        if y1 * y2 < 0:
            m = np.min([initialGuess,x2])
            mm = np.max([initialGuess,x2])
            return [m,mm]
    return None


def InitialMomentumReshuffle(Ecm, x1, x2, Q1, Q2, pext, PID_lepton, RA, aA, wA, Z, ion):
    [ph1, ph2] = 2 * np.pi * np.random.rand(2)
    pext2 = SetQ(Ecm, x1, x2, Q1, Q2, ph1, ph2, pext)
    if not(azimuthalSmearing):
        return pext2
    g1,g2,_,_ = np.array(pext2)
    pt_old = pt(g1+g2)
    dPhiArray = np.linspace(0, np.pi, 10000)
    distrib = phi_distribution(dPhiArray, pext2, Ecm, PID_lepton, RA, aA, wA, Z, ion)
    dphiaim = np.random.choice(dPhiArray, p=distrib / np.sum(distrib))

    def f1(x):
        return MinimizeFunction(Ecm, x1, x2, Q1, Q2, x, ph2, pext, dphiaim)

    def f2(x):
        return MinimizeFunction(Ecm, x1, x2, Q1, Q2, ph1, x, pext, dphiaim)

    bounds = DefineBound(ph1, f1)
    counter = 0
    while bounds is None and counter < 10:
        counter += 1
        guess = 2 * np.pi * np.random.rand()
        bounds = DefineBound(guess, f1)
    if not (bounds is None):
        root1 = optimize.root_scalar(f1, bracket=bounds).root
        pext2 = SetQ(Ecm, x1, x2, Q1, Q2, root1, ph2, pext)
        g1,g2,_,_ = np.array(pext2)
        pt_new = pt(g1+g2)
        L.append(pt_old-pt_new)
        return pext2

    bounds = DefineBound(ph2, f2)
    counter = 0
    while bounds is None and counter < 10:
        counter += 1
        guess = 2 * np.pi * np.random.rand()
        bounds = DefineBound(guess, f2)
    if not (bounds is None):
        root2 = optimize.root_scalar(f2, bracket=bounds).root
        pext2 = SetQ(Ecm, x1, x2, Q1, Q2, ph1, root2, pext)
        g1,g2,_,_ = np.array(pext2)
        pt_new = pt(g1+g2)
        L.append(pt_old-pt_new)
        return pext2
    return None
    # X = np.linspace(0, 2 * np.pi, 100)
    # Y1 = np.array([f1(x) for x in X])
    # Y2 = np.array([f2(x) for x in X])
    #
    # neg1 = X[Y1 < 0]
    # pos1 = X[Y1 > 0]
    # neg2 = X[Y2 < 0]
    # pos2 = X[Y2 > 0]
    #
    # if len(neg1) != 0 and len(pos1) != 0:
    #     neg1 = neg1[0]
    #     pos1 = pos1[0]
    #     bound1 = np.min([neg1, pos1])
    #     bound2 = np.max([neg1, pos1])
    #     res = optimize.root_scalar(f1, bracket=[bound1, bound2])
    #     ph1 = res.root
    #     pext2 = SetQ(Ecm, x1, x2, Q1, Q2, ph1, ph2, pext)
    # elif len(neg2) != 0 and len(pos2) != 0:
    #     neg2 = neg2[0]
    #     pos2 = pos2[0]
    #     bound1 = np.min([neg2, pos2])
    #     bound2 = np.max([neg2, pos2])
    #     res = optimize.root_scalar(f2, bracket=[bound1, bound2])
    #     ph2 = res.root
    #     pext2 = SetQ(Ecm, x1, x2, Q1, Q2, ph1, ph2, pext)
    # return pext2


headers = []
inits = []
events = []

ninit0 = 0
ninit1 = 0
firstinit = ""
E_beam1 = 0
E_beam2 = 0
PID_beam1 = 0
PID_beam2 = 0
nan_count = 0
nevent = 0
ilil = 0
count = 0
if nuclei[0] != "p":
    A1 = int("".join([a for a in nuclei[0] if a.isdigit()]))
    A1sym = "".join([a for a in nuclei[0] if a and not a.isdigit()])
    Z1 = WoodsSaxon[nuclei[0]][3]
    IonID_beam1 = 1000000000 + Z1 * 10000 + A1 * 10
if nuclei[1] != "p":
    A2 = int("".join([a for a in nuclei[1] if a.isdigit()]))
    A2sym = "".join([a for a in nuclei[1] if a and not a.isdigit()])
    Z2 = WoodsSaxon[nuclei[1]][3]
    IonID_beam2 = 1000000000 + Z2 * 10000 + A2 * 10

for i, file in enumerate(files):
    N_event = 0

    stream = open(file, "r")
    tqdm.tqdm.write("INFO: Counting the number of events in file")
    for line in stream:
        if "<event>" in line or "<event " in line:
            N_event += 1
    tqdm.tqdm.write(f"INFO: Number of events in file = {N_event}")
    pbar = tqdm.tqdm(total=N_event)
    stream.close()
    stream = open(file, "r")
    headQ = True
    initQ = False
    iinit = -1
    ievent = -1
    eventQ = False
    this_event = []
    n_particles = 0
    rwgtQ = False
    procid = None
    proc_dict = {}
    tqdm.tqdm.write("INFO: Start processing the file")
    for line in stream:
        sline = line.replace("\n", "")
        if "<init>" in line or "<init " in line:
            initQ = True
            headQ = False
            iinit = iinit + 1
            if i == 0:
                inits.append(sline)
        elif headQ and i == 0:
            headers.append(sline)
        elif "</init>" in line or "</init " in line:
            initQ = False
            iinit = -1
            if i == 0:
                inits.append(sline)
        elif initQ:
            iinit = iinit + 1
            if "<generator name=" in line:
                inits.append(sline)
            elif iinit == 1:
                if i == 0:
                    firstinit = sline
                    ninit0 = len(inits)
                    inits.append(sline)
                    firstinit = firstinit.rsplit(" ", 1)[0]
                    ff = firstinit.strip().split()
                    PID_beam1 = int(ff[0])
                    PID_beam2 = int(ff[1])
                    E_beam1 = float(ff[2]) / A1
                    E_beam2 = float(ff[3]) / A2
                    if abs(PID_beam1) != 2212 and abs(PID_beam1) != IonID_beam1:
                        raise ValueError(f"The first beam does not match. In lhe file:{PID_beam1}, expected: {IonID_beam1}")
                    if abs(PID_beam2) != 2212 and abs(PID_beam2) != IonID_beam2:
                        raise ValueError(f"The second beam does not match.  In lhe file:{PID_beam2}, expected: {IonID_beam2}")
                    ninit1 = int(sline.rsplit(" ", 1)[-1])
                else:
                    ninit1 = ninit1 + int(sline.rsplit(" ", 1)[-1])
                    sline = sline.rsplit(" ", 1)[0]
                    if not sline == firstinit:
                        tqdm.tqdm.write("the beam information of the LHE files is not identical")
                        raise Exception
            elif iinit >= 2:
                procid = sline.split()[-1]
                procpos = sline.index(" " + procid)
                ilil = ilil + 1
                # sline=sline[:procpos]+(' %d'%(offset+ilil))
                proc_dict[procid] = offset + ilil
                tqdm.tqdm.write(sline)
                if i == 0:
                    inits.append(sline)
                else:
                    inits.insert(-1, sline)
            else:
                tqdm.tqdm.write("should not reach here. Do not understand the <init> block")
                raise Exception
        elif "<event>" in line or "<event " in line:
            eventQ = True
            ievent = ievent + 1
            events.append(sline)
        elif "</event>" in line or "</event " in line:
            nevent = nevent + 1
            eventQ = False
            rwgtQ = False
            ievent = -1
            this_event = []
            n_particles = 0
            events.append(sline)
        elif eventQ:
            ievent = ievent + 1
            if ievent == 1:
                found = False
                for procid, new_procid in proc_dict.items():
                    if " " + procid + " " not in sline:
                        continue
                    procpos = sline.index(" " + procid + " ")
                    found = True
                    sline = sline[:procpos] + (" %d" % (new_procid)) + sline[procpos + len(" " + procid) :]
                    break
                if not found:
                    tqdm.tqdm.write("do not find the correct proc id !")
                    raise Exception
                n_particles = int(sline.split()[0])
                # procpos=sline.index(' '+procid)
                # sline=sline[:procpos]+(' %d'%(1+i))+sline[procpos+len(' '+procid):]
            elif "<mgrwt" in sline:
                rwgtQ = True
            elif "</mgrwt" in sline:
                rwgtQ = False
            elif not rwgtQ:
                sline2 = sline.split()
                particle = [
                    int(sline2[0]),
                    int(sline2[1]),
                    int(sline2[2]),
                    int(sline2[3]),
                    int(sline2[4]),
                    int(sline2[5]),
                    float(sline2[6]),
                    float(sline2[7]),
                    float(sline2[8]),
                    float(sline2[9]),
                    float(sline2[10]),
                    float(sline2[11]),
                    float(sline2[12]),
                ]
                this_event.append(particle)
                if ievent == n_particles + 1:
                    # get the momenta and masses
                    x1 = this_event[0][9] / E_beam1
                    x2 = this_event[1][9] / E_beam2
                    if np.isnan(x1) or np.isnan(x2):
                        if nan_count < 5:
                            tqdm.tqdm.write("Warning: x1 or x2 is nan")
                        if nan_count == 5:
                            tqdm.tqdm.write("Other warning will be suppressed")
                        nan_count = nan_count + 1
                        continue
                    pext = []
                    mass = []
                    for j in range(n_particles):
                        pext.append([this_event[j][9], this_event[j][6], this_event[j][7], this_event[j][8]])
                        mass.append(this_event[j][10])
                    PID_lepton = np.abs(this_event[-1][0])

                    # first we need to boost from antisymmetric beams to symmetric beams
                    if E_beam1 != E_beam2:
                        pext = boostToEcm(E_beam1, E_beam2, pext)
                    Ecm = 2 * np.sqrt(E_beam1 * E_beam2)
                    pext_new = None
                    Q1 = 0
                    Q2 = 0
                    while pext_new == None:
                        # generate Q1 and Q2
                        if nuclei[0] == "p":
                            RA, aA, wA, Z = None, None, None, None
                            Q12 = generate_Q2_epa_proton(x1, Q2max)
                        else:
                            RA, aA, wA, Z = WoodsSaxon[nuclei[0]]
                            Q12 = generate_Q2_epa_ion(0, x1, Q2max, RA, aA, wA)
                        if nuclei[1] == "p":
                            RA, aA, wA, Z = None, None, None, None
                            Q22 = generate_Q2_epa_proton(x2, Q2max)
                        else:
                            if nuclei[0] == nuclei[1]:
                                RA, aA, wA, Z = WoodsSaxon[nuclei[0]]
                                Q22 = generate_Q2_epa_ion(0, x2, Q2max, RA, aA, wA)
                            else:
                                RA, aA, wA, Z = WoodsSaxon[nuclei[1]]
                                Q22 = generate_Q2_epa_ion(1, x2, Q2max, RA, aA, wA)
                        Q1 = np.sqrt(Q12)
                        Q2 = np.sqrt(Q22)
                        # perform the initial momentum reshuffling
                        pext_new = InitialMomentumReshuffle(Ecm, x1, x2, Q1, Q2, pext, PID_lepton, RA, aA, wA, Z, nuclei[0])
                        count = count + 1
                    if E_beam1 != E_beam2:
                        # boost back from the symmetric beams to antisymmetric beams
                        pext_new = boostFromEcm(E_beam1, E_beam2, pext_new)
                    # update the event information
                    # negative invariant mass means negative invariant mass square (-Q**2, spacelike)
                    this_event[0][10] = -Q1
                    this_event[1][10] = -Q2
                    for j in range(n_particles):
                        this_event[j][9] = pext_new[j][0]
                        this_event[j][6] = pext_new[j][1]
                        this_event[j][7] = pext_new[j][2]
                        this_event[j][8] = pext_new[j][3]
                        # Correct the issue with precision
                        # newsline_old="      %d    %d     %d    %d    %d    %d  %12.7e  %12.7e  %12.7e  %12.7e  %12.7e  %12.7e  %12.7e"%tuple(this_event[j])
                        newsline = "      %d    %d     %d    %d    %d    %d  %12.16e  %12.16e  %12.16e  %12.16e  %12.16e  %12.16e  %12.16e" % tuple(this_event[j])

                        events.append(newsline)
                continue
            events.append(sline)
        if "<event>" in line or "<event" in line:
            pbar.update(1)
    stream.close()
    pbar.close()

# modify the number of process information
firstinit = firstinit + (" %d" % ninit1)
inits[ninit0] = firstinit

text = "\n".join(headers) + "\n"
text = text + "\n".join(inits) + "\n"
text = text + "\n".join(events)
text = text + "\n</LesHouchesEvents>"
stream = open(outfile, "w")
stream.write(text)
stream.close()
retry = count - nevent
tqdm.tqdm.write(f"INFO: The final produced lhe file is {outfile}")
if retry > nevent / 10:
    tqdm.tqdm.write(f"WARNING: A lot of retries were needed for the momentum reshuffling, please check the input lhe files and be careful with the results")
if nan_count > 0:
    tqdm.tqdm.write(f"INFO: The ratio of nan is {nan_count/nevent} for {nevent} events")
if nan_count / nevent > 0.01:
    tqdm.tqdm.write("WARNING: The ratio of nan is too large, please check the input lhe files")
plt.hist(L, 100)
plt.show()
