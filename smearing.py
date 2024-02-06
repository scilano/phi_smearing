#! /usr/bin/env python

import sys
import os
import math
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

Q2max=1.0 # 1 GeV^2 as the maximally allowed Q2
ion_Form=1 # Form1: Q**2=kT**2+(mn*x)**2, Qmin**2=(mn*x)**2; 
           # Form2: Q**2=kT**2/(1-x)+(mn*x)**2/(1-x), Qmin**2=(mn*x)**2/(1-x)


files=[arg for arg in sys.argv[1:] if arg.startswith('--file=')]
nuclei=[arg for arg in sys.argv[1:] if arg.startswith('--beams=')]
azimuthalSmearing=[arg for arg in sys.argv[1:] if arg.startswith('--azimuthal_smearing=')]

if not files or not nuclei:
    raise Exception("The usage of it should be e.g., ./smearing.py --beams='Pb208 Pb208' --file='/PATH/TO/file.lhe' --out='ktsmearing.lhe4upc' --azimuthal_smearing='True' ")
files=files[0]
files=files.replace('--file=','')
#files=[file.lower() for file in files.split(' ')]
files=[file for file in files.split(' ')]
files=[files[0]]
nuclei=nuclei[0]
nuclei=nuclei.replace('--beams=','')
nuclei=[nucleus.rstrip().lstrip() for nucleus in nuclei.split(' ')]
if not azimuthalSmearing:
    azimuthalSmearing = True
else:
    azimuthalSmearing=azimuthalSmearing[0]
    azimuthalSmearing=azimuthalSmearing.replace('--azimuthal_smearing=','')
    azimuthalSmearing=literal_eval(azimuthalSmearing)

# name:(RA,aA,wA), RA and aA are in fm, need divide by GeVm12fm to get GeV-1
GeVm12fm=0.1973
config = configparser.ConfigParser()
config.read('/afs/cern.ch/user/n/ncrepet/work/scripts/phi_smearing/config.ini')
WoodsSaxon = eval(config.get("Ion","WoodsSaxon"))

if azimuthalSmearing:
    if nuclei[0] != nuclei[1]:
        raise ValueError("Azimuthal distribution is only implemented for symmetric collisions")
    else:
        config.set("General","ion_name",nuclei[0])

if nuclei[0] != 'p' and nuclei[0] not in WoodsSaxon.keys():
    raise ValueError('do not know the first beam type = %s'%nuclei[0])

if nuclei[1] != 'p' and nuclei[1] not in WoodsSaxon.keys():
    raise ValueError('do not know the second beam type = %s'%nuclei[1])

outfile=[arg for arg in sys.argv[1:] if arg.startswith('--out=')]
if not outfile:
    outfile=['ktsmearing.lhe4upc']

outfile=outfile[0]
outfile=outfile.replace('--out=','')

currentdir=os.getcwd()

p_Q2max_save=1
p_x_array=None # an array of log10(1/x)
p_xmax_array=None # an array of maximal function value at logQ2/Q02, where Q02=0.71
p_fmax_array=None # an array of maximal function value
p_xmax_interp=None
p_fmax_interp=None

offset=100

def generate_Q2_epa_proton(x,Q2max):
    if x >= 1.0 or x <= 0:
        raise ValueError("x >= 1 or x <= 0")
    mp=0.938272081 # proton mass in unit of GeV
    mupomuN=2.793
    Q02=0.71  # in unit of GeV**2
    mp2=mp**2
    Q2min=mp2*x**2/(1-x)

    def xmaxvalue(Q2MAX):
        val=(math.sqrt(Q2MAX*(4*mp2+Q2MAX))-Q2MAX)/(2*mp2)
        return val

    global p_x_array
    global p_Q2max_save
    global p_xmax_array
    global p_fmax_array
    global p_xmax_interp
    global p_fmax_interp

    if Q2max <= Q2min or x >= xmaxvalue(Q2max) : return Q2max

    logQ2oQ02max = math.log(Q2max/Q02)
    logQ2oQ02min = math.log(Q2min/Q02)

    def distfun(xx,logQ2oQ02):
        exp=math.exp(logQ2oQ02)
        funvalue=(-8*mp2**2*xx**2+exp**2*mupomuN**2*Q02**2*\
                       (2-2*xx+xx**2)+2*exp*mp2*Q02*(4-4*xx+mupomuN**2*xx**2))\
                       /(2*exp*(1+exp)**4*Q02*(4*mp2+exp*Q02))
        return funvalue

    if p_x_array is None or (p_Q2max_save != Q2max):
        # we need to generate the grid first
        p_Q2max_save = Q2max
        xmaxQ2max=xmaxvalue(Q2max)
        log10xmaxQ2maxm1=math.log10(1/xmaxQ2max)
        p_x_array=[]
        p_xmax_array=[]
        p_fmax_array=[]
        for log10xm1 in range(10):
            for j in range(10):
                tlog10xm1=log10xmaxQ2maxm1+0.1*j+log10xm1
                p_x_array.append(tlog10xm1)
                xx=10**(-tlog10xm1)
                if log10xm1 == 0 and j == 0:
                    max_Q2 = logQ2oQ02max
                    max_fun = distfun(xx,max_Q2)
                    p_xmax_array.append(max_Q2)
                    p_fmax_array.append(max_fun)
                else:
                    max_Q2 = optimize.fmin(lambda x0: -distfun(xx,x0),\
                                                    (logQ2oQ02max+logQ2oQ02min)/2,\
                                               full_output=False,disp=False)
                    max_fun = distfun(xx,max_Q2[0])
                    p_xmax_array.append(max_Q2[0])
                    p_fmax_array.append(max_fun)
        p_x_array=np.array(p_x_array)
        p_xmax_array=np.array(p_xmax_array)
        p_fmax_array=np.array(p_fmax_array)
        c1 = interpolate.splrep(p_x_array,p_xmax_array)
        c2 = interpolate.splrep(p_x_array,p_fmax_array)
        p_xmax_interp= lambda x: interpolate.splev(x,c1)
        p_fmax_interp= lambda x: interpolate.splev(x,c2)
    log10xm1=math.log10(1/x)
    max_x = p_xmax_interp(log10xm1)
    max_fun = p_fmax_interp(log10xm1)
    logQ2oQ02now=logQ2oQ02min
    while True:
        r1=random.random() # a random float number between 0 and 1
        logQ2oQ02now=(logQ2oQ02max-logQ2oQ02min)*r1+logQ2oQ02min
        w=distfun(x,logQ2oQ02now)/max_fun
        r2=random.random() # a random float number between 0 and 1
        if r2 <= w: break
    Q2v=math.exp(logQ2oQ02now)*Q02
    return Q2v

A_Q2max_save=[1,1]
A_x_array=[None,None]  # an array of log10(1/x)
A_xmax_array=[None,None] # an array of maximal function value at logQ2/Q02, where Q02=0.71
A_fmax_array=[None,None] # an array of maximal function value
A_xmax_interp=[None,None]
A_fmax_interp=[None,None]

# first beam: ibeam=0; second beam: ibeam=1
def generate_Q2_epa_ion(ibeam,x,Q2max,RA,aA,wA):
    if x >= 1.0 or x <= 0:
        raise ValueError("x >= 1 or x <= 0")
    if ibeam not in [0,1]:
        raise ValueError("ibeam != 0,1")
    mn=0.9315 # averaged nucleon mass in unit of GeV
    Q02=0.71
    mn2=mn**2
    if ion_Form == 2:
        Q2min=mn2*x**2/(1-x)
    else:
        Q2min=mn2*x**2
    RAA=RA/GeVm12fm # from fm to GeV-1
    aAA=aA/GeVm12fm # from fm to GeV-1
    
    
    def xmaxvalue(Q2MAX):
        val=(math.sqrt(Q2MAX*(4*mn2+Q2MAX))-Q2MAX)/(2*mn2)
        return val

    global A_x_array
    global A_Q2max_save
    global A_xmax_array
    global A_fmax_array
    global A_xmax_interp
    global A_fmax_interp

    if Q2max <= Q2min or x >= xmaxvalue(Q2max) : return Q2max

    logQ2oQ02max = math.log(Q2max/Q02)
    logQ2oQ02min = math.log(Q2min/Q02)

    # set rhoA0=1 (irrelvant for this global factor)
    def FchA1(q):
        piqaA=math.pi*q*aAA
        funval=4*math.pi**4*aAA**3/(piqaA**2*math.sinh(piqaA)**2)*\
            (piqaA*math.cosh(piqaA)*math.sin(q*RAA)*(1-wA*aAA**2/RAA**2*\
            (6*math.pi**2/math.sinh(piqaA)**2+math.pi**2-3*RAA**2/aAA**2))\
            -q*RAA*math.sinh(piqaA)*math.cos(q*RAA)*(1-wA*aAA**2/RAA**2*\
            (6*math.pi**2/math.sinh(piqaA)**2+3*math.pi**2-RAA**2/aAA**2)))
        return funval

    # set rhoA0=1 (irrelvant for this global factor
    def FchA2(q):
        funval=0
        # only keep the first two terms
        for n in range(1,3):
            funval=funval+(-1)**(n-1)*n*math.exp(-n*RAA/aAA)/(n**2+q**2*aAA**2)**2*\
                (1+12*wA*aAA**2/RAA**2*(n**2-q**2*aAA**2)/(n**2+q**2*aAA**2)**2)
        funval=funval*8*math.pi*aAA**3
        return funval

    def distfun(xx,logQ2oQ02):
        exp=math.exp(logQ2oQ02)*Q02
        if ion_Form == 2:
            FchA=FchA1(math.sqrt((1-xx)*exp))+FchA2(math.sqrt((1-xx)*exp))
        else:
            FchA=FchA1(math.sqrt(exp))+FchA2(math.sqrt(exp))
        funvalue=(1-Q2min/exp)*FchA**2
        return funvalue
    
    if A_x_array[ibeam] is None or (A_Q2max_save[ibeam] != Q2max):
        # we need to generate the grid first
        tqdm.tqdm.write("INFO: Generate the grid")
        A_Q2max_save[ibeam] = Q2max
        xmaxQ2max=xmaxvalue(Q2max)
        log10xmaxQ2maxm1=math.log10(1/xmaxQ2max)
        A_x_array[ibeam]=[]
        A_xmax_array[ibeam]=[]
        A_fmax_array[ibeam]=[]
        for log10xm1 in range(10):
            for j in range(10):
                tlog10xm1=log10xmaxQ2maxm1+0.1*j+log10xm1
                A_x_array[ibeam].append(tlog10xm1)
                xx=10**(-tlog10xm1)
                if log10xm1 == 0 and j == 0:
                    max_Q2 = logQ2oQ02max
                    max_fun = distfun(xx,max_Q2)
                    A_xmax_array[ibeam].append(max_Q2)
                    A_fmax_array[ibeam].append(max_fun)
                else:
                    max_Q2 = optimize.fmin(lambda x0: -distfun(xx,x0),\
                                                    (logQ2oQ02max+logQ2oQ02min)/2,\
                                               full_output=False,disp=False)
                    max_fun = distfun(xx,max_Q2[0])
                    A_xmax_array[ibeam].append(max_Q2[0])
                    A_fmax_array[ibeam].append(max_fun)
        A_x_array[ibeam]=np.array(A_x_array[ibeam])
        A_xmax_array[ibeam]=np.array(A_xmax_array[ibeam])
        A_fmax_array[ibeam]=np.array(A_fmax_array[ibeam])
        c1 = interpolate.splrep(A_x_array[ibeam],A_xmax_array[ibeam],k=1)
        c2 = interpolate.splrep(A_x_array[ibeam],A_fmax_array[ibeam],k=1)
        A_xmax_interp[ibeam]=lambda x: interpolate.splev(x,c1)
        A_fmax_interp[ibeam]=lambda x: interpolate.splev(x,c2)
        tqdm.tqdm.write("INFO: Grid generated")
    log10xm1=math.log10(1/x)
    max_x = A_xmax_interp[ibeam](log10xm1)
    max_fun = A_fmax_interp[ibeam](log10xm1)
    logQ2oQ02now=logQ2oQ02min
    n_rand = 0
    while True:
        r1=random.random() # a random float number between 0 and 1
        logQ2oQ02now=(logQ2oQ02max-logQ2oQ02min)*r1+logQ2oQ02min
        w=distfun(x,logQ2oQ02now)/max_fun
        r2=random.random() # a random float number between 0 and 1
        if r2 <= w: break
        n_rand+=1
        if n_rand == int(1e6):
            tqdm.tqdm.write("WARNING: It's maybe impossible to find a correct answer")
            tqdm.tqdm.write("logQ2oQ02max,logQ2oQ02min,w =" ,logQ2oQ02max,logQ2oQ02min,w)
        if n_rand == int(1e7):
            tqdm.tqdm.write("ERROR: It's impossible")
    Q2v=math.exp(logQ2oQ02now)*Q02
    return Q2v

#stream=open("Q2.dat",'w')
#for i in range(100000):
#    Q2v=generate_Q2_epa_ion(1,1e-1,1.0,WoodsSaxon['Pb208'][0],\
#                                WoodsSaxon['Pb208'][1],WoodsSaxon['Pb208'][2])
#    stream.write('%12.7e\n'%Q2v)
#stream.close()

def boostl(Q,PBOO,P):
    """Boost P via PBOO with PBOO^2=Q^2 to PLB"""
    # it boosts P from (Q,0,0,0) to PBOO
    # if P=(PBOO[0],-PBOO[1],-PBOO[2],-PBOO[3])
    # it will boost P to (Q,0,0,0)
    PLB=[0,0,0,0] # energy, px, py, pz in unit of GeV
    PLB[0]=(PBOO[0]*P[0]+PBOO[3]*P[3]+PBOO[2]*P[2]+PBOO[1]*P[1])/Q
    FACT=(PLB[0]+P[0])/(Q+PBOO[0])
    for j in range(1,4):
        PLB[j]=P[j]+FACT*PBOO[j]
    return PLB

def boostl2(Q,PBOO1,PBOO2,P):
    """Boost P from PBOO1 (PBOO1^2=Q^2) to PBOO2 (PBOO2^2=Q^2) frame"""
    PBOO10=[PBOO1[0],-PBOO1[1],-PBOO1[2],-PBOO1[3]]
    PRES=boostl(Q,PBOO10,P) # PRES is in (Q,0,0,0) frame
    PLB=boostl(Q,PBOO2,PRES) # PLB is in PBOO2 frame
    return PLB

def boostToEcm(E1,E2,pext):
    Ecm=2*math.sqrt(E1*E2)
    PBOO=[E1+E2,0,0,E2-E1]
    pext2=copy.deepcopy(pext)
    for j in range(len(pext)):
        pext2[j]=boostl(Ecm,PBOO,pext[j])
    return pext2

def boostFromEcm(E1,E2,pext):
    Ecm=2*math.sqrt(E1*E2)
    PBOO=[E1+E2,0,0,E1-E2]
    pext2=copy.deepcopy(pext)
    for j in range(len(pext)):
        pext2[j]=boostl(Ecm,PBOO,pext[j])
    return pext2


def deltaPhi(pt1,pt2,phi1,phi_diff):
    phi2 = phi1 + phi_diff
    d = np.abs(np.arctan2(pt1*np.sin(phi1)+pt2*np.sin(phi2),pt1*np.cos(phi1)+pt2*np.cos(phi2)) - phi1)
    if d > np.pi:
        d = 2*np.pi - d
    return d

L_aim = []
L_real = []
err = []

def sufflePhi(pext2,X,w):
    g1,g2,l1,l2 = np.array(pext2)
    N = len(X)
    qt1 = pt(l1)
    qt2 = pt(l2)
    phi1 = phi(l1)
    phi2 = phi(l2)
    phi_diff = phi2-phi1
   
    phi_diffmin = np.min([0.8*phi_diff,1.2*phi_diff])
    phi_diffmax = np.max([0.8*phi_diff,1.2*phi_diff])
    qt1min = qt1*0.8
    qt1max = qt1*1.2
    qt2min = qt2*0.8
    qt2max = qt2*1.2
    dphi = deltaPhi(qt1,qt2,phi1,phi_diff)
    if dphi < np.pi/4:
        X,w = X[:N//4],w[:N//4]
    elif dphi < 2*np.pi/4:
        X,w = X[N//4:2*N//4],w[N//4:2*N//4]
    elif dphi < 3*np.pi/4:
        X,w = X[2*N//4:3*N//4],w[2*N//4:3*N//4]
    else:
        X,w = X[3*N//4:],w[3*N//4:]
    w = w/np.sum(w)
    
    dphi_choosen = np.random.choice(X,p=w)
    qt_pair2 = qt1**2 + qt2**2 + 2*qt1*qt2*np.cos(phi_diff)
    #L_aim.append(dphi_choosen)

    constrain = optimize.NonlinearConstraint(lambda x : np.abs((x**2+qt2**2+2*x*qt2*np.cos(phi_diff))-qt_pair2),0,0.1)
    res1 = optimize.minimize(lambda x : np.abs(deltaPhi(x,qt2,phi1,phi_diff)-dphi_choosen),qt1,bounds=[(qt1min,qt1max)],tol=1e-5,constraints=constrain)
    
    if res1.fun < 1e-1:
        qt1 = res1.x[0]
        dphi = deltaPhi(qt1,qt2,phi1,phi_diff)
        #L_real.append(dphi)
        #err.append(np.abs(dphi-dphi_choosen))
        return phi1,phi2,qt1,qt2
    else:
        constrain = optimize.NonlinearConstraint(lambda x : np.abs((qt1**2+x**2+2*qt1*x*np.cos(phi_diff))-qt_pair2),0,0.1)
        res2 = optimize.minimize(lambda x : np.abs(deltaPhi(qt1,x,phi1,phi_diff)-dphi_choosen),qt2,bounds=[(qt2min,qt2max)],tol=1e-5,constraints=constrain)
        
        qt2 = res2.x[0]
        dphi = deltaPhi(qt1,qt2,phi1,phi_diff)
        #L_real.append(dphi)
        #err.append(np.abs(dphi-dphi_choosen))
        return phi1,phi2,qt1,qt2

    

def rapidity(p):
    return 0.5*np.log((p[0]+p[-1])/(p[0]-p[-1]))

def pt(p):
    return np.sqrt(p[1]**2 + p[2]**2)

def phi(p):
    return np.arctan2(p[2],p[1])

def M(p):
    return np.sqrt(p[0]**2 - p[1]**2 - p[2]**2 - p[3]**2)
X_list = []
def phi_distribution(X,pext2,sqrt_s,PID_lepton,RA,aA,wA,Z):
    dict_mass = {11:0.000511,13:0.105658}
    g1,g2,l1,l2 = np.array(pext2)
    pair = g1+g2
    kt = 0.5*(l1-l2)
    Kt = pt(kt)
    ml = dict_mass[PID_lepton]
    mt = np.sqrt(ml**2 + kt[1]**2 + kt[2]**2)
    y1 = rapidity(l1)
    y2 = rapidity(l2)
    m_pair = M(pair)
    x1 = mt/sqrt_s*(np.exp(y1)+np.exp(y2))
    x2 = mt/sqrt_s*(np.exp(-y1)+np.exp(-y2))
    X_list.append(x1)
    X_list.append(x2)
    z = 1/(np.exp(y1-y2)+1)
    qt = pt(pair)
    A = c.A_gammagamma(x1,x2,qt,Kt,ml,m_pair,RA,aA,Z)
    B = c.B_gammagamma(x1,x2,qt,Kt,ml,m_pair,RA,aA,Z)
    C = c.C_gammagamma(x1,x2,qt,Kt,ml,m_pair,RA,aA,Z)
    if np.abs(B>A) >1 or np.abs(C/A) > 1:
        tqdm.tqdm.write(f"WARNING: B or C is larger than A for x1={x1},x2={x2},qt={qt},Kt={Kt},ml={ml},m_pair={m_pair},RA={RA},aA={aA},Z={Z}")
        C = C/10
    w = A + B*np.cos(2*X) + C*np.cos(4*X)
    return w


def InitialMomentumReshuffle(Ecm,x1,x2,Q1,Q2,pext,sqrt_s,PID_lepton,RA,aA,wA,Z):
    r1 = np.random.random()  # a random float number between 0 and 1
    r2 = np.random.random()  # a random float number between 0 and 1
    ph1 = 2 * math.pi * r1
    ph2 = 2 * math.pi * r2

    Kperp2 = Q1 ** 2 + Q2 ** 2 + 2 * Q1 * Q2 * math.cos(ph1 - ph2)
    Kperp2max = Ecm**2*(min(1,x1/x2,x2/x1)-x1*x2)
    if Kperp2 >= Kperp2max:
        return None
    x1bar=math.sqrt(x1/x2*Kperp2/Ecm**2+x1**2)
    x2bar=math.sqrt(x2/x1*Kperp2/Ecm**2+x2**2)
    if x1bar >= 1.0 or x2bar >= 1.0: return None
    pext2=copy.deepcopy(pext)
    # new initial state
    pext2[0][0]=Ecm/2*x1bar
    pext2[0][1]=Q1*math.cos(ph1)
    pext2[0][2]=Q1*math.sin(ph1)
    pext2[0][3]=Ecm/2*x1bar
    pext2[1][0]=Ecm/2*x2bar
    pext2[1][1]=Q2*math.cos(ph2)
    pext2[1][2]=Q2*math.sin(ph2)
    pext2[1][3]=-Ecm/2*x2bar
    # new final state
    PBOO1=[0,0,0,0]
    PBOO2=[0,0,0,0]
    for j in range(4):
        PBOO1[j]=pext[0][j]+pext[1][j]
        PBOO2[j]=pext2[0][j]+pext2[1][j]
    Q=math.sqrt(x1*x2)*Ecm
    for j in range(2,len(pext)):
        pext2[j]=boostl2(Q,PBOO1,PBOO2,pext[j])
        
    if azimuthalSmearing:
        X = np.linspace(0,np.pi,1000)
        prob = phi_distribution(X,pext2,sqrt_s,PID_lepton,RA,aA,wA,Z)
        w = prob/np.sum(prob)
        phi1,phi2,qt1,qt2 = sufflePhi(pext2,X,w)
        pext2[2] = [pext2[2][0],qt1*np.cos(phi1),qt1*np.sin(phi1),pext2[2][3]]
        pext2[3] = [pext2[3][0],qt2*np.cos(phi2),qt2*np.sin(phi2),pext2[3][3]]
        
    return pext2


headers=[]
inits=[]
events=[]

ninit0=0
ninit1=0
firstinit=""
E_beam1=0
E_beam2=0
PID_beam1=0
PID_beam2=0
nan_count = 0
nevent=0
ilil=0
for i,file in enumerate(files):
    N_event=0
    
    stream=open(file,'r')
    tqdm.tqdm.write("INFO: Counting the number of events in file")
    for line in stream:
        if "<event>" in line or "<event " in line: N_event +=1
    tqdm.tqdm.write(f"INFO: Number of events in file = {N_event}")
    pbar = tqdm.tqdm(total=N_event)
    stream.close()
    stream=open(file,'r')
    headQ=True
    initQ=False
    iinit=-1
    ievent=-1
    eventQ=False
    this_event=[]
    n_particles=0
    rwgtQ=False
    procid=None
    proc_dict={}
    tqdm.tqdm.write("INFO: Start processing the file")
    for line in stream:
        sline=line.replace('\n','')
        if "<init>" in line or "<init " in line:
            initQ=True
            headQ=False
            iinit=iinit+1
            if i==0: 
                inits.append(sline)
        elif headQ and i == 0:
            headers.append(sline)
        elif "</init>" in line or "</init " in line:
            initQ=False
            iinit=-1
            if i==0: 
                inits.append(sline)
        elif initQ:
            iinit=iinit+1
            if "<generator name=" in line:
                inits.append(sline)
            elif iinit == 1:
                if i == 0:
                    firstinit=sline
                    ninit0=len(inits)
                    inits.append(sline)
                    firstinit=firstinit.rsplit(' ',1)[0]
                    ff=firstinit.strip().split()
                    PID_beam1=int(ff[0])
                    PID_beam2=int(ff[1])
                    E_beam1=float(ff[2])
                    E_beam2=float(ff[3])
                    if abs(PID_beam1) != 2212 or abs(PID_beam2) != 2212:
                        tqdm.tqdm.write("Not a proton-proton collider")
                        raise ValueError
                    ninit1=int(sline.rsplit(' ',1)[-1])
                else:
                    ninit1=ninit1+int(sline.rsplit(' ',1)[-1])
                    sline=sline.rsplit(' ',1)[0]
                    if not sline == firstinit:
                        tqdm.tqdm.write("the beam information of the LHE files is not identical")
                        raise Exception
            elif iinit >= 2:
                procid=sline.split()[-1]
                procpos=sline.index(' '+procid)
                ilil=ilil+1
                sline=sline[:procpos]+(' %d'%(offset+ilil))
                proc_dict[procid]=offset+ilil
                if i == 0:
                    inits.append(sline)
                else:
                    inits.insert(-1,sline)
            else:
                tqdm.tqdm.write("should not reach here. Do not understand the <init> block")
                raise Exception
        elif "<event>" in line or "<event " in line:
            eventQ=True
            ievent=ievent+1
            events.append(sline)
        elif "</event>" in line or "</event " in line:
            nevent=nevent+1
            eventQ=False
            rwgtQ=False
            ievent=-1
            this_event=[]
            n_particles=0
            events.append(sline)
        elif eventQ:
            ievent=ievent+1
            if ievent == 1:
                found=False
                for procid,new_procid in proc_dict.items():
                    if ' '+procid+' ' not in sline: continue
                    procpos=sline.index(' '+procid+' ')
                    found=True
                    sline=sline[:procpos]+(' %d'%(new_procid))+sline[procpos+len(' '+procid):]
                    break
                if not found:
                    tqdm.tqdm.write("do not find the correct proc id !")
                    raise Exception
                n_particles=int(sline.split()[0])
                #procpos=sline.index(' '+procid)
                #sline=sline[:procpos]+(' %d'%(1+i))+sline[procpos+len(' '+procid):]
            elif "<mgrwt" in sline:
                rwgtQ=True
            elif "</mgrwt" in sline:
                rwgtQ=False
            elif not rwgtQ:
                sline2=sline.split()
                particle=[int(sline2[0]),int(sline2[1]),int(sline2[2]),int(sline2[3]),\
                              int(sline2[4]),int(sline2[5]),float(sline2[6]),float(sline2[7]),\
                              float(sline2[8]),float(sline2[9]),float(sline2[10]),\
                              float(sline2[11]),float(sline2[12])]
                this_event.append(particle)
                if ievent == n_particles+1:
                    # get the momenta and masses
                    x1=this_event[0][9]/E_beam1
                    x2=this_event[1][9]/E_beam2
                    if math.isnan(x1) or math.isnan(x2):
                        if nan_count < 5:
                            tqdm.tqdm.write("Warning: x1 or x2 is nan")
                        if nan_count == 5:
                            tqdm.tqdm.write("Other warning will be suppressed")
                        nan_count = nan_count + 1
                        continue
                    pext=[]
                    mass=[]
                    for j in range(n_particles):
                        pext.append([this_event[j][9],this_event[j][6],\
                                         this_event[j][7],this_event[j][8]])
                        mass.append(this_event[j][10])
                    PID_lepton = np.abs(this_event[-1][0])
                    
                    # first we need to boost from antisymmetric beams to symmetric beams
                    if E_beam1 != E_beam2:
                        pext=boostToEcm(E_beam1,E_beam2,pext)
                    Ecm=2*np.sqrt(E_beam1*E_beam2)
                    sqrt_s = Ecm/np.sqrt((float(nuclei[0][2:])*float(nuclei[1][2:])))
                    pext_new = None
                    Q1=0
                    Q2=0
                    while pext_new == None:
                        # generate Q1 and Q2
                        if nuclei[0] == 'p':
                            Q12=generate_Q2_epa_proton(x1,Q2max)
                        else:
                            RA,aA,wA,_=WoodsSaxon[nuclei[0]]
                            Q12=generate_Q2_epa_ion(0,x1,Q2max,RA,aA,wA)
                        if nuclei[1] == 'p':
                            Q22=generate_Q2_epa_proton(x2,Q2max)
                        else:
                            if nuclei[0] == nuclei[1]:
                                RA,aA,wA,Z=WoodsSaxon[nuclei[0]]
                                Q22=generate_Q2_epa_ion(0,x2,Q2max,RA,aA,wA)
                            else:
                                RA,aA,wA,Z=WoodsSaxon[nuclei[1]]
                                Q22=generate_Q2_epa_ion(1,x2,Q2max,RA,aA,wA)
                        Q1=np.sqrt(Q12)
                        Q2=np.sqrt(Q22)
                        # perform the initial momentum reshuffling
                        pext_new=InitialMomentumReshuffle(Ecm,x1,x2,Q1,Q2,pext,sqrt_s,PID_lepton,RA,aA,wA,Z)
                    if E_beam1 != E_beam2:
                        # boost back from the symmetric beams to antisymmetric beams
                        pext_new=boostFromEcm(E_beam1,E_beam2,pext_new)
                    # update the event information
                    # negative invariant mass means negative invariant mass square (-Q**2, spacelike)
                    this_event[0][10]=-Q1
                    this_event[1][10]=-Q2
                    for j in range(n_particles):
                        this_event[j][9]=pext_new[j][0]
                        this_event[j][6]=pext_new[j][1]
                        this_event[j][7]=pext_new[j][2]
                        this_event[j][8]=pext_new[j][3]
                        newsline="      %d    %d     %d    %d    %d    %d  %12.7e  %12.7e  %12.7e  %12.7e  %12.7e  %12.7e  %12.7e"%tuple(this_event[j])
                        events.append(newsline)
                continue
            events.append(sline)
        if "<event>" in line or "<event" in line:pbar.update(1)
    stream.close()
    pbar.close()

# modify the number of process information
firstinit=firstinit+(' %d'%ninit1)
inits[ninit0]=firstinit

text='\n'.join(headers)+'\n'
text=text+'\n'.join(inits)+'\n'
text=text+'\n'.join(events)
text=text+'\n</LesHouchesEvents>'
stream=open(outfile,'w')
stream.write(text)
stream.close()
tqdm.tqdm.write(f"INFO: The final produced lhe file is {outfile}")
if nan_count > 0:
    tqdm.tqdm.write(f"INFO: The ratio of nan is {nan_count/nevent} for {nevent} events")
if nan_count/nevent > 0.01:
    tqdm.tqdm.write("WARNING: The ratio of nan is too large, please check the input lhe files")

if azimuthalSmearing:
    """ plt.hist(L_aim,bins=100,alpha=0.5,label='aim')
    plt.hist(L_real,bins=100,alpha=0.5,label='real')
    plt.hist(err,bins=100,alpha=0.5,label='err') """
    bins = np.logspace(-8,-1,100)
    plt.hist(X_list,bins=bins)
    plt.legend()
    plt.show()
    print(len([e for e in err if e > 0.1]))
