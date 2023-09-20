"""

This module defines functions for converting between the ECEF and ECI coordinate frames and uses IERS nutation/Earth orientation parameter (EOP) data.

The main functions to pay attention to are ecef_eci to provide parameters for the coordinate conversion.
This is separated from the actual coordinate conversion functions because the ecef_eci parameters only need to be calculated once per given time,
and so may be reused for multiple conversions at shared time steps.
Coordinate conversions ecef2eci and eci2ecef use the output of ecef_eci to perform the coordinate frame conversion.

Additionally find functions for determining sun and moon positions in the ECI frame at given times.

Some functions are compiled with Numba for improved runtimes, but not all functions in this module are compatible with Numba due to the need to call other functions / load text files and other IERS data.

Algorithms are taken from the Vallado textbook.
Values use SI base units unless otherwise specified.

"""

import numpy as np
from numpy import cos, sin, pi
from numpy.linalg import norm
from numba import jit

import pandas as pd

from constants import J2000, MJD, dUTC_TAI, DEG2RAD, ARCSEC2RAD, MILLIARCSEC2RAD, AU, Rearth

global nut  # load nutation data from IERS
nut = np.loadtxt("./code_data_files/nut80.dat")
global EOP_data  # load EOP data from IERS
EOP_data = pd.read_csv('./code_data_files/finals.all.csv',delimiter=';')
EOP_data.set_index('MJD',inplace=True)

# Define x, y, and z 3D rotation matrices
def rot1(angle): return np.array([[1,0,0],[0,cos(angle),sin(angle)],[0,-sin(angle),cos(angle)]])
def rot2(angle): return np.array([[cos(angle),0,-sin(angle)],[0,1,0],[sin(angle),0,cos(angle)]])
def rot3(angle): return np.array([[cos(angle),sin(angle),0],[-sin(angle),cos(angle),0],[0,0,1]])

def polar_motion(date_utc):
    # Load EOP data from finals.all.csv
    # Accessed via https://www.iers.org/IERS/EN/DataProducts/EarthOrientationData/eop.html
    date_MJD = int(date_utc - MJD)
    
    xp = EOP_data.at[date_MJD,'x_pole'] * ARCSEC2RAD
    yp = EOP_data.at[date_MJD,'x_pole'] * ARCSEC2RAD
    dUT1 = EOP_data.at[date_MJD,'UT1-UTC']
    LOD = EOP_data.at[date_MJD,'LOD']  # ms
    LOD_s = LOD * 1e-3  # s
    return xp, yp, dUT1, LOD_s
    # TODO : interpolate between days, possibly with a 3rd order polynomial

def nutations(date_utc):
    """Calculates nutations in longitude and latitude based on the 1980 IAU Theory of Nutation"""
    # See Vallado ALGORITHM 16: ConvTime
    TAI = date_utc + dUTC_TAI/86400
    TT = TAI + 32.184/86400  # expressed in days
    TTT = (TT-J2000)/36525
    TTT2 = TTT**2
    TTT3 = TTT**3

    # Eqn 3-81 of vallado
    bareps = (84381.448 - 46.8150*TTT - 0.00059*TTT2 + 0.001813*TTT3) * ARCSEC2RAD

    # Eqn 3-82 of Vallado, fixed in Errata V4: https://celestrak.org/software/vallado/ErrataVer4.pdf
    r = 360  # [deg]
    Mmoon = (134.96298139 + (1325*r + 198.8673981)*TTT + 0.0086972*TTT2 + 1.78e-5*TTT3)  # [deg]
    Msun = (357.52772333 + (99*r + 359.0503400)*TTT - 0.0001603*TTT2 - 3.3e-6*TTT3)  # [deg]
    uMmoon = (93.27191028 + (1342*r + 82.0175381)*TTT - 0.0036825*TTT2 + 3.1e-6*TTT3)  # [deg]
    dSun = (297.85036306 + (1236*r + 307.1114800)*TTT - 0.0019142*TTT2 + 5.3e-6*TTT3)  # [deg]
    Omegamoon = (125.04452222 - (5*r + 134.1362608)*TTT + 0.0020708*TTT2 + 2.2e-6*TTT3)  # [deg]
    Delaunay = np.array([Mmoon, Msun, uMmoon, dSun, Omegamoon])
    Delaunay = np.mod(Delaunay, 360)
    
    api = np.einsum('i,ji->j', Delaunay, nut[:,:5]) * DEG2RAD
    
    # Units of .0001" per Julian century
    Ai = nut[:,5]
    Bi = nut[:,6]
    Ci = nut[:,7]
    Di = nut[:,8]

    dPsi = np.dot(Ai + Bi*TTT, sin(api)) * ARCSEC2RAD/10000
    dEps = np.dot(Ci + Di*TTT, cos(api)) * ARCSEC2RAD/10000
    
    eps = bareps + dEps
    
    return dPsi, dEps, eps, bareps, Delaunay

def precession(date_utc, return_extras=False):
    # Born Appendix H.2 and/or Vallado Eqn. 3-88
    TAI = date_utc + dUTC_TAI/86400
    TT = TAI + 32.184/86400  # expressed in days
    TTT = (TT-J2000)/36525
    zetaA = (2306.2181*TTT + 0.30188*TTT**2 + 0.017998*TTT**3) * ARCSEC2RAD  # [rad]
    thetaA = (2004.3109*TTT - 0.42665*TTT**2 - 0.041833*TTT**3) * ARCSEC2RAD  # [rad]
    zA = (2306.2181*TTT + 1.09468*TTT**2 + 0.018203*TTT**3) * ARCSEC2RAD  # [rad]

    if return_extras:
        return zetaA,thetaA,zA,TTT
    return zetaA, thetaA, zA

def ecef_eci(date_utc, return_extras=False):
    """Return a rotation matrix going from from ECEF to ECI at the given time"""
    xp, yp, dUT1, LOD = polar_motion(date_utc)
    dPsi, dEps, eps, bareps, Delaunay = nutations(date_utc)
    zeta, theta, z = precession(date_utc)
    
    UT1 = date_utc + dUT1/86400
    T_UT1 = (int(UT1) - J2000)/36525
    
    # Vallado Eqn. 3-45
    theta_GMST0 = 24110.54841 + 8640184.812866*T_UT1 + 0.093104*T_UT1**2 - 6.2e-6*T_UT1**3

    theta_GMST0 = np.mod(theta_GMST0, 86400) / 240
    # omega_prec = 2*pi * (1.002737909350795 + 5.9006e-11*T_UT1 - 5.9e-15*T_UT1**2)
    omega_prec = 7.292115146706979e-5 * 86400 * (1 - LOD / 86400)
    
    # Vallado Eqn. 3-46
    theta_GMST1982 = theta_GMST0*DEG2RAD + omega_prec * (UT1 - int(UT1)) - pi
    # TODO : add LOD to omega prec and omega vec calculations

    # Vallado Eqn. 3-79
    Eq_equinox1982 = dPsi*cos(bareps) + 0.00264*ARCSEC2RAD*sin(Delaunay[4]*DEG2RAD) + 0.000063*ARCSEC2RAD*sin(2*Delaunay[4]*DEG2RAD)
    theta_GAST1982 = theta_GMST1982 + Eq_equinox1982
    
    P = rot3(zeta) @ rot2(-theta) @ rot3(z)  # MOD to ECI
    N = rot1(-bareps) @ rot3(dPsi) @ rot1(eps)  # TOD to MOD
    R = rot3(-theta_GAST1982)  # PEF to TOD
    W = np.array([[1, 0, -xp],[0, 1, yp],[xp, -yp, 1]])  # ECEF to PEF

    # omega_prec * (1-LOD/86400) # Check Vallado
    omega_vec = np.array([0, 0, omega_prec])/86400 # * (1 - LOD / 86400)])/86400  # is this correct? or is it omega_vec = W @ omega_vec ? is it negative?
    # vTOD = rot3(-theta_GAST1982) vPEF + np.cross(omega_vec, rPEF)  # Vallado Eqn. 3-80
    
    if return_extras:
        PN = P @ N
        T_UT1 = (UT1-J2000)/36525
        return PN @ R @ W, PN, T_UT1, dEps, omega_vec, R, W  # TODO: int(), ie use a JD, or not int? (for T_UT1: (UT1 - J2000)/36525?)
    return P @ N @ R @ W

@jit
def eci2ecef(rECI, vECI, PNRW, PN, T_UT1, dEps, omega_vec, R, W):
    """Convert postion and velocity from ECI to ECEF.
    Position rECI should be given as a 3-array in x, y, and z. [m]
    Velocity vECI should be given as a 3-array in x, y, and z. [m/s]
    Other args are the output of ecef_eci(date_utc, return_extras=True)."""
    # Vallado pg 229 #2
    PNR = PN @ R  # ECI to PEF
    rPEF = PNR.T @ rECI
    rECEF = PNRW.T @ rECI
    vECEF = W.T @ (PNR.T @ vECI - np.cross(omega_vec, rPEF))

    return rECEF, vECEF

@jit
def ecef2eci(rECEF, vECEF, PNRW, PN, T_UT1, dEps, omega_vec, R, W):
    """Convert postion and velocity from ECEF to ECI.
    Position rECEF should be given as a 3-array in x, y, and z. [m]
    Velocity vECEF should be given as a 3-array in x, y, and z. [m/s]
    Other args are the output of ecef_eci(date_utc, return_extras=True)."""
    # Vallado pg 229 #1
    PNR = PN @ R
    rPEF = W @ rECEF.astype(np.float64)
    rECI = PNRW @ rECEF.astype(np.float64)
    vECI = PNR @ (W @ vECEF + np.cross(omega_vec, rPEF))

    return rECI, vECI    

@jit
def sun_vector(T_UT1, PN):
    """Return the sun vector in meters in the ECI frame. See Vallado for derivation"""
    eE = 0.016708617  # Earth's eccentricity about the sun
    lambdaMsun = np.mod(280.460 + 36000.771*T_UT1, 360)  # deg
    Msun = np.mod(357.5291092 + 35999.05034*T_UT1, 360)  # deg
    # vSun = Msun + 2*eE*sin(Msun*DEG2RAD) + 5/4*eE**2*sin(2*Msun*DEG2RAD)  # deg
    
    lambdaEcliptic = lambdaMsun + 1.914666471*sin(Msun*DEG2RAD) + 0.019994643*sin(2*Msun*DEG2RAD)  # deg
    epsilon = 23.439291 - 0.0130042*T_UT1  # deg
    
    lambdaEcliptic = lambdaEcliptic * DEG2RAD
    epsilon = epsilon * DEG2RAD
    
    dist = 1.00040612 - eE*cos(Msun*DEG2RAD) - 0.000139589*cos(2*Msun*DEG2RAD)  # AU
    # dist = dist  # m
    
    rsun = dist * np.array([
        cos(lambdaEcliptic),
        cos(epsilon) * sin(lambdaEcliptic),
        sin(epsilon) * sin(lambdaEcliptic)
    ])  # AU
    
    # A direct comparison is possible only after converting the MOD vector to the ICRS (GCRF)
    # However, using an approximate relation (TTT ≈ TUT1) and accounting only for precession,
    # the difference is about 28873 km, or 0.0002 AU.
    
    # otherwise use P @ N @ rsun to convert TOD (see errata) to ECI
    return (PN @ rsun) * AU  # m

@jit
def moon_vector(T_UT1, PN, dEps):
   """Return the moon vector in meters in the ECI frame. See Vallado for derivation"""
    # TODO: use TDB instead of UT1 for higher accuracy
    T_TDB = T_UT1  # approximately

    lambdaEcliptic = (
        218.32
        + 481267.8813 * T_TDB
        + 6.29 * sin((134.9 + 477198.85 * T_TDB) * DEG2RAD)
        - 1.27 * sin((259.2 - 413335.38 * T_TDB) * DEG2RAD)
        + 0.66 * sin((235.7 + 890534.23 * T_TDB) * DEG2RAD)
        + 0.21 * sin((269.9 + 954397.70 * T_TDB) * DEG2RAD)
        - 0.19 * sin((357.5 + 35999.05 * T_TDB) * DEG2RAD)
        - 0.11 * sin((186.6 + 966404.05 * T_TDB) * DEG2RAD)
    ) * DEG2RAD

    phiEcliptic = (
        5.13 * sin((93.3 + 483202.03 + T_TDB) * DEG2RAD)
        + 0.28 * sin((228.2 + 960400.87 * T_TDB) * DEG2RAD)
        - 0.28 * sin((318.3 + 6003.18 * T_TDB) * DEG2RAD)
        - 0.17 * sin((217.6 - 407332.20 * T_TDB) * DEG2RAD)
    ) * DEG2RAD

    fancyP = (
        0.9508
        + 0.0518 * cos((134.9 + 477198.85 * T_TDB) * DEG2RAD)
        + 0.0095 * cos((259.2 - 413335.38 * T_TDB) * DEG2RAD)
        + 0.0078 * cos((235.7 + 890534.23 * T_TDB) * DEG2RAD)
        + 0.0028 * cos((269.9 + 954397.70 * T_TDB) * DEG2RAD)
    ) * DEG2RAD

    barEps = (
        23.439291
        - 0.0130042 * T_TDB
        - 1.64e-7 * T_TDB**2
        + 5.04e-7 * T_TDB**3
    ) * DEG2RAD

    eps = barEps + dEps

    dist = Rearth / sin(fancyP)

    rmoon = dist * np.array([
        cos(phiEcliptic) * cos(lambdaEcliptic),
        cos(eps) * cos(phiEcliptic) * sin(lambdaEcliptic) - sin(eps) * sin(phiEcliptic),
        sin(eps) * cos(phiEcliptic) * sin(lambdaEcliptic) + cos(eps) * sin(phiEcliptic)
    ])

    return PN @ rmoon

