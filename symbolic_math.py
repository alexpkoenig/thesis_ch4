"""

This module implements analytical evaluations of Jacobians used in the EKF.

Rather than re-deriving the analytical Jacobian during each evaluation,
the analytical versions are computed once with SymPy via symbolic_math_writer.py and are simply loaded within this module for evaluation.

Jacobian names map to those defined in Chapter 4 of the thesis.

"""

import numpy as np
import sympy as sp
import dill
from numba import jit  # not used; did not improve runtime

import constants
from coordinates import ecef_eci


# Load the lambdified expressions from disk
with open("./code_data_files/jacobian_func.pkl", "rb") as f:
    Jacobian_func = dill.load(f)
with open("./code_data_files/htilde_func.pkl", "rb") as f:
    Htilde_func = dill.load(f)
with open("./code_data_files/hrtilde_func.pkl", "rb") as f:
    Hrtilde_func = dill.load(f)
with open("./code_data_files/hotilde_func.pkl", "rb") as f:
    Hotilde_func = dill.load(f)

def evaluateA(s, sun_vec, moon_vec):
    A = Jacobian_func(
        constants.muE, constants.muS, constants.muM,
        constants.J2, constants.J3, constants.J4,
        constants.Rearth,
        constants.rho0, constants.r0, constants.H, constants.Cd,
        constants.pSRP,
        constants.mass, constants.xface['area'],
        s[0], s[1], s[2],
        s[3], s[4], s[5],
        sun_vec[0], sun_vec[1], sun_vec[2],
        moon_vec[0], moon_vec[1], moon_vec[2])
        
    return A.astype(np.float64)

def evaluateH(sECI, epoch, station_ECEF):
    sECI = np.concatenate([sECI, np.ones(1) * constants.Cd])
    PNRW, PN, T_UT1, dEps, omega_vec, R, W = ecef_eci(epoch, return_extras=True)
    PNR = PN @ R

    W_num = np.array(W)
    PNR_num = np.array(PNR)

    # Evaluate the lambda function
    H = Htilde_func(
        sECI[0], sECI[1], sECI[2],
        sECI[3], sECI[4], sECI[5],
        station_ECEF[0], station_ECEF[1], station_ECEF[2],
        W_num, PNR_num, omega_vec[0], omega_vec[1], omega_vec[2])

    return H.astype(np.float64)

def evaluateHr(sECI, stationECI):
    sECI = np.concatenate([sECI, np.ones(1) * constants.Cd])

    H = Hrtilde_func(
        sECI[0], sECI[1], sECI[2],
        sECI[3], sECI[4], sECI[5],
        stationECI[0], stationECI[1], stationECI[2],
        stationECI[3], stationECI[4], stationECI[5],
    )

    return H.astype(np.float64)

def evaluateHo(sECI, stationECI):
    H = Hotilde_func(
        sECI[0], sECI[1], sECI[2],
        sECI[3], sECI[4], sECI[5],
        stationECI[0], stationECI[1], stationECI[2],
        stationECI[3], stationECI[4], stationECI[5],
    )

    return H.astype(np.float64)
