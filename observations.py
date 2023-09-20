"""

This module provides functions for observations of satellites/debris.

range_and_rate gives the radar range and range-rate for a ground-based radar station observing a satellite.
radar_vis determines whether a satellite is above the horizon of a given radar station.

sat_vis determines the SNR and whether a star tracker can observe a given object.
alt_and_dec gives the angular location of the object.
ang_rate determines the magnitude of the angular rate.

"""

import numpy as np
from numpy.linalg import norm
import scipy.integrate
from numba import jit

from constants import c, s_1, s_2, s_3, DEG2RAD
from coordinates import ecef_eci, eci2ecef, ecef2eci, sun_vector, moon_vector
from force_models import dynamics, shadow


def dyn_wrapper(t,s):
    """Call the dynamics function specifying to use an EGM96 gravity model, a cannonball model for SRP and drag, and taking into account lunisolar perturbations.
    This is used to determine light time corrections for radar range and range-rate.
    """
    return dynamics(t, s, [1,'cannonball','cannonball',1,1])

def range_and_rate(t, sECI, station_state, return_extras=False, using_ECI=False):
    """Calculate range and range-rate for a ground-based radar station at position station_state observing an object with state sECI at time t,
    incorporating light time correction.
    Assumes station_state is a 3-array of ECEF position unless specifying using_ECI=True, in which case station_state should be a 6-array of ECI position and velocity.
    """
    JD = t/86400
    # station ECI position and velocity at the given measurement time stays the same
    if using_ECI:
        r_statECI, v_statECI = station_state[:3], station_state[3:6]
    else:
        r_statECI, v_statECI = ecef2eci(station_state, np.zeros(3), *ecef_eci(JD, return_extras=True))

    # get an initial range estimate
    range_ = norm(sECI[:3] - r_statECI)

    # iterate over light time corrections to get the light time corrected satellite state.
    # this could be improved with an actual fsolve rather than an iteration, but this was a quick and dirty solution,
    # and in almost all cases it only takes 1-2 iterations to converge.
    slt_guess = sECI
    delta = np.inf
    while delta > 0.001:  # iterate until converged to ~1 mm position difference
        lt = range_/c  # one way light travel time
        t_lt = t - lt  # light-corrected time

	# propagate to the state at the light-corrected time
        s_lt = scipy.integrate.solve_ivp(
            dyn_wrapper,
            t_span = [t, t_lt],
            t_eval = [t, t_lt],
            y0 = sECI[:6]
        ).y[:,-1]

        delta = norm(s_lt[:3] - slt_guess[:3])  # change in position from this new position with the previous guess of the position
        range_ = norm(s_lt[:3] - r_statECI)  # new range
        slt_guess = s_lt  # update our guess of the position at the light corrected time
    v_rel = slt_guess[3:6] - v_statECI  # relative velocity between the object and station
    range_rate = np.dot(v_rel, slt_guess[:3]-r_statECI)/range_
    if return_extras:  # for debugging
        return np.array([range_, range_rate]), np.concatenate([r_statECI, v_statECI])
    return np.array([range_, range_rate])

def range_and_rate_old(t, sECI, station_id):
    """Old function to calculate range and range-rate for a radar station observing an object at state sECI"""
    if station_id == 1:
        stationECEF = s_1
    elif station_id == 2:
        stationECEF = s_2
    elif station_id == 3:
        stationECEF = s_3
    else:
        print("inappropriate station ID")
        raise Exception

    JD = t/86400
    rECEF, vECEF = eci2ecef(sECI[0:3], sECI[3:6], *ecef_eci(JD, return_extras=True))
    range_ = norm(rECEF - stationECEF)

    delta = np.inf
    while delta > 0.01:  # 0.001 m = 1 mm
        lt = range_/c
        t_lt = t - lt  # s
        JD_lt = JD - lt/86400  # s
        s_lt = scipy.integrate.solve_ivp(
            dyn_wrapper,
            t_span = [t, t_lt],
            t_eval = [t, t_lt],
            y0 = sECI[0:6]
        ).y[:,-1]
        rECEF_new, vECEF_new = eci2ecef(s_lt[0:3], s_lt[3:6], *ecef_eci(JD_lt, return_extras=True))
        delta = norm(rECEF_new - rECEF)
        rECEF = rECEF_new
        vECEF = vECEF_new
        range_ = norm(rECEF - stationECEF)
    range_rate = np.dot(vECEF,(rECEF - stationECEF))/range_
    # TODO : range rate calculation in ECI frame
    # range_rate = dot(vECI,(rECI - stationECI))/range
    return np.array([range_, range_rate])

# @jit  # just-in-time compiling does not decrease runtime for this function
def radar_vis(t, sECI, stationECEF, ecef_eci_quants):
    """Determine whether an object with ECI state sECI can be observed by ground-based radar station at ECEF position stationECEF.
    ecef_eci_quants are needed to convert from the ECEF frame to ECI.
    
    Returns:
        (Bool): whether the object is above the horizon at the radar station
    """
    # station ECI position and velocity at the given measurement time stays the same
    r_statECI, v_statECI = ecef2eci(stationECEF, np.zeros(3), *ecef_eci_quants)
    stat_to_sat = sECI[:3] - r_statECI
    stat_to_sat /= norm(stat_to_sat)
    stat = r_statECI / norm(r_statECI)
    return np.dot(stat, stat_to_sat) > np.cos(90 * DEG2RAD)

@jit
def sat_vis(t, sECI, obsECI, sun, moon):
    """Determine visibility constraints at time t for an observer with ECI state obsECI looking at an object with state sECI.
    Include sun and moon vectors as well.
    
    Returns:
    	(Bool): True or False, depending on whether the object can be observed
    	(float): 0 or SNR, if the object can be observed, where SNR is the signal-to-noise ratio
    """
    FOV_deg = 10
    FOV_half = FOV_deg/2

    # If the spacecraft is eclipsed, it is not visible
    eclipse = shadow(sECI[:3],sun)
    if eclipse == 0 or eclipse == 0.5:
        return False,0

    intrack = obsECI[3:6]/norm(obsECI[3:6])
    radial = obsECI[:3]/norm(obsECI[:3])

    obs_to_sat = sECI[:3] - obsECI[:3]
    dist = norm(obs_to_sat)
    obs_to_sat /= dist

    # If the spacecraft has low angular separation with the sun or moon, it cannot be observed
    if np.dot(obs_to_sat, sun) > np.cos(40 * DEG2RAD):
        return False,0
    if np.dot(obs_to_sat, moon) > np.cos(40 * DEG2RAD):
        return False,0

    # Determine whether the spacecraft is within the star tracker FOV
    radial_ang = np.dot(obs_to_sat, radial)
    track_ang = np.dot(obs_to_sat, intrack)
    if radial_ang > np.cos((90-FOV_half) * DEG2RAD) or radial_ang < np.cos((90+FOV_half) * DEG2RAD):
        return False,0
    if (track_ang > np.cos((45-FOV_half) * DEG2RAD)) or (track_ang < np.cos((45+FOV_half) * DEG2RAD) and track_ang > np.cos((135-FOV_half) * DEG2RAD)) or (track_ang < np.cos((135+FOV_half) * DEG2RAD)):
        return False,0

    SNR = 4000 * (1000 * 1e3 / dist) * (1/np.maximum(1, 180/np.pi * ang_rate(sECI, obsECI)))
    # Set an SNR threshold of 10 for observability
    if SNR < 10:
        return False,0

    return True, SNR

@jit
def alt_and_dec(s_objECI, s_obsECI):
    """Calculate altitude and declination angles for observer with state s_obsECI looking at object with state s_objECI"""
    obs2obj = s_objECI[:3] - s_obsECI[:3]
    dist = norm(obs2obj)
    u_obs2obj = obs2obj/dist
    theta = np.arccos(u_obs2obj[2])
    phi = np.arctan2(u_obs2obj[1], u_obs2obj[0])
    return theta, phi  # rad

@jit
def ang_rate(s_objECI, s_obsECI):
    """Calculate angular rate magnitude for observer with state s_obsECI looking at object with state s_objECI"""
    obs2obj = s_objECI[:3] - s_obsECI[:3]
    dist = norm(obs2obj)
    u_obs2obj = obs2obj/dist
    v_rel = s_objECI[3:6] - s_obsECI[3:6]
    v_perp = v_rel - u_obs2obj * np.dot(v_rel, u_obs2obj)
    v_ang = v_perp / dist  # rad/s
    return norm(v_ang)  # rad/s
