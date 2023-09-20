"""

This module implements the dynamical model for orbiting objects, including force models for:

EGM-96 20x20 gravity field
Cannonball/facet-based drag
Cannonball/facet-based solar pressure, including eclipse states
Solar gravitational perturbation
Lunar gravitational perturbation

All values are represented in SI base units.

"""


import numpy as np
from numpy import cos, sin, tan, arccos, arctan2
from numpy.linalg import norm
import scipy.special
from numba import jit, njit

from constants import *
from coordinates import ecef_eci, sun_vector, moon_vector, eci2ecef
from egm96 import grav_odp, egm96
from symbolic_math import evaluateA

def dynamics_full(t, s, *args):
    """Xdot = F(t,X), along with the state transition matrix
    
    """
    # determine which accelerations/perturbations to take into account. same order as returned by accelerations().
    fflags = np.ones(5)
    alt_grav = fflags[0]
    alt_drag = fflags[1]
    alt_srp = fflags[2]
    # determine whether to use different force models when calculating acceleration, or whether to turn any perturbations off entirely.
    if args != ():
        fflags = args[0]
        if fflags[0] is not None:
            alt_grav = fflags[0]
            fflags[0] = 1
        if fflags[1] is not None:
            alt_drag = fflags[1]
            fflags[1] = 1
        if fflags[2] is not None:
            alt_srp = fflags[2]
            fflags[2] = 1

    # dynamics use SI base units (m, s, ...)
    state = s[0:7]  # object state
    phi = s[7:].reshape((7,7))  # state transition matrix
    ds = np.zeros(7+49)  # change in state and state transition matrix (flattened)
    
    # The derivative of position is the current velocity
    ds[:3] = s[3:6]
    
    # Calculate accelerations. this function also returns sun and moon positions.
    accels = accelerations(t,state[:6], use_alt_grav=alt_grav, use_alt_drag=alt_drag, use_alt_srp=alt_srp, return_extras=True)
    rsun = accels[5]
    rmoon = accels[6]

    # The derivative velocity is the sum of the gravitational acceleration and perturbations
    ds[3:6] = sum([accels[i]*fflags[i] for i in range(5)])

    # Update the state transition matrix
    A = evaluateA(state, rsun, rmoon)
    dphi = A @ phi
    ds[7:] = dphi.flatten()

    return ds


def dynamics(t, s, *args):
    """Xdot = F(t,X). Does not calculate the state transition matrix"""
    # determine which accelerations/perturbations to take into account. same order as returned by accelerations().
    fflags = np.ones(5)
    alt_grav = fflags[0]
    alt_drag = fflags[1]
    alt_srp = fflags[2]
    # determine whether to use different force models when calculating acceleration, or whether to turn any perturbations off entirely.
    if args != ():
        fflags = args[0]
        if fflags[0] is not None:
            alt_grav = fflags[0]
            fflags[0] = 1
        if fflags[1] is not None:
            alt_drag = fflags[1]
            fflags[1] = 1
        if fflags[2] is not None:
            alt_srp = fflags[2]
            fflags[2] = 1

    # dynamics use SI base units (m, s, ...)    
    ds = np.zeros(6)
    ds[:3] = s[3:6]
    accels = accelerations(t,s, use_alt_grav=alt_grav, use_alt_drag=alt_drag, use_alt_srp=alt_srp)

    ds[3:6] = sum([accels[i]*fflags[i] for i in range(5)])
    return ds

def accelerations(t, s, return_extras=False, use_alt_grav=1, use_alt_drag=1, use_alt_srp=1):
    r = s[:3]
    v = s[3:]
    PNRW, PN, T_UT1, dEps, omega_vec, R, W = ecef_eci(t/86400, return_extras=True)

    rsun = sun_vector(T_UT1, PN)  # ECI. [m]
    rmoon = moon_vector(T_UT1, PN, dEps)  # ECI. [m]
    eclipse = shadow(r,rsun)  # [0, 0.5, 1] for [umbra, penumbra, sunlight]

    if use_alt_grav == 1:
        a_gravity = PNRW.T @ grav_odp(*(PNRW @ r))
    elif use_alt_grav == 'egm96':
        a_gravity = PNRW.T @ egm96(np.atleast_2d(PNRW @ r),20)[:,0]
    elif use_alt_grav == '2body':
        a_gravity = twoBody(s)
    elif use_alt_grav == 'J2':
        a_gravity = JN(s,2)
    elif use_alt_grav == 'J3':
        a_gravity = JN(s,3)
    elif use_alt_grav == 'J4':
        a_gravity = PNRW.T @ JN(PNRW @ r,4)
    else:
        print('must specify a gravity model')
        raise Exception

    a_sungrav = solar_gravity(t,r,rsun)
    a_moongrav = lunar_gravity(t,r,rmoon)

    if use_alt_drag == 0:
        a_drag = np.zeros(3)
    elif use_alt_drag == 1:
        a_drag = drag(r,v,rsun,eclipse)
    elif use_alt_drag == 'cannonball':
        # recef, vecef = eci2ecef(r, v, PNRW, PN, T_UT1, dEps, omega_vec, R, W)
        a_drag = drag_cannonball(r, v)
    else:
        print('must specify a drag model')
        raise Exception

    if use_alt_srp == 0:
        a_srp = np.zeros(3)
    elif use_alt_srp == 1:
        a_srp = solarpressure(r,v,rsun,eclipse)
    elif use_alt_srp == 'cannonball':
        a_srp = solarpressure_cannonball(r,rsun,eclipse)
    else:
        print('must specify a solar pressure model')
        raise Exception
    
    if return_extras:
        return [a_gravity, a_drag, a_srp, a_sungrav, a_moongrav, rsun, rmoon]
    return [a_gravity, a_drag, a_srp, a_sungrav, a_moongrav]

def twoBody(s): 
    """Implement simple 2-body gravity
    
    Args:
    	s (np.array): state vector [x,y,z,vx,vy,vz]
    
    Returns:
    	(np.array): 3-array of gravitational acceleration in x, y, z
    """
    return -muE/norm(s[:3])**3 * s[:3]

def JN(s,deg=4):
    """Calculate zonal gravitational perturbations up to degree 4 at state s. Degree 2 is simply the J2 perturbation.
    
    Args:
    	s (np.array): state vector [x,y,z,vx,vy,vz]. ideally ECEF but can apply to ECI with lesser accuracy
    
    Returns:
    	(np.array): 3-array of gravitational acceleration in x, y, z

    TODO: implement degrees other than order 4
    TODO: is this function faster with @njit?
    """
    RE = Rearth
    x,y,z = s[0],s[1],s[2]
    r2 = x**2+y**2+z**2
    if deg != 4:
        print('not yet implemented')
        raise Exception
    # Degree 2 and 3 were never implemented -- this was a first attempt:
    # J3X = 0.5*muE*x/(r2**16) * (15*J2*RE**2*z**2*r2**12.5-3*J2*RE**2*r2**13.5+35*J3*RE**3*z**3*r2**11.5-15*J3*RE**3*z*r2**12.5-2*r2**14.5)
    # J3Y = 0.5*muE*y/(r2**16) * (15*J2*RE**2*z**2*r2**12.5-3*J2*RE**2*r2**13.5+35*J3*RE**3*z**3*r2**11.5-15*J3*RE**3*z*r2**12.5-2*r2**14.5)

    # Derived manually / assisted by Wolfram Alpha
    J4X = 0.125*muE*x/r2**6 * (
        60*J2*RE**2*z**2*r2**2.5
        -12*J2*RE**2*r2**3.5
        +140*J3*RE**3*z**3*r2**1.5
        -60*J3*RE**3*z*r2**2.5
        +306*J4*RE**4*z**4*r2**0.5
        -210*J4*RE**4*z**2*r2**1.5
        +15*J4*RE**4*r2**2.5
        -8*r2**4.5
    )
    J4Y = 0.125*muE*y/r2**6 * (
        60*J2*RE**2*z**2*r2**2.5
        -12*J2*RE**2*r2**3.5
        +140*J3*RE**3*z**3*r2**1.5
        -60*J3*RE**3*z*r2**2.5
        +306*J4*RE**4*z**4*r2**0.5
        -210*J4*RE**4*z**2*r2**1.5
        +15*J4*RE**4*r2**2.5
        -8*r2**4.5
    )
    J4Z = 0.125*muE/r2**6 * (
        60*J2*RE**2*z**3*r2**2.5
        -36*J2*RE**2*z*r2**3.5
        +140*J3*RE**3*z**4*r2**1.5
        -120*J3*RE**3*z**2*r2**2.5
        +12*J3*RE**3*r2**3.5
        +306*J4*RE**4*z**5*r2**0.5
        -346*J4*RE**4*z**3*r2**1.5
        +75*J4*RE**4*z*r2**2.5
        -8*z*r2**4.5
    )

    return np.array([J4X,J4Y,J4Z])

def solar_gravity(t,r,rsun):
    """Calculate solar perturbative force. Derivation can be found in Born and possibly Vallado, or performed by hand.
    Note that this is the *difference* solar-gravity-based acceleration between the Earth and an object orbiting the Earth, which is why this is not a simple 1/r^2 term.
    
    The time argument (variable 't') is unused. (It originally was passed here to compute the sun position, but was since orphaned and left undeleted)
    
    Args:
    	t (float): JD time represented in seconds
    	r (np.array): object position in ECI
    	rsun (np.array): sun position in ECI
    	
    Returns:
    	(np.array): acceleration vector
    """
    Deltaj = rsun - r
    return muS * (Deltaj/norm(Deltaj)**3 - rsun/norm(rsun)**3)

def lunar_gravity(t,r,rmoon):
    """Calculate lunar perturbative force. Derivation can be found in Born and possibly Vallado, or performed by hand.
    Note that this is the *difference* lunar-gravity-based acceleration between the Earth and an object orbiting the Earth, which is why this is not a simple 1/r^2 term.
    
    The time argument (variable 't') is unused. (It originally was passed here to compute the moon position, but was since orphaned and left undeleted)
    
    Args:
    	t (float): JD time represented in seconds
    	r (np.array): object position in ECI
    	rmoon (np.array): moon position in ECI
    	
    Returns:
    	(np.array): acceleration vector
    """
    Deltaj = rmoon - r
    return muM * (Deltaj/norm(Deltaj)**3 - rmoon/norm(rmoon)**3)

@njit
def shadow(r,rsun):
    """
    Implements algorithm 34: Shadow from Vallado

    Args:
    	r (np.array): object position in ECI
    	rsun (np.array): sun position in ECI
    	
    Returns:
        0    if the object is in Earth's umbra (completely shadowed)
        0.5  if the object is in Earth's penumbra (partially shadowed)
        1    if the object is not eclipsed (in full sunlight)
    """
    shadow = 1  # start by assuming the object is not shadowed

    rrsun_dot = np.dot(rsun,r)
    if rrsun_dot < 0:
        rnorm = norm(r)
        rsunnorm = norm(rsun)
        sunsep_angle = arccos(rrsun_dot / (rnorm * rsunnorm))

        sat_horiz = rnorm * cos(sunsep_angle)
        sat_vert = rnorm * sin(sunsep_angle)

        a_umb = 0.264121687 * DEG2RAD  # rad
        a_pen = 0.269007205 * DEG2RAD  # rad

        x = Rearth/sin(a_pen)
        pen_vert = tan(a_pen) * (x + sat_horiz)
        if sat_vert <= pen_vert:
            shadow = 0.5  # penumbra
            y = Rearth/sin(a_umb)
            umb_vert = tan(a_umb) * (y - sat_horiz)
            if sat_vert <= umb_vert:
                shadow = 0  # umbra
    return shadow

def solarpressure(r,v,rsun,eclipse):
    """
    Return the solar radiation force for a given ECI position r
    AT LEAST, have a Cannon Ball Model for SRP
    BETTER, Model the SRP due to Solar Panel
    BEST, Attitude Dependent, Facet Based SRP Model (Refer to Precise Orbit Determination for
    TOPEX/POSEDION). Include Specular and Diffuse reflectivity for each face.
    
    You need to take into account the effects of the Earth's Shadow. If the Spacecraft is in the
    Earth's Shadow, then no light shines on it, so there is no solar radiation pressure force
    experienced!! A very simple algorithm to compute if the spacecraft is in the shadow or not
    (based on Line of Sight vectors) can be found in Vallado.
    """
    if eclipse == 0:
        return np.array([0,0,0])
    
    # sun-to-satellite unit vector
    u_sunsat = (r - rsun) / norm(r - rsun)
    
    # face unit normal vectors
    u_xface = v / norm(v)
    u_zface = r / norm(r)
    u_yface = np.cross(u_xface, u_zface)
    u_scell = -u_sunsat
    faces_normals = [u_xface, u_yface, u_zface, u_scell]
    faces_properties = [xface, yface, zface, scell]
    
    def srp_face_accel(u_sunsat, u_face, face_properties):
        # http://www.ub.edu/wai/wp-content/uploads/2021/07/congress_presentation.pdf
        # https://www.hindawi.com/journals/ijae/2015/928206/
        dot_uface = np.dot(u_sunsat, u_face)  # = cos(theta), the angle between the sun and the face normal
        if dot_uface > 0:
            u_face = -u_face
            dot_uface = -dot_uface
        # Pressures. TODO: Check signs and values!!
        P_absorbed = face_properties['C_a'] * -1 * pSRP * dot_uface * u_sunsat
        P_specular = face_properties['C_s'] * -2 * pSRP * dot_uface**2 * u_face
        P_diffused = face_properties['C_d'] * -1 * pSRP * dot_uface * (u_sunsat + 2/3*u_face) # is this correct???
        a_face = face_properties['area'] * (P_absorbed + P_specular + P_diffused) / mass
        return a_face
    
    a_srp = sum([srp_face_accel(u_sunsat, u_face, face_properties)
                 for u_face, face_properties in zip(faces_normals, faces_properties)
                ])
    
    return a_srp * eclipse

def solarpressure_cannonball(r,rsun,eclipse):
    """
    Based on Ch 8 Vallado. Implements solar radiation force given a canonball model for satellite reflectivity.
    """
    if eclipse == 0:
        return np.array([0,0,0])
    
    # sun-to-satellite unit vector
    distance_m = norm(r - rsun)
    distance_AU = distance_m / AU
    u_sunsat = (r - rsun) / distance_m
    
    C_R = 0.63
    A = 6  # m^2
    return pSRP/distance_AU**2 * C_R * A/mass * u_sunsat * eclipse

def solarpressure_facet(r,rsun,eclipse):
    """
    Implements solar radiation force given a full facet-based model for satellite reflectivity.
    Satellite dimensions and reflectivity coefficients (gamma, beta) are hard-coded in this function. TODO: make these args.
    Return the solar radiation force for a given ECI position r.
    AT LEAST, have a Cannon Ball Model for SRP
    BETTER, Model the SRP due to Solar Panel
    BEST, Attitude Dependent, Facet Based SRP Model (Refer to Precise Orbit Determination for
    TOPEX/POSEDION). Include Specular and Diffuse reflectivity for each face.
    

    """
    if eclipse == 0:
        return np.array([0,0,0])  # zero radiation pressure if the object is eclipsed by Earth
    
    # sun-to-satellite unit vector
    distance_m = norm(r - rsun)
    distance_AU = distance_m / AU
    u_sunsat = (r - rsun) / distance_m
    normal = -u_sunsat

    gamma_ = 0.5
    beta = 0.5
    A = 10  # m^2

    theta = 0
    nu = 1/3 * ((1-beta)*gamma_)
    mu = 0.5*beta*gamma_
    B_theta = 2*nu*cos(theta) + 4*mu*cos(theta)**2
    F_srp = -pSRP/distance_AU**2 * (B_theta * normal + (1-mu)*cos(theta)**2 * -u_sunsat) * A

    return F_srp/mass * eclipse

def drag(r,v,rsun,eclipse):
    """
    Return the drag force for a given ECI position r
    Full Fidelity Spacecraft Model with Drag face as the face in the Spacecraft Velocity Direction.
    Spacecraft configs are currently hard-coded in the constants.py module. TODO: make them function args.
    """
    vA = np.array([v[0]+d_theta_E*r[1], v[1]-d_theta_E*r[0],v[2]])
    vAnorm = norm(vA)

    # assume the solar cell is sun-pointed
    scell_normal = (r - rsun) / norm(r - rsun)
    scell_area = np.abs(np.dot(scell_normal, vA/vAnorm)) * scell['area']

    A = xface['area'] + scell_area
    rho = rho0 * np.exp(-(norm(r)-r0)/H)
    F_drag = -0.5*rho*vAnorm*vA * Cd * A
    return F_drag/mass

def drag_cannonball(r,v):
    """Return the drag force for a given ECI position r and velocity v. just use a cannonball model for drag.
    Object area is hard-coded in this function."""
    vA = np.array([v[0]+d_theta_E*r[1], v[1]-d_theta_E*r[0],v[2]])
    vAnorm = norm(vA)
    A = 6  # m^2
    rho = rho0 * np.exp(-(norm(r)-r0)/H)
    F_drag = -0.5*rho*vAnorm*vA * Cd * A
    return F_drag/mass

def gravity(r):
    """
    Return the gravitational acceleration for a given ECEF position r [km]
    A 20x20 EGM-96 gravity field model is used.
    
    TODO: fit vs truncated 20x20 model --- currently the coefficients are just truncated from a higher-order model
    """
    # https://people.sc.fsu.edu/~lb13f/projects/space_environment/egm96.php
    # https://github.com/lukasbystricky/SpaceSimulator/blob/master/Environment/Geopoential/geopotential.m
    r_ecef = r
    rnorm = norm(r_ecef)
    theta = arccos(r_ecef[2]/rnorm)
    phi = arctan2(r_ecef[1],r_ecef[0])

    n_arr = np.arange(21)
    m_arr = np.arange(21)

    lpmn = scipy.special.lpmn(20,20,sin(theta))
    Ymn = lpmn[0]
    Ymn_deriv = lpmn[1]

    cos_theta = cos(theta)
    sin_theta = sin(theta)
    cos_phi = cos(phi)
    sin_phi = sin(phi)

    # Create a mask for the region where m < n
    mask = np.tri(*Ymn.shape, k=0, dtype=bool)

    # Calculate cos(phi * m) and sin(phi * m)
    cos_phi_m = np.cos(m_arr * phi)
    sin_phi_m = np.sin(m_arr * phi)
    
    Rearth_n_arr = Rearth**n_arr
    rnorm_n_arr_2 = rnorm**(n_arr+2)

    inner_sum = np.ma.array(Ymn * (C * cos_phi_m[:, None] + S * sin_phi_m[:, None]), mask=mask)
    g_r = -muE/rnorm**2 - (muE*(n_arr+1)*Rearth_n_arr / rnorm_n_arr_2 * inner_sum.sum(axis=0)).sum()
    
    inner_sum = np.ma.array(Ymn_deriv * cos_theta * (C * cos_phi_m[:, None] + S * sin_phi_m[:, None]), mask=mask)
    g_theta = (muE*Rearth_n_arr / rnorm_n_arr_2 * inner_sum.sum(axis=0)).sum()
    
    # this was incorrect, probably due to m_arr in the inner sum
    #inner_sum = np.ma.array(m_arr * Ymn * (S * cos_phi_m[:, None] - C * sin_phi_m[:, None]), mask=mask)
    #g_phi = (muE*Rearth**n_arr / (rnorm**(n_arr+2)*sin_theta) * inner_sum.sum(axis=0)).sum()
    
    g_phi = sum([
        muE*Rearth**n / (rnorm**(n+2)*sin_theta) * sum([
            m*Ymn[m,n] * (S[m,n]*cos_phi_m[m] - C[m,n]*sin_phi_m[m])
            for m in range(n)
        ])
        for n in range(1,21)
    ])
    
    g = np.array([g_r, g_theta, g_phi])
    
    # Rotate from spherical coordinates to cartesian coordinates
    rot = np.array([
        [sin_theta*cos_phi, cos_theta*cos_phi, -sin_phi],
        [sin_theta*sin_phi, cos_theta*cos_phi, cos_phi],
        [cos_theta, -sin_theta, 0]
    ])
    
    return rot @ g


