#!/usr/bin/env python3

"""

Implement conversions between Cartesian state vectors and Keplerian orbital elements.

The formatting convention used for states throughout this module and the rest of the code is as follows:
state vector (np.array of length 6): Cartesian state vector of [position in x, position in y, position in z, velocity in x, velocity in y, velocity in z], in SI base units
orbital elements (np.array of length 6) Keplerian elements of [semjax, eccentricity, inclination, longitude of the ascending node, argument of periapsis, true anomaly

TODO: can runtime be improved with Numba's @jit or @njit compiling?

"""

import numpy as np
import scipy.spatial.transform

from constants import muE as mu
from constants import Rearth


def rotate(vector,axis,angle):
    """Rotate a vector about an axis by an angle (in radians). Works also for 6D state vectors."""
    axis = axis / np.linalg.norm(axis)
    if vector.shape[0] == 6:
        return scipy.spatial.transform.Rotation.from_rotvec(angle * axis).apply(vector.reshape(2,3)).reshape(6)
    else:
        return scipy.spatial.transform.Rotation.from_rotvec(angle * axis).apply(vector)

def Rotx(angle):
    "Create a rotation matrix (angle in radians) about the x axis."""
    return np.array([
        [1,0,0],
        [0,np.cos(angle),-np.sin(angle)],
        [0,np.sin(angle),np.cos(angle)]
    ])

def Rotz(angle):
    """Create a rotation matrix (angle in radians) about the z axis."""
    return np.array([
        [np.cos(angle),-np.sin(angle),0],
        [np.sin(angle),np.cos(angle),0],
        [0,0,1]
    ])

def car2kep(s):
    """Convert Cartesian state vector s (position, velocity 6-vector) into Keplerian elements."""
    r,v = s[:3],s[3:]
    
    a = 1/(2/np.linalg.norm(r) - np.linalg.norm(v)**2/mu)
    h = np.cross(r,v)
    e = np.cross(v,h)/mu - r/np.linalg.norm(r)
    ecc = np.linalg.norm(e)
    i = np.arccos(h[2]/np.linalg.norm(h))
    
    n = np.cross(np.array([0,0,1]),h)
    
    nu = np.arccos(np.dot(e,r)/(ecc*np.linalg.norm(r)))
    if np.dot(r,v)<0:
        nu = 2*np.pi - nu
    
    w = np.arccos(np.dot(n,e)/(ecc*np.linalg.norm(n)))
    if e[2]<0:
        w = 2*np.pi - w
    
    # Option: calculate mean and eccentric anomalies.
    # E = 2*np.arctan2(np.tan(nu/2), np.sqrt((1+ecc)/(1-ecc)))
    # M = E - ecc*np.sin(E)
    
    lan = np.arccos(n[0]/np.linalg.norm(n))
    if n[1]<0:
        lan = 2*np.pi - lan
    
    return np.array([a,ecc,i,lan,w,nu])

def kep2car(s):
    """Convert Keplerian elements s into a Cartesian state vector."""
    a,ecc,i,lan,w,nu = s
    
    E = 2*np.arctan2(np.tan(nu/2), np.sqrt((1+ecc)/(1-ecc)))
    
    dist = a*(1-ecc*np.cos(E))
    r0 = np.array([np.cos(nu),np.sin(nu),0]) * dist
    v0 = np.array([-np.sin(E),np.sqrt(1-ecc**2)*np.cos(E),0]) * np.sqrt(mu*a)/dist
    
    Rot_lan, Rot_i, Rot_w = Rotz(lan), Rotx(i), Rotz(w)
    r = np.matmul(Rot_lan, np.matmul(Rot_i, np.matmul(Rot_w, r0)))
    v = np.matmul(Rot_lan, np.matmul(Rot_i, np.matmul(Rot_w, v0)))
    
    return np.hstack([r,v])

def walker_delta(Ntot, Nplane, inc):
    """
    Return elems for a walker delta constellation
    Ntot sats distributed across Nplane orbital planes, each at inclination inc
    """
    Nsat = int(Ntot/Nplane)
    plane = lambda idx_plane, idx_sat, Nplane,Nsat: [
        Rearth + 500e3,  # semjax
        0,  # eccentricity
        inc,  # inclination
        idx_plane*(2*np.pi/Nplane),  # LAN
        0,  # arg perigee
        idx_sat*(2*np.pi/Nsat)+idx_plane*(2*np.pi/(Nplane*Nsat))  # phasing
    ]
    return [*[plane(j,i, Nplane, Nsat) for i in range(Nsat) for j in range(Nplane)]]
