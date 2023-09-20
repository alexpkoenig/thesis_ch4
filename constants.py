"""

This module defines constants, coefficients, and conversion factors used throughout the rest of the code.

Spacecraft model properties (e.g. mass, face areas, diffuse/specular reflectivity and absorptivity coefficients) are also defined here.
TODO: make these properties modular rather than hard-coded. (This did not matter as much for single-object analysis.)

All values use SI base units unless otherwise specified.


NOTE:
Any absolute times expressed in seconds (generally assigned to variable 't' throughout the rest of these modules)
are assumed to be expressed in terms of a Julian date converted to total seconds.
I.e., J2000 is a JD of 2451545.0, which would be expressed as 2451545*86400 seconds.

"""

import numpy as np

from timefunctions import str2time

### INITIALIZATION ###

# epoch for initial conditions
init_epoch_str = '23 Mar 2018, 08:55:03.0 UTC'
fin_epoch_str = '30 Mar 2018, 08:55:03.0 UTC'
init_epoch = str2time(init_epoch_str)  # Julian date in days
fin_epoch = str2time(fin_epoch_str)
init_epoch_s = init_epoch * 86400  # s
fin_epoch_s = fin_epoch * 86400

# alternate epoch for initial conditions
init_epoch_str5 = '01 Feb 2018, 05:00:00.0 UTC'
init_epoch5 = str2time(init_epoch_str5)  # Julian date in days

# initial satellite state
r_i = np.array([6984.45711518852, 1612.2547582643, 13.0925904314402])*1e3  # m
v_i = np.array([-1.67667852227336, 7.26143715396544, 0.259889857225218])*1e3  # m s^-1
s_i = np.concatenate([r_i,v_i])
# alternate state
r_i5 = np.array([6990077.798814194, 1617465.311978378, 22679.810569245355])  # m
v_i5 = np.array([-1675.13972506056, 7273.72441330686, 252.688512916741])  # m s^-1
s_i5 = np.concatenate([r_i5, v_i5])

# station coordinates in body-fixed coordinate system
s_1 = np.array([-6143584, 1364250, 1033743])  # m
s_2 = np.array([1907295, 6030810, -817119])  # m
s_3 = np.array([2390310, -5564341, 1994578])  # m

# spacecraft model properties
mass = 2000  # kg
xface = {'area':6, 'C_d':0.04, 'C_s':0.59}
yface = {'area':8, 'C_d':0.04, 'C_s':0.59}
zface = {'area':12, 'C_d':(0.80+0.28)/2, 'C_s':(0.04+0.18)/2}
scell = {'area':15, 'C_d':0.04, 'C_s':0.04}
xface['C_a'] = 1 - xface['C_d'] - xface['C_s']
yface['C_a'] = 1 - yface['C_d'] - yface['C_s']
zface['C_a'] = 1 - zface['C_d'] - zface['C_s']
scell['C_a'] = 1 - scell['C_d'] - scell['C_s']


### CONSTANTS ###

# general constants
pi         = np.pi
muE        = 398600.4415 * 1e9  # m^3 s^-2
muS        = 132712440018 * 1e9  # m^3 s^-2
muM        = 4902.800066 * 1e9  # m^3 s^-2
Rearth     = 6378.1363 * 1e3  # m
AU         = 149597870.7 * 1e3  # m
eccE       = 0.081819221456  # Earth eccentricity
omegaE     = 7.292115146706979e-5  # rad s^-1
c          = 2.998e8  # m s^-1
DAY_s      = 86400
MINUTE_s   = 60

# conversions
DEG2RAD    = pi/180
ARCSEC2RAD = pi/648000
MILLIARCSEC2RAD = pi/648000 * 1e-3

# time constants
from timefunctions import J2000
MJD        = 2400000.5  # days. Used for EOP data reference
dUTC_TAI   = 37  # seconds between UTC and TAI. valid from 2017-01 onward, as per IERS bulletin C

# drag constants -- the atmospheric model is exponential and defined at a 700 km reference altitude
d_theta_E = 7.292115146706979e-5  # rad/s. Earth rotation rate.
rho0 = 3.614e-13  # [kg m^-3]. reference density
r0 = 700000.0 + Rearth  # [m]. reference altitude
H = 88667.0  # [m]. atmospheric scale height
Cd = 1.88  # drag coefficient

# solar radiation constants
pSRP = 1367 / c

# EGM96 gravity field coefficients
EGM = np.loadtxt('./code_data_files/EGM96coeffs.txt')
EGM_n = EGM[:,0]
EGM_m = EGM[:,1]
C_mn = EGM[:,2]
S_mn = EGM[:,3]
C = np.zeros((21,21))
S = np.zeros((21,21))
i = 2
idx = 0
while i < 21:
    for j in range(i+1):
        C[j,i] = C_mn[idx+j]
        S[j,i] = S_mn[idx+j]
    i += 1
    idx += i

# Zonal gravity coefficients
J2 = -C[0,2] * np.sqrt(2*2+1)
J3 = -C[0,3] * np.sqrt(2*3+1)
J4 = -C[0,4] * np.sqrt(2*4+1)

C = C.T
S = S.T
