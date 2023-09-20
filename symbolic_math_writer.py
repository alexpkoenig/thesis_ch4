"""

This module derives analytical Jacobians used in the EKF.

Evaluations of the analytical Jacobians are implemented in symbolic_math.py.
This module simply derives the Jacobians and saves them to pickle files for later use.
(This is much more efficient than analytically re-deriving the Jacobians during each evaluation!)

Jacobian names map to those defined in Chapter 4 of the thesis.

"""

import sympy as sp

import constants


R2, J2, J3, J4, muE, muM, muS = sp.symbols(
    'R2 J2 J3 J4 mu_E mu_M mu_S'
)

x, y, z, xdot, ydot, zdot, Cd_, xddot, yddot, zddot, Cddot = sp.symbols(
    'x y z xdot ydot zdot C_d xddot yddot zddot Cddot'
)

rho, Ax, m, H, rho0, r0, r, v = sp.symbols(
    'rho A_x m H rho_0 r_0 r v'
)

Rearth, r, pSRP, Acell, xsun, ysun, zsun, xmoon, ymoon, zmoon = sp.symbols(
    'R_E r p_SRP A_cell x_sun y_sun z_sun x_moon y_moon z_moon'
)

def symbolicJacobian(): 
    r = (x**2 + y**2 + z**2)**(1/2)
    v = (xdot**2 + ydot**2 + zdot**2)**(1/2)

    # Gravity terms: 2-body, J2, J3, J4 (and lunisolar perturbations)
    # See Vallado pg 593-594 (Section 8.7.1)
    Grav = muE/r
    R2 = -(J2*muE*Rearth**2)/(2*r**3) * (3*(z/r)**2 - 1)
    R3 = -(J3*muE)/(2*r)*(Rearth/r)**3 * (5*(z/r)**3 - 3*(z/r))
    R4 = -(J4*muE)/(8*r)*(Rearth/r)**4 * (34*(z/r)**4 - 30*(z/r)**2 + 3)
    grav_potential = Grav + R2 + R3 + R4

    # Acceleration due to gravity
    xddot_grav = sp.diff(grav_potential, x).simplify()
    yddot_grav = sp.diff(grav_potential, y).simplify()
    zddot_grav = sp.diff(grav_potential, z).simplify()

    # Solar perturbations -- see Born pg. 63, Eqn. 2.3.39
    DeltaSunX = xsun - x
    DeltaSunY = ysun - y
    DeltaSunZ = zsun - z
    normSun = (xsun**2 + ysun**2 + zsun**2)**(1/2)
    normDeltaSun = (DeltaSunX**2 + DeltaSunY**2 + DeltaSunZ**2)**(1/2)
    xddot_sun = muS * (DeltaSunX/normDeltaSun**3 - xsun/normSun**3)
    yddot_sun = muS * (DeltaSunY/normDeltaSun**3 - ysun/normSun**3)
    zddot_sun = muS * (DeltaSunZ/normDeltaSun**3 - zsun/normSun**3)

    # Lunar perturbations -- see Born pg. 63, Eqn. 2.3.39
    DeltaMoonX = xmoon - x
    DeltaMoonY = ymoon - y
    DeltaMoonZ = zmoon - z
    normMoon = (xmoon**2 + ymoon**2 + zmoon**2)**(1/2)
    normDeltaMoon = (DeltaMoonX**2 + DeltaMoonY**2 + DeltaMoonZ**2)**(1/2)
    xddot_moon = muM * (DeltaMoonX/normDeltaMoon**3 - xmoon/normMoon**3)
    yddot_moon = muM * (DeltaMoonY/normDeltaMoon**3 - ymoon/normMoon**3)
    zddot_moon = muM * (DeltaMoonZ/normDeltaMoon**3 - zmoon/normMoon**3)

    # Drag terms
    vAx = xdot + constants.d_theta_E * y
    vAy = ydot - constants.d_theta_E * x
    vAz = zdot
    vAnorm = (vAx**2 + vAy**2 + vAz**2)**(1/2)
    rho = rho0 * sp.exp(-(r-r0)/H)
    dragPrefix = -0.5*rho*Cd_*Ax/m * vAnorm

    # Acceleration due to drag
    xddot_drag = dragPrefix * vAx
    yddot_drag = dragPrefix * vAy
    zddot_drag = dragPrefix * vAz

    # Solar radiation pressure terms
    # TODO: don't leave these hard-coded in this function (was OK for single-object analysis)
    Cd_cell = 0.04
    Cs_cell = 0.04
    Ca_cell = 1-0.04-0.04
    srpPrefix = - pSRP/m * Acell * (Ca_cell + 2*Cs_cell + 4/3*Cd_cell)

    # Acceleration due to solar radiation pressure
    xddot_srp = srpPrefix * xsun
    yddot_srp = srpPrefix * ysun
    zddot_srp = srpPrefix * zsun


    # Compute the sum of acceleration contributions
    xddot = xddot_grav + xddot_sun + xddot_moon + xddot_drag + xddot_srp
    yddot = yddot_grav + yddot_sun + yddot_moon + yddot_drag + yddot_srp
    zddot = zddot_grav + zddot_sun + zddot_moon + zddot_drag + zddot_srp

    state_vector = [x, y, z, xdot, ydot, zdot, Cd_]
    f = sp.Matrix([xdot, ydot, zdot, xddot, yddot, zddot, 0])
    Jacobian = f.jacobian(state_vector)
    Jacobian.simplify()
    
    return Jacobian, f

Jacobian, F = symbolicJacobian()

# Create a lambda function from the Jacobian expression. The terms below are all required for Jacobian evaluation
Jacobian_func = sp.lambdify(
    (muE, muS, muM, J2, J3, J4, Rearth, rho0, r0, H, Cd_, pSRP, m, Ax, x, y, z, xdot, ydot, zdot, xsun, ysun, zsun, xmoon, ymoon, zmoon),
    Jacobian, 'numpy')

xsat,ysat,zsat,xdotsat,ydotsat,zdotsat,xstat,ystat,zstat,xdotstat,ydotstat,zdotstat,Cd,omeg0,omeg1,omeg2 = sp.symbols(
    'x y z xdot ydot zdot x_stat y_stat z_stat xdot_stat ydot_stat zdot_stat Cd omega_0 omega_1 omega_2'
)
# Coordinate rotations; see Vallado ECEF <-> ECI
PNR_ = sp.MatrixSymbol('PNR', 3,3)
W_ = sp.MatrixSymbol('W', 3,3)

def symbolicH():
    omega_vec = sp.Matrix([omeg0, omeg1, omeg2])

    reci = sp.Matrix([xsat,ysat,zsat])
    veci = sp.Matrix([xdotsat,ydotsat,zdotsat])
    PNRW_ = PNR_ * W_
    rpef = (PNR_.T * reci).as_explicit()
    recef = (PNRW_.T * reci).as_explicit()
    vecef = (W_.T * (PNR_.T * veci - omega_vec.cross(rpef.T))).as_explicit()

    stationecef = sp.Matrix([xstat,ystat,zstat])

    range_ = (recef - stationecef).dot(recef-stationecef)**0.5
    range_rate = (recef - stationecef).dot(vecef) / range_

    state_vector = [xsat, ysat, zsat, xdotsat, ydotsat, zdotsat, Cd]
    f = sp.Matrix([range_, range_rate])
    Jacobian = f.jacobian(state_vector)

    return Jacobian

def symbolicHr_ECI():
    reci = sp.Matrix([xsat,ysat,zsat])
    veci = sp.Matrix([xdotsat,ydotsat,zdotsat])
    rStateci = sp.Matrix([xstat,ystat,zstat])
    vStateci = sp.Matrix([xdotstat,ydotstat,zdotstat])
    range_ = (reci - rStateci).dot(reci - rStateci)**0.5
    range_rate = (reci - rStateci).dot(veci - vStateci) / range_
    state_vector = [xsat, ysat, zsat, xdotsat, ydotsat, zdotsat, Cd]
    f = sp.Matrix([range_, range_rate])
    Jacobian = f.jacobian(state_vector)
    return Jacobian

def symbolicHo():
    reci = sp.Matrix([xsat,ysat,zsat])
    # veci = sp.Matrix([xdotsat,ydotsat,zdotsat])
    rSateci = sp.Matrix([xstat,ystat,zstat])
    # vSateci = sp.Matrix([xdotstat,ydotstat,zdotstat])
    
    obs2obj = reci - rSateci
    dist = (obs2obj).dot(obs2obj)**0.5
    obs2obj /= dist

    theta = sp.acos(obs2obj[2])
    phi = sp.atan2(obs2obj[1], obs2obj[0])

    state_vector = [xsat, ysat, zsat, xdotsat, ydotsat, zdotsat, Cd]
    f = sp.Matrix([theta, phi])
    Jacobian = f.jacobian(state_vector)
    return Jacobian

Htilde = symbolicH()
Htilde_func = sp.lambdify(
    (xsat, ysat, zsat, xdotsat, ydotsat, zdotsat, xstat, ystat, zstat, W_, PNR_, omeg0, omeg1, omeg2),
    Htilde, 'numpy')

Hrtilde = symbolicHr_ECI()
Hrtilde_func = sp.lambdify(
    (xsat, ysat, zsat, xdotsat, ydotsat, zdotsat, xstat, ystat, zstat, xdotstat, ydotstat, zdotstat),
    Hrtilde, 'numpy'
)

Hotilde = symbolicHo()
Hotilde_func = sp.lambdify(
    (xsat, ysat, zsat, xstat, ystat, zstat),
    Hotilde, 'numpy'
)

import dill

# Save the lambdified expressions to disk for later evaluation
with open("./code_data_files/jacobian_func.pkl", "wb") as f:
    dill.dump(Jacobian_func, f)
with open("./code_data_files/htilde_func.pkl", "wb") as f:
    dill.dump(Htilde_func, f)
with open("./code_data_files/hrtilde_func.pkl", "wb") as f:
    dill.dump(Hrtilde_func, f)
with open("./code_data_files/hotilde_func.pkl", "wb") as f:
    dill.dump(Hrtilde_func, f)
