import numpy as np
from numba import jit

from constants import C, S, Rearth, muE

Re,GM = Rearth, muE
RE = Rearth
mu = muE

@jit
def grav_odp(x,y,z):
    nmaxp1, mmaxp1 = C.shape
    nmax = nmaxp1 - 1
    mmax = mmaxp1 - 1
    Anm = np.zeros(nmaxp1 + 1)
    Anm1 = np.zeros(nmaxp1 + 1)
    Anm2 = np.zeros(nmaxp1 + 2)
    R = np.zeros(nmaxp1 + 1)
    I = np.zeros(nmaxp1 + 1)
    rb2 = x * x + y * y + z * z
    rb = np.sqrt(rb2)
    mur2 = mu / rb2
    mur3 = mur2 / rb

    #   direction of spacecraft position
    s = x / rb
    t = y / rb
    u = z / rb

    #	/* ***********************************************************/
    #	/* Calculate contribution of only Zonals                     */s
    #	/* ***********************************************************/

    Anm1[0] = 0
    Anm1[1] = np.sqrt(3)
    Anm2[1] = 0
    Anm2[2] = np.sqrt(3.75)
    as_ = 0.0
    at = 0.0
    au = 0.0
    ar = 0.0
    rat1 = 0
    rat2 = 0
    Dnm = 0
    Enm = 0
    Fnm = 0
    Apor = np.zeros(nmaxp1)
    Apor[0] = 1
    Apor[1] = RE / rb

    for n in range(1, nmax + 1):
        i = n
        an2 = 2 * n
        rat1 = np.sqrt((an2 + 3.0) * (((an2 + 1.0) / n) / (n + 2.0)))
        rat2 = np.sqrt((n + 1.0) * (((n - 1.0) / (an2 - 1.0)) / (an2 + 1.0)))
        Anm1[i + 1] = rat1 * (u * Anm1[i] - rat2 * Anm1[i - 1])
        Apor[i] = Apor[i - 1] * Apor[1]
        if n < mmaxp1:
            rat1 = np.sqrt((an2 + 5.0) * (((an2 + 3.0) / n) / (n + 4.0)))
            rat2 = np.sqrt((n + 3.0) * (((n - 1.0) / (an2 + 1.0)) / (an2 + 3.0)))
            Anm2[i + 2] = rat1 * (u * Anm2[i + 1] - rat2 * Anm2[i])
        if n < nmaxp1:
            rat1 = np.sqrt(0.5 * n * (n + 1.0))
            au = au - Apor[i] * rat1 * Anm1[i] * (-C[i, 0])
            rat2 = np.sqrt(0.5 * ((an2 + 1.0) / (an2 + 3.0)) * (n + 1.0) * (n + 2.0))
            ar = ar + Apor[i] * rat2 * Anm1[i + 1] * (-C[i, 0]) 

    #   /* ***********************************************************/
    #   /* Calculate contribution of Tesserals                       */
    #   /* ***********************************************************/

    R[0] = 1
    I[0] = 0

    for m in range(1, mmax + 1):
        j = m
        am2 = 2 * m
        R[j] = s * R[j - 1] - t * I[j - 1]
        I[j] = s * I[j - 1] + t * R[j - 1]
        for l in range(m, mmax + 1):
            i = l
            Anm[i] = Anm1[i]
            Anm1[i] = Anm2[i]
        Anm1[-1] = Anm2[-1]
            
        for l in range(m, mmax + 1):
            i = l
            an2 = 2 * l
            if l == m:
                Anm2[j + 1] = 0.0
                Anm2[j + 2] = np.sqrt((am2 + 5.0) / (am2 + 4.0)) * Anm1[j + 1]
            else:
                rat1 = np.sqrt((an2 + 5.0) * (((an2 + 3.0) / (l - m)) / (l + m + 4.0)))
                rat2 = np.sqrt((l + m + 3.0) * (((l - m - 1.0) / (an2 + 1.0)) / (an2 + 3.0)))
                Anm2[i + 2] = rat1 * (u * Anm2[i + 1] - rat2 * Anm2[i])

            Dnm = C[i,j]*R[j] + S[i,j]*I[j]
            Enm = C[i,j]*R[j-1] + S[i,j]*I[j-1]
            Fnm = S[i,j]*R[j-1] - C[i,j]*I[j-1]

            rat1 = np.sqrt((l + m + 1.0) * (l - m))
            rat2 = np.sqrt(((an2 + 1.0) / (an2 + 3.0)) * (l + m + 1.0) * (l + m + 2.0))

            as_ = as_ + Apor[i] * m * Anm[i] * Enm
            at = at + Apor[i] * m * Anm[i] * Fnm
            au = au + Apor[i] * rat1 * Anm1[i] * Dnm  #
            ar = ar - Apor[i] * rat2 * Anm1[i + 1] * Dnm  #
            
    #   /* ***********************************************************/
    #   /* Calculate the spacecraft acceleration in ECEF             */
    #   /* ***********************************************************/

    agx_ECEF = -mur3 * x + mur2 * (as_ + ar * s)
    agy_ECEF = -mur3 * y + mur2 * (at + ar * t)
    agz_ECEF = -mur3 * z + mur2 * (au + ar * u)

    accel =  np.array([agx_ECEF, agy_ECEF, agz_ECEF]).flatten()
    return accel



def egm96(p, maxdeg):
    r = np.linalg.norm(p)

    # Compute geocentric latitude
    phic = np.arcsin(p[:, 2] / r)

    # Compute lambda
    lambda_ = np.arctan2(p[:, 1], p[:, 0])

    smlambda = np.zeros((p.shape[0], maxdeg + 1))
    cmlambda = np.zeros((p.shape[0], maxdeg + 1))

    slambda = np.sin(lambda_)
    clambda = np.cos(lambda_)
    smlambda[:, 0] = 0
    cmlambda[:, 0] = 1
    smlambda[:, 1] = slambda
    cmlambda[:, 1] = clambda

    for m in range(2, maxdeg + 1):
        smlambda[:, m] = 2.0 * clambda * smlambda[:, m - 1] - smlambda[:, m - 2]
        cmlambda[:, m] = 2.0 * clambda * cmlambda[:, m - 1] - cmlambda[:, m - 2]

    def loc_gravLegendre(phi, maxdeg):
        P = np.zeros((maxdeg + 3, maxdeg + 3, len(phi)))
        scaleFactor = np.zeros((maxdeg + 3, maxdeg + 3, len(phi)))
        cphi = np.cos(np.pi / 2 - phi)
        sphi = np.sin(np.pi / 2 - phi)

        # force numerically zero values to be exactly zero
        cphi[np.abs(cphi) <= np.finfo(float).eps] = 0
        sphi[np.abs(sphi) <= np.finfo(float).eps] = 0

        # Seeds for recursion formula
        P[0, 0, :] = 1  # n = 0, m = 0;
        P[1, 0, :] = np.sqrt(3) * cphi  # n = 1, m = 0;
        scaleFactor[0, 0, :] = 0
        scaleFactor[1, 0, :] = 1
        P[1, 1, :] = np.sqrt(3) * sphi  # n = 1, m = 1;
        scaleFactor[1, 1, :] = 0

        for n in range(2, maxdeg + 2):
            k = n + 1
            for m in range(0, n + 1):
                p = m + 1
                if n == m:
                    P[k, k, :] = np.sqrt(2 * n + 1) / np.sqrt(2 * n) * sphi * P[k - 1, k - 1, :]
                    scaleFactor[k, k, :] = 0
                elif m == 0:
                    P[k, p, :] = (np.sqrt(2 * n + 1) / n) * (np.sqrt(2 * n - 1) * cphi * P[k - 1, p, :] - (n - 1) / np.sqrt(2 * n - 3) * P[k - 2, p, :])
                    scaleFactor[k, p, :] = np.sqrt((n + 1) * n / 2)
                else:
                    P[k, p, :] = np.sqrt(2 * n + 1) / (np.sqrt(n + m) * np.sqrt(n - m)) * (np.sqrt(2 * n - 1) * cphi * P[k - 1, p, :] - np.sqrt(n + m - 1) * np.sqrt(n - m - 1) / np.sqrt(2 * n - 3) * P[k - 2, p, :])
                    scaleFactor[k, p, :] = np.sqrt((n + m + 1) * (n - m))
        return P, scaleFactor


    def loc_gravityPCPF(p, maxdeg, P, C, S, smlambda, cmlambda, GM, Re, r, scaleFactor):
        rRatio = Re / r
        rRatio_n = rRatio

        # Initialize summation of gravity in radial coordinates
        dUdrSumN = 1
        dUdphiSumN = 0
        dUdlambdaSumN = 0

        # Summation of gravity in radial coordinates
        for n in range(1, maxdeg):# + 1):
            k = n + 1
            rRatio_n = rRatio_n * rRatio
            dUdrSumM = 0
            dUdphiSumM = 0
            dUdlambdaSumM = 0
            for m in range(0, n):# + 1):
                j = m + 1
                dUdrSumM = dUdrSumM + P[k, j, :].reshape(r.shape) * (C[k, j] * cmlambda[:, j] + S[k, j] * smlambda[:, j])
                dUdphiSumM = dUdphiSumM + ((P[k, j + 1, :].reshape(r.shape) * scaleFactor[k, j, :].reshape(r.shape)) - (p[:, 2] / np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) * m * P[k, j, :].reshape(r.shape))) * (C[k, j] * cmlambda[:, j] + S[k, j] * smlambda[:, j])
                dUdlambdaSumM = dUdlambdaSumM + m * P[k, j, :].reshape(r.shape) * (S[k, j] * cmlambda[:, j] - C[k, j] * smlambda[:, j])
            dUdrSumN = dUdrSumN + dUdrSumM * rRatio_n * k
            dUdphiSumN = dUdphiSumN + dUdphiSumM * rRatio_n
            dUdlambdaSumN = dUdlambdaSumN + dUdlambdaSumM * rRatio_n

        # Gravity in spherical coordinates
        dUdr = -GM / (r * r) * dUdrSumN
        dUdphi = GM / r * dUdphiSumN
        dUdlambda = GM / r * dUdlambdaSumN

        # Gravity in ECEF coordinates
        gx = ((1. / r) * dUdr - (p[:, 2] / (r * r * np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2))) * dUdphi) * p[:, 0] - (dUdlambda / (p[:, 0] ** 2 + p[:, 1] ** 2)) * p[:, 1]
        gy = ((1. / r) * dUdr - (p[:, 2] / (r * r * np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2))) * dUdphi) * p[:, 1] + (dUdlambda / (p[:, 0] ** 2 + p[:, 1] ** 2)) * p[:, 0]
        gz = (1. / r) * dUdr * p[:, 2] + ((np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)) / (r * r)) * dUdphi

        # Special case for poles
        atPole = np.abs(np.arctan2(p[:, 2], np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2))) == np.pi / 2
        if np.any(atPole):
            gx[atPole] = 0
            gy[atPole] = 0
            gz[atPole] = (1. / r[atPole]) * dUdr[atPole] * p[atPole, 2]

        return gx, gy, gz

    # Compute normalized associated legendre polynomials
    P, scaleFactor = loc_gravLegendre(phic, maxdeg)

    # Compute gravity in ECEF coordinates
    gx, gy, gz = loc_gravityPCPF(p, maxdeg, P, C[0:maxdeg+1, 0:maxdeg+1],
                                S[0:maxdeg+1, 0:maxdeg+1], smlambda,
                                cmlambda, GM, Re, r, scaleFactor)

    return np.array([gx,gy,gz])

if __name__ == '__main__':
    grav_odp([7e6,0,0])