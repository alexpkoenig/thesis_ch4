"""

Module implementing the Extended Kalman Filter (EKF).
Function names map to those defined in Chapter 4 of the thesis.

"""

import numpy as np
from numpy.linalg import norm
import scipy.integrate
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.linalg import eigh
from numba import jit,njit

from constants import c, s_1, s_2, s_3, Cd
from coordinates import ecef_eci, eci2ecef
from force_models import dynamics, dynamics_full, accelerations
from observations import range_and_rate, alt_and_dec
from symbolic_math import evaluateH, evaluateHr, evaluateHo, evaluateA

# Assumes ECI station_state
def G_r(X, t, station_state):
    return range_and_rate(t, X, station_state, using_ECI = True)

def H_r(X, t, station_state):
    return evaluateHr(X, station_state)

def G_o(X, t, station_state):
    return alt_and_dec(X, station_state)

def H_o(X, t, station_state):
    return evaluateHo(X, station_state)

def F(t, X):
    # print(t)
    return dynamics_full(t, X, [1,'cannonball','cannonball',1,1])

def A(X, t):
    accels = accelerations(t,X[:6], use_alt_grav='2body', return_extras=True)
    rsun = accels[5]
    rmoon = accels[6]
    A = evaluateA(X, rsun, rmoon)
    return A

def Q(dt, sigX, sigY, sigZ):
    return dt**2 * np.array([
        [dt**2/4 * sigX**2, 0, 0, dt/2*sigX**2, 0, 0, 0],
        [0, dt**2/4 * sigY**2, 0, 0, dt/2*sigY**2, 0, 0],
        [0, 0, dt**2/4 * sigZ**2, 0, 0, dt/2*sigZ**2, 0],
        [dt/2 * sigX**2, 0, 0, sigX**2, 0, 0, 0],
        [0, dt/2 * sigY**2, 0, 0, sigY**2, 0, 0],
        [0, 0, dt/2 * sigZ**2, 0, 0, sigZ**2, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ])

def Rot_BF2ECI(s):
    Rhat = s[:3]/norm(s[:3])
    Nhat = np.cross(s[:3], s[3:6]) / norm(np.cross(s[:3], s[3:6]))
    That = np.cross(Nhat, Rhat)
    return np.vstack([Rhat, That, Nhat]).T

def Rot_ECI2BF(s):
    return Rot_BF2ECI(s).T

def residuals(prediction, data, t, plot=False, plottitle='residuals'):
    idx_1 = np.where(data[:,0] == 1)
    idx_2 = np.where(data[:,0] == 2)
    idx_3 = np.where(data[:,0] == 3)

    ti = min(t)

    obs1 = np.array([[1,t-ti,*G(s, t, 1)/1e3] for t,s in zip(t[idx_1], prediction[idx_1])])
    obs2 = np.array([[1,t-ti,*G(s, t, 2)/1e3] for t,s in zip(t[idx_2], prediction[idx_2])])
    obs3 = np.array([[1,t-ti,*G(s, t, 3)/1e3] for t,s in zip(t[idx_3], prediction[idx_3])])

    residuals = np.zeros_like(data)
    residuals[:,0] = data[:,0]
    residuals[:,1] = data[:,1]
    residuals[idx_1,2] = obs1[:,2] - data[idx_1,2]
    residuals[idx_2,2] = obs2[:,2] - data[idx_2,2]
    residuals[idx_3,2] = obs3[:,2] - data[idx_3,2]
    residuals[idx_1,3] = obs1[:,3] - data[idx_1,3]
    residuals[idx_2,3] = obs2[:,3] - data[idx_2,3]
    residuals[idx_3,3] = obs3[:,3] - data[idx_3,3]

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        fig.set_dpi(150)
        axs[0].scatter(obs1[:,1]/3600, residuals[idx_1,2], label='Station 1', s=4)
        axs[0].scatter(obs2[:,1]/3600, residuals[idx_2,2], label='Station 2', s=4)
        axs[0].scatter(obs3[:,1]/3600, residuals[idx_3,2], label='Station 3', s=4)
        axs[0].set_xlabel('t [hr]')
        axs[0].set_ylabel('range residual [km]')
        axs[0].legend()
        axs[1].scatter(obs1[:,1]/3600, residuals[idx_1,3], label='Station 1', s=4)
        axs[1].scatter(obs2[:,1]/3600, residuals[idx_2,3], label='Station 2', s=4)
        axs[1].scatter(obs3[:,1]/3600, residuals[idx_3,3], label='Station 3', s=4)
        axs[1].set_xlabel('t [hr]')
        axs[1].set_ylabel('range-rate residual [km/s]')
        fig.suptitle(plottitle)
        plt.tight_layout()
        plt.show()

    return residuals

def EKF(s_i, epoch, data, position_stddev=1e4, velocity_stddev=10, s1_R=None, s2_R=None, s3_R=None):
    # Initialization
    # X*_i-1
    X = np.array([*s_i,Cd])
    Xhat = X
    # P_i-1
    P = np.diag(np.concatenate([np.ones(3)*position_stddev**2,np.ones(3)*velocity_stddev**2,np.ones(1)*1e-3]))
    # Phi_(t0,t0)
    Phi = np.eye(7)
    # Epoch of initial state (s)
    t_0 = epoch * 86400
    t_prev = t_0

    # Set empty lists for updated positions, covariances, and STMs
    Xhats = []
    Phats = []
    Phis = []
    Ss = []

    # Loop over all the measurements in the data
    for i in tqdm(range(data.shape[0])):
        # print(data[i])
        meas_type = data[i,0]
        dt = data[i,2]  # Time passed from initial epoch at the time of the measurement (s)
        t_next = t_0 + dt  # Absolute time of measurement (s)
        station_id = data[i,1]
        station_state = data[i,9:]  # given in ECI in m and m/s
        
        # Read in the measurement for this observation
        Y = data[i,3:5]  # m, m/s
        R = (data[i,5:9]).reshape((2,2))

        # print(i, Y,R)
        # if i==0:
        if t_prev == t_next:
        #     # Assume the first observation occurs at t=t_0 and do not propagate.
        #     # But make sure this assumption is true with the following assertion!
            # print(t_next)
            # print(dt)
            # print(i)
            assert t_prev == t_next
        # Otherwise, propagate the state at the previous observation to this observation
        else:
            # print('dif')
            # print(t_next - t_prev)
            # print('t_next')
            # print(t_next)
            Phi = np.eye(7)
            flat_state = np.concatenate([Xhat,Phi.flatten()])
            t_eval = np.array([t_prev, t_next])
            solution = scipy.integrate.solve_ivp(
                F,
                t_span = [t_prev, t_next],
                t_eval = t_eval,
                y0 = flat_state,
                rtol=1e-10,
                atol=1e-12
            )
            X = solution.y[:7,-1]
            Phi = solution.y[7:,-1].reshape((7,7))  # STM from the previous measurement to this one
        
        # Time update:
        if t_prev != t_next:
            P = Phi @ P @ Phi.T + Q(t_next - t_prev, 1e-8, 1e-8, 1e-8)
        else:
            P = Phi @ P @ Phi.T
        t_prev = t_next
        
        # Intermediate computations
        # print('obs',X,t_next,station_state)
        if meas_type == 1:
            Htilde = H_r(X, t_next, station_state)
            y = Y - G_r(X, t_next, station_state)
        elif meas_type == 0:
            Htilde = H_o(X, t_next, station_state)
            y = Y - G_o(X, t_next, station_state)
        else:
            print("invalid measurement type")
            raise Exception
        S = Htilde @ P @ Htilde.T + R
        K = P @ Htilde.T @ np.linalg.inv(S)
        
        # Measurement update
        Xhat = X + K @ y
        I_KH = (np.eye(7) - K @ Htilde)
        Phat = I_KH @ P @ I_KH.T + K @ R @ K.T
        P = Phat

        # print('f',i, t_prev, t_next, dt, P, Xhat)
        # print('f',Htilde)

        # print(t_prev, t_next)
        
        # Store variables
        Xhats.append(Xhat)
        Phats.append(Phat)
        Phis.append(Phi)
        Ss.append(S)

        Phi = np.eye(7)
    
    return Xhats, Phats, Phis, Ss

def propagate_to_delivery(Xhat_DCO, Phat_DCO, t_DCO, t_delivery):
    Phi_DCO = np.eye(7)
    state_flattened = np.concatenate([Xhat_DCO, Phi_DCO.flatten()])

    solution = scipy.integrate.solve_ivp(
        F,
        t_span = [t_DCO, t_delivery],
        t_eval = [t_DCO, t_delivery],
        y0 = state_flattened,
        rtol=1e-10,
        atol=1e-12
    )
    
    delivery_state = solution.y[:6,-1]
    Phi_end = solution.y[7:,-1].reshape((7,7))
    delivery_cov = Phi_end @ Phat_DCO @ Phi_end.T + Q(t_delivery - t_DCO, 1e-9, 1e-9, 1e-9)
    
    return delivery_state, delivery_cov

def RTS_smoother(X_estimates, P_estimates, Phis, epoch, data):
    # Initialize lists to store smoothed state estimates and covariances
    X_smoothed = [X_estimates[-1]]
    P_smoothed = [P_estimates[-1]]

    # Iterate backward through the state estimates
    for i in reversed(tqdm(range(1, len(X_estimates)))):
        t = data[i,1] + epoch*86400
        t_prev = data[i-1,1] + epoch*86400
        X = X_estimates[i-1]
        P = P_estimates[i-1]
        X_next = X_estimates[i]
        P_next = P_estimates[i]

        Phi = Phis[i-1]
        Phi_next = Phis[i]

        # Compute smoother gain
        Q_ = Q(t-t_prev, 1, 1, 1)
        C = P @ Phi.T @ np.linalg.inv(Phi @ P @ Phi.T + Q_)

        # Update state and covariance estimates
        X_smooth = X + C @ (X_next - X_pred)
        P_smooth = P + C @ (P_next - Phi @ P @ Phi.T - Q_) @ C.T

        # Store the smoothed state and covariance estimates
        X_smoothed.insert(0, X_smooth)
        P_smoothed.insert(0, P_smooth)

    return np.array(X_smoothed), np.array(P_smoothed)


def RTS_smoother_2(X_estimates, P_estimates, Phis, epoch, data, sigma_accel=1e-9):
    # Initialize lists to store smoothed state estimates and covariances
    X_smoothed = [X_estimates[-1]]
    P_smoothed = [P_estimates[-1]]

    # Iterate backward through the state estimates
    for i in reversed(tqdm(range(1, len(X_estimates)))):
        t = data[i,1] + epoch*86400
        t_prev = data[i-1,1] + epoch*86400
        X = X_estimates[i-1]
        P = P_estimates[i-1]
        X_next = X_estimates[i]
        P_next = P_estimates[i]

        # Get the Phi matrix for the current step
        Phi = Phis[i-1]

        # Compute smoother gain
        Q_ = Q(t-t_prev, sigma_accel, sigma_accel, sigma_accel)
        C_ = P @ Phi.T @ np.linalg.inv(Phi @ P @ Phi.T + Q_)

        # Update state and covariance estimates
        X_pred = Phi @ X
        X_smooth = X + C_ @ (X_next - X_pred)
        P_smooth = P + C_ @ (P_next - Phi @ P @ Phi.T - Q_) @ C_.T

        # Store the smoothed state and covariance estimates
        X_smoothed.insert(0, X_smooth)
        P_smoothed.insert(0, P_smooth)

    return np.array(X_smoothed), np.array(P_smoothed)

def plot_covariances(cov_matrices, state_vectors):
    """
    Provide values in the form:
    cov_matrices = {letter:cov for letter,cov in zip(['A','B','C','F'],[delivery_covA, delivery_covB, delivery_covC, delivery_covF])}
    state_vectors = {letter:cov for letter,cov in zip(['A','B','C','F'],[delivery_stateA, delivery_stateB, delivery_stateC, delivery_stateF])}
    """
    def confidence_ellipse(ax,cov, mean, edgecolor, facecolor, label, nstd=3.0):
        """
        Create a plot of the covariance confidence ellipse of *x* and *y*.

        Parameters
        ----------
        x, y : array-like, shape (n, )
            Input data.

        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.

        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.

        **kwargs
            Forwarded to `~matplotlib.patches.Ellipse`

        Returns
        -------
        matplotlib.patches.Ellipse
        """
    #     if x.size != y.size:
    #         raise ValueError("x and y must be the same size")

    #     cov = np.cov(x, y)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensional dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                        facecolor=facecolor,edgecolor=edgecolor,label=label)
                        #facecolor=facecolor,
    #                       **kwargs)

        # Calculating the standard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * nstd
    #     mean_x = np.mean(x)
        mean_x = mean[0]

        # calculating the standard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * nstd
    #     mean_y = np.mean(y)
        mean_y = mean[1]

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ellipse
    
    def plot_cov_ellipse(ax,cov, mean, nstd=3, **kwargs):
        """
        Plots a covariance ellipse on a given axes.
        """
        vals, vecs = eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * nstd * np.sqrt(vals)
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)

        return ellipse

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = ['R','T','N']
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    for i, (x, y) in enumerate([(0, 1), (0, 2), (1, 2)]):
        ax = axes[i]
        xlim_min, xlim_max, ylim_min, ylim_max = None, None, None, None
        for letter, color in zip(cov_matrices.keys(), colors):
            state = state_vectors[letter]
            Q_ECI2BF = Rot_ECI2BF(state)
            cov_ECI = cov_matrices[letter][:3, :3]
            cov_BF = Q_ECI2BF @ cov_ECI @ Q_ECI2BF.T
            cov_matrix = cov_BF[[x, y]][:, [x, y]]
            mean = np.zeros(2)

            ellipse = plot_cov_ellipse(ax,cov_matrix, mean, nstd=3, edgecolor=color, facecolor='none', label=letter)
            ax.add_artist(ellipse)

            # Update the limits to include all ellipses
            if xlim_min is None or mean[0] - 3 * np.sqrt(cov_matrix[0, 0]) < xlim_min:
                xlim_min = mean[0] - 3 * np.sqrt(cov_matrix[0, 0])
            if xlim_max is None or mean[0] + 3 * np.sqrt(cov_matrix[0, 0]) > xlim_max:
                xlim_max = mean[0] + 3 * np.sqrt(cov_matrix[0, 0])
            if ylim_min is None or mean[1] - 3 * np.sqrt(cov_matrix[1, 1]) < ylim_min:
                ylim_min = mean[1] - 3 * np.sqrt(cov_matrix[1, 1])
            if ylim_max is None or mean[1] + 3 * np.sqrt(cov_matrix[1, 1]) > ylim_max:
                ylim_max = mean[1] + 3 * np.sqrt(cov_matrix[1, 1])

        # Set axes limits slightly larger than the largest ellipse
        ax.set_xlim(xlim_min - 0.1 * (xlim_max - xlim_min), xlim_max + 0.1 * (xlim_max - xlim_min))
        ax.set_ylim(ylim_min - 0.1 * (ylim_max - ylim_min), ylim_max + 0.1 * (ylim_max - ylim_min))

        ax.set_xlabel(labels[x])
        ax.set_ylabel(labels[y])
        ax.legend()

    plt.show()
