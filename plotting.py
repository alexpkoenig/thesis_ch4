"""

Module for plotting orbits.
Plotting functions accept the output of SciPy's solve_ivp() and are assumed to use SI base units.

"""

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.cm import ScalarMappable

from force_models import accelerations
from constants import Rearth


def get_and_rescale(i, arrows):
    vecs = np.array([arrow[i] for arrow in arrows])
    return vecs/np.mean(norm(vecs,axis=1))

def plot_eci_x(soln):
    # Your solution data: soln.y[0], soln.y[1], soln.y[2], and soln.t
    soln_y0 = soln.y[0]
    soln_y1 = soln.y[1]
    soln_y2 = soln.y[2]
    soln_t = soln.t

    indices = np.arange(0, len(soln_t), 5)

    arrows = [accelerations(t,s,return_extras=True) for t,s in zip(soln.t[indices], soln.y.T[indices])]

    grav = get_and_rescale(0, arrows)
    drag = get_and_rescale(1, arrows)
    srp = get_and_rescale(2, arrows)
    sungrav = get_and_rescale(3, arrows)
    moongrav = get_and_rescale(4, arrows)
    sun = get_and_rescale(5, arrows)
    moon = get_and_rescale(6, arrows)

    perturbs = [drag,srp,sungrav,moongrav]
    colors = ['tab:blue','tab:red','tab:orange','tab:grey']
    labels = ['drag', 'SRP', 'sun grav', 'moon grav']

    vector_scale = 1e6

    # Scale soln.t from 0 to 1
    scaled_t = (soln_t - np.min(soln_t)) / (np.max(soln_t) - np.min(soln_t))

    # Create a 1x3 grid of subplots
    fig = plt.figure(figsize=(8,5),dpi=150)

    circle = matplotlib.patches.Circle((0, 0), Rearth, color='blue', alpha=0.2)
    plt.gca().add_artist(circle)

    # Create scatter plots for each pair of soln_y0, soln_y1, and soln_y2
    sc0 = plt.scatter(soln_y0, soln_y1, c=scaled_t, cmap='plasma',s=2)
    for i in range(len(perturbs)):
        plt.quiver(soln_y0[indices], soln_y1[indices], perturbs[i][:,0], perturbs[i][:,1], angles='xy', scale_units='xy', scale=1/vector_scale, color=colors[i],label=labels[i])#plt.cm.plasma(scaled_t[indices]))
    plt.quiver(np.zeros_like(indices),np.zeros_like(indices),sun[:,0],sun[:,1], angles='xy', scale_units='xy', scale=0.1/vector_scale,color='tab:orange',label='Sun')
    plt.quiver(np.zeros_like(indices),np.zeros_like(indices),moon[:,0],moon[:,1], angles='xy', scale_units='xy', scale=0.1/vector_scale, color='tab:grey',label='Moon')
    plt.gca().set_xlabel("x [m]")
    plt.gca().set_ylabel("y [m]")
    plt.gca().set_aspect('equal')
    plt.gca().set_xlim([-8e6,8e6])
    plt.gca().set_ylim([-8e6,8e6])

    plt.legend(loc='center')

    # Create a custom mappable object with the original soln.t range
    mappable = ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=np.min(soln_t), vmax=np.max(soln_t)))

    # Create the colorbar with the custom mappable object
    cbar = fig.colorbar(mappable, pad=0.01)
    cbar.set_label('time [s]')

    plt.savefig('orbit.png')
    # Display the plot
    plt.show()

def plot_eci(soln):
    # Your solution data: soln.y[0], soln.y[1], soln.y[2], and soln.t
    soln_y0 = soln.y[0]
    soln_y1 = soln.y[1]
    soln_y2 = soln.y[2]
    soln_t = soln.t

    indices = np.arange(0, len(soln_t), 5)

    arrows = [accelerations(t,s,return_extras=True) for t,s in zip(soln.t[indices], soln.y.T[indices])]

    grav = get_and_rescale(0, arrows)
    drag = get_and_rescale(1, arrows)
    srp = get_and_rescale(2, arrows)
    sungrav = get_and_rescale(3, arrows)
    moongrav = get_and_rescale(4, arrows)
    sun = get_and_rescale(5, arrows)
    moon = get_and_rescale(6, arrows)

    perturbs = [drag,srp,sungrav,moongrav]
    colors = ['tab:blue','tab:red','tab:orange','tab:grey']

    vector_scale = 1e6

    # Scale soln.t from 0 to 1
    scaled_t = (soln_t - np.min(soln_t)) / (np.max(soln_t) - np.min(soln_t))

    # Create a 1x3 grid of subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.set_dpi(150)

    for ax in axs:
        circle = matplotlib.patches.Circle((0, 0), Rearth, color='blue', alpha=0.2)
        ax.add_artist(circle)

    # Create scatter plots for each pair of soln_y0, soln_y1, and soln_y2
    sc0 = axs[0].scatter(soln_y0, soln_y1, c=scaled_t, cmap='plasma',s=2)
    for i in range(len(perturbs)):
        axs[0].quiver(soln_y0[indices], soln_y1[indices], perturbs[i][:,0], perturbs[i][:,1], angles='xy', scale_units='xy', scale=1/vector_scale, color=colors[i])#plt.cm.plasma(scaled_t[indices]))
    axs[0].quiver(np.zeros_like(indices),np.zeros_like(indices),sun[:,0],sun[:,1], angles='xy', scale_units='xy', scale=0.01/vector_scale)
    axs[0].quiver(np.zeros_like(indices),np.zeros_like(indices),moon[:,0],moon[:,1], angles='xy', scale_units='xy', scale=0.01/vector_scale, color='grey')
    axs[0].set_xlabel("x [m]")
    axs[0].set_ylabel("y [m]")
    axs[0].set_aspect('equal')
    axs[0].set_xlim([-8e6,8e6])
    axs[0].set_ylim([-8e6,8e6])

    sc1 = axs[1].scatter(soln_y0, soln_y2, c=scaled_t, cmap='plasma',s=2)
    for i in range(4):
        axs[1].quiver(soln_y0[indices], soln_y2[indices], perturbs[i][:,0], perturbs[i][:,2], angles='xy', scale_units='xy', scale=1/vector_scale, color=colors[i])#plt.cm.plasma(scaled_t[indices]))
    axs[1].quiver(np.zeros_like(indices),np.zeros_like(indices),sun[:,0],sun[:,2], angles='xy', scale_units='xy', scale=0.01/vector_scale)
    axs[1].quiver(np.zeros_like(indices),np.zeros_like(indices),moon[:,0],moon[:,2], angles='xy', scale_units='xy', scale=0.01/vector_scale, color='grey')
    axs[1].set_xlabel("x [m]")
    axs[1].set_ylabel("z [m]")
    axs[1].set_aspect('equal')
    axs[1].set_xlim([-8e6,8e6])
    axs[1].set_ylim([-8e6,8e6])

    sc2 = axs[2].scatter(soln_y1, soln_y2, c=scaled_t, cmap='plasma',s=2)
    for i in range(4):
        axs[2].quiver(soln_y1[indices], soln_y2[indices], perturbs[i][:,1], perturbs[i][:,2], angles='xy', scale_units='xy', scale=1/vector_scale, color=colors[i])#plt.cm.plasma(scaled_t[indices]))
    axs[2].quiver(np.zeros_like(indices),np.zeros_like(indices),sun[:,1],sun[:,2], angles='xy', scale_units='xy', scale=0.01/vector_scale)
    axs[2].quiver(np.zeros_like(indices),np.zeros_like(indices),moon[:,1],moon[:,2], angles='xy', scale_units='xy', scale=0.01/vector_scale, color='grey')
    axs[2].set_xlabel("y [m]")
    axs[2].set_ylabel("z [m]")
    axs[2].set_aspect('equal')
    axs[2].set_xlim([-8e6,8e6])
    axs[2].set_ylim([-8e6,8e6])

    # Create a custom mappable object with the original soln.t range
    mappable = ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=np.min(soln_t), vmax=np.max(soln_t)))

    # Create the colorbar with the custom mappable object
    cbar = fig.colorbar(mappable, ax=axs.ravel().tolist(), pad=0.01)
    cbar.set_label('time [s]')

    # Display the plot
    plt.savefig('orbit.png',dpi=500)
    # plt.show()

def plot_forces(soln):
    indices = np.arange(0, len(soln.t), 1)
    arrows = [accelerations(t,s,return_extras=True) for t,s in zip(soln.t[indices], soln.y.T[indices])]
    grav = get_and_rescale(0, arrows)
    drag = get_and_rescale(1, arrows)
    srp = get_and_rescale(2, arrows)
    plt.plot(soln.t[indices],norm(grav,axis=1),label='gravity')
    plt.plot(soln.t[indices],norm(drag,axis=1),label='drag')
    plt.plot(soln.t[indices],norm(srp,axis=1),label='srp')
    plt.legend()
    plt.show()
