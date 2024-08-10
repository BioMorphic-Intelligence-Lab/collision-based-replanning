import numpy as np
import matplotlib.pyplot as plt

from gradient_field import GradientField
from sinusoidal_gradient_field import SinusoidalGradientField

import matplotlib

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"

def main():
    # Number of trajectory points
    n = 1000

    # Init the time vector
    t = np.linspace(0, 1, n)

    # Define the polynomial coefficents for the 2d trajectory
    coeff = np.array(
        [[1, 2*np.pi, 0],
         [1, 2*np.pi, np.pi / 2],
         [2, 0, 0]],
        #[[0, 10],
        # [0, 10],
        # [2, 0]],
        #[[0, 10, 0, 0, 0, 0], # x axis
        # [0*10, 18.11667*10, -13.25*10**2, 3.59375*10**3, -0.40625*10**4, 0.01614583*10**5], # y-axis
        # [2, 0, 0, 0, 0, 0]], # z axis
    dtype=float) 
    
    # Init the gradient field class
    gf = SinusoidalGradientField(t=t, coeff=coeff)

    # Add random collisions along the path
    collisions_loc = np.zeros([1, 3])
    for i in range(1):
        collisions_loc[i, :] =  gf.f(0.3 * (i+1))
        gf.add_collision(collisions_loc[i, :])
    
    # Generate the trajectory
    minVal = min(gf.traj.flatten())
    maxVal = max(gf.traj.flatten())
    range_x = (minVal-0.1, maxVal+0.1)
    range_y = (minVal-0.1, maxVal+0.1)
    range_z = (minVal-0.1, maxVal+0.1)

    # Plot gradient field
    resolution = 6
    # Meshgrid 
    x, y, z = np.meshgrid(np.linspace(range_x[0], range_x[1], resolution),  
                          np.linspace(range_y[0], range_y[1], resolution),
                          np.linspace(range_z[0], range_z[1], resolution)) 

    u, v, w = np.zeros((resolution, resolution, resolution), dtype=float), \
              np.zeros((resolution, resolution, resolution), dtype=float), \
              np.zeros((resolution, resolution, resolution), dtype=float)
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                u1, v1, w1 = gf.field(x[i, j, k], y[i, j, k], z[i, j, k])
                u[i, j, k] = u1
                v[i, j, k] = v1
                w[i, j, k] = w1
    
    # Find trajectories from all possible starting points 
    alt_start_trajs = np.zeros([20, 3, 10*n])
    for i in range(20):
        x0 = np.random.uniform(low=range_x[0], high=range_x[1])
        y0 = np.random.uniform(low=range_y[0], high=range_y[1])
        z0 = np.random.uniform(low=range_z[0], high=range_z[1])
        alt_start_trajs[i, 0, 0] = x0
        alt_start_trajs[i, 1, 0] = y0
        alt_start_trajs[i, 2, 0] = z0
        for t in range(1, 10*n):
            dx, dy, dz = gf.field(alt_start_trajs[i,0, t - 1],
                                  alt_start_trajs[i,1, t - 1],
                                  alt_start_trajs[i,2, t - 1])
            alt_start_trajs[i, :, t] = alt_start_trajs[i, :, t-1]  + 1.0/n * np.array([dx, dy, dz])
    
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(gf.traj[0, :], gf.traj[1, :], gf.traj[2, :],
            label="Original Trajectory", color="black")
    ax.quiver(x,y,z,
              u,v,w,
              label="Gradient Field", color="grey",
              length=0.05 * np.abs(maxVal - minVal), normalize=True)
    ax.scatter(collisions_loc[:, 0],
               collisions_loc[:, 1],
               collisions_loc[:, 2], marker="x", color="#D63000", label="Collision Location")
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.set_zlabel(r"$z$ [m]")
    ax.set_xlim(range_x)
    ax.set_ylim(range_y)
    ax.set_zlim(range_z)

    ax.legend()

    for i in range(20):
        ax.plot(alt_start_trajs[i, 0, :],
                alt_start_trajs[i, 1, :],
                alt_start_trajs[i, 2, :],
                zorder=2,
                linestyle="--",
                color="#00A6D6")
                
    fig.set_size_inches((8, 20))
    plt.savefig("collision_based_replanning.png", bbox_inches='tight', dpi=500)

if __name__ == "__main__":
    main()