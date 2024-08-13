import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation
from gradient_field import GradientField
from sinusoidal_gradient_field import SinusoidalGradientField

import matplotlib

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"

def main():
    # Number of trajectory points
    n = 50

    # Init the time vector
    t = np.linspace(0, 1, n)

    # Define the polynomial coefficents for the 2d trajectory
    coeff = np.array(
        [[2.0, 2*np.pi, 0],
         [2.0, 2*np.pi, np.pi / 2],
         [1.0, 0, 0]],
        #[[-2.0, 4.0],
        # [0, 0.0],
        # [1,   0]],
        #[[0, 10, 0, 0, 0, 0], # x axis
        # [0*10, 18.11667*10, -13.25*10**2, 3.59375*10**3, -0.40625*10**4, 0.01614583*10**5], # y-axis
        # [2, 0, 0, 0, 0, 0]], # z axis
    dtype=float) 
    
    # Init the gradient field class
    gf = SinusoidalGradientField(t=t, coeff=coeff)

    # Add random collisions along the path
    collisions_loc = np.zeros([2, 3])
    for i in range(0, 1, 2):
        loc = gf.f(0.3 * (i + 1))
        der = gf.df(0.3 * (i + 1))
        rot = Rotation.from_rotvec(np.array([0, 0, 1]) * np.arctan2(der[1], der[0])).as_matrix()
        collisions_loc[i, :] =  loc - rot @ np.array([0, 0.22, 0])
        collisions_loc[i+1, :] =  loc + rot @ np.array([0, 0.22, 0])
        gf.add_collision(collisions_loc[i, :])
        gf.add_collision(collisions_loc[i+1, :])
    
    # Generate the trajectory
    maxVal = max(np.abs(gf.traj.flatten()))

    range_x = (-np.abs(maxVal)-0.25, maxVal+0.25)
    range_y = (-np.abs(maxVal)-0.25, maxVal+0.25)
    range_z = (-np.abs(maxVal)-0.25, maxVal+0.25)

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
    n_sample_trajs = 100
    alt_start_trajs = np.zeros([n_sample_trajs, 3, 10 * n])

    for i in range(n_sample_trajs):
        x0 = np.random.uniform(low=range_x[0], high=range_x[1])
        y0 = np.random.uniform(low=range_y[0], high=range_y[1])
        z0 = np.random.uniform(low=range_z[0], high=range_z[1])
        alt_start_trajs[i, 0, 0] = x0
        alt_start_trajs[i, 1, 0] = y0
        alt_start_trajs[i, 2, 0] = z0
        for t in range(1, 10 * n):
            dx, dy, dz = gf.field(alt_start_trajs[i,0, t - 1],
                                  alt_start_trajs[i,1, t - 1],
                                  alt_start_trajs[i,2, t - 1])
            alt_start_trajs[i, :, t] = alt_start_trajs[i, :, t-1]  + 1.0/n * np.array([dx, dy, dz])
    
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(gf.traj[0, :], gf.traj[1, :], gf.traj[2, :],
            label="Original Trajectory", color="black", alpha=0.8)
    ax.quiver(x,y,z,
              u,v,w,
              label="Gradient Field", color="grey",
              length=0.05 * np.abs(2 * maxVal), normalize=True)
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

    for i in range(n_sample_trajs):
        ax.plot(alt_start_trajs[i, 0, :],
                alt_start_trajs[i, 1, :],
                alt_start_trajs[i, 2, :],
                zorder=2,
                linestyle="--",
                color="#00A6D6")
                
    fig.set_size_inches((8, 20))
    plt.show()
    #plt.savefig("collision_based_replanning.png", bbox_inches='tight', dpi=500)

if __name__ == "__main__":
    main()
