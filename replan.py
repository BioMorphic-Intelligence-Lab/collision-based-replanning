import numpy as np
import matplotlib.pyplot as plt

from gradient_field import GradientField

import matplotlib

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"

def main():
    # Number of trajectory points
    n = 1000

    # Init the time vector
    t = np.linspace(0, 1, n)

    # Define the polynomial coefficents for the 2d trajectory
    coeff = np.array([[0, 10, 0, 0, 0, 0], # x axis
                      [0*10, 18.11667*10,
                    -13.25*10**2, 3.59375*10**3,
                    -0.40625*10**4, 0.01614583*10**5]], # y axis
                     dtype=float) 
    
    # Init the gradient field class
    gf = GradientField(t=t, coeff=coeff)

    # Add random collisions along the path
    collisions_loc = np.zeros([4, 2])
    for i in range(4):
        collisions_loc[i, :] =  gf.f(0.2 * (i+1))
        gf.add_collision(collisions_loc[i, :])
    
    # Generate the trajectory
    range_x = (min(gf.traj[0,:])-1, max(gf.traj[0,:])+1)
    range_y = (min(gf.traj[1,:])-1, max(gf.traj[1,:])+1)

    # Plot gradient field
    resolution = 40
    # Meshgrid 
    x, y = np.meshgrid(np.linspace(range_x[0], range_x[1], resolution),  
                       np.linspace(range_y[0], range_y[1], resolution)) 

    u, v = np.zeros((resolution, resolution), dtype=float), \
           np.zeros((resolution, resolution), dtype=float)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            u1, v1 = gf.field(x[i, j], y[i, j])
            u[i, j] = u1
            v[i, j] = v1
    
    # Find trajectories from all possible starting points 
    alt_start_trajs = np.zeros([resolution, resolution, n, 2])
    for i in range(0, x.shape[0], 5):
        for j in range(0, y.shape[0], 5):
            alt_start_trajs[i, j, 0, 0] = x[i,j]
            alt_start_trajs[i, j, 0, 1] = y[i,j]
            for k in range(1, n):
                dx, dy = gf.field(alt_start_trajs[i,j, k - 1, 0],
                                  alt_start_trajs[i,j, k - 1, 1],
                                  gamma=0.5)
                alt_start_trajs[i, j, k, :] = alt_start_trajs[i, j, k-1, :]  + 10.0/n * np.array([dx, dy])
    
    fig, ax = plt.subplots()
    ax.plot(gf.traj[0, :], gf.traj[1, :], label="Original Trajectory", color="black")
    ax.quiver(x,y,u,v, label="Gradient Field", color="grey")
    ax.scatter(collisions_loc[:, 0], collisions_loc[:, 1], marker="x", color="#D63000", label="Collision Location")
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.set_xlim(range_x)
    ax.set_ylim(range_y)

    ax.legend()

    for i in range(0, x.shape[0], 5):
        for j in range(0, y.shape[0], 5):
            ax.plot(alt_start_trajs[i, j, :, 0],
                    alt_start_trajs[i, j, :, 1],
                    zorder=0,
                    linestyle="--",
                    color="#00A6D6")
    
    fig.set_size_inches((8, 20))
    plt.savefig("collision_based_replanning.png", bbox_inches='tight', dpi=500)

if __name__ == "__main__":
    main()