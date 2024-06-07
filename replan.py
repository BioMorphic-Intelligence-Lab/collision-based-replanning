import numpy as np
import matplotlib.pyplot as plt

from gradient_field import GradientField


def main():
    # Init the time vector
    t = np.linspace(0, 10, 1000)

    # Define the polynomial coefficents for the 2d trajectory
    coeff = np.array([[0, 1, 0, 0, 0, 0], # x axis
                    [0, 18.11667,
                    -13.25, 3.59375,
                    -0.40625, 0.01614583]], dtype=float) # y axis
    
    # Define the collision locations
    collisions_loc = [(4.5, 4)]

    # Init the gradient field class
    gf = GradientField(t=t, coeff=coeff, collisions=collisions_loc)
    
    # Generate the trajectory
    range_x = (min(gf.traj[0,:]), max(gf.traj[0,:]))
    range_y = (min(gf.traj[1,:]), max(gf.traj[1,:]))

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
    
    # Find new trajector
    traj_new = np.zeros_like(gf.traj)
    traj_new[:, 0] = gf.traj[:, 0]
    for i in range(1, len(t)):
        dx, dy = gf.traj[0, i]-gf.traj[0,i-1], gf.traj[1, i]-gf.traj[1,i-1] 
        dx_new, dy_new = gf.augment_traj_delta(dx=dx, dy=dy, x=gf.traj[0, i-1], y=gf.traj[1, i-1])
        traj_new[0, i] = gf.traj[0, i] + dx_new
        traj_new[1, i] = gf.traj[1, i] + dy_new

    fig, ax = plt.subplots(2)
    ax[0].plot(t, gf.traj[0,:], label="x", color="blue")
    ax[0].plot(t, gf.traj[1,:], label="y", color="orange")
    ax[0].plot(t, traj_new[0,:],":", label="x replanned", color="blue")
    ax[0].plot(t, traj_new[1,:],":", label="y replanned", color="orange")
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Distance [m]")
    ax[0].set_aspect("auto")
    ax[0].legend()
    
    ax[1].plot(gf.traj[0, :], gf.traj[1, :], label="original", color="blue")
    ax[1].plot(traj_new[0, :], traj_new[1, :], ":", label="replanned", color="green")
    for collision in gf.collisions:
        ax[1].scatter(collision[0], collision[1], marker="x", color="red", label="Collision Location")
    ax[1].quiver(x,y,u,v, label="Gradient Field")
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("x [m]")
    ax[1].set_ylabel("y [m]")
    ax[1].legend()

    fig.set_size_inches((8, 20))
    plt.savefig("collision_based_replanning.png", bbox_inches='tight', dpi=500)

if __name__ == "__main__":
    main()