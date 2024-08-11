import numpy as np
from scipy.optimize import minimize

class GradientField(object):
    def __init__(self, t, coeff, speed=1.0, collisions=[]) -> None:
        self.t = t
        self.coeff = coeff
        self.traj = self.f(t=self.t)
        self.collisions = collisions
        self.speed = speed

    def f(self, t):
        return np.array([np.sum([self.coeff[0, i]*t**i for i in range(len(self.coeff[0]))], axis=0),
                         np.sum([self.coeff[1, i]*t**i for i in range(len(self.coeff[1]))], axis=0),
                         np.sum([self.coeff[2, i]*t**i for i in range(len(self.coeff[2]))], axis=0)])

    def df(self, t):
        return np.array([np.sum([i * self.coeff[0, i] * t**(i-1) for i in range(1, len(self.coeff[0]))], axis=0),
                         np.sum([i * self.coeff[1, i] * t**(i-1) for i in range(1, len(self.coeff[1]))], axis=0),
                         np.sum([i * self.coeff[2, i] * t**(i-1) for i in range(1, len(self.coeff[2]))], axis=0)])
    
    def ddf(self, t):
        return np.array([np.sum([i * (i-1) * self.coeff[0, i] * t**(i - 2) for i in range(2, len(self.coeff[0]))], axis=0),
                         np.sum([i * (i-1) * self.coeff[1, i] * t**(i - 2) for i in range(2, len(self.coeff[1]))], axis=0),
                         np.sum([i * (i-1) * self.coeff[2, i] * t**(i - 2) for i in range(2, len(self.coeff[2]))], axis=0)])
   
    def add_collision(self, collision: tuple) -> None:
        self.collisions.append(collision)
     
    def find_nearest_t(self, x, iter=2):

        distances2 = (self.traj[0, :] - x[0])**2 + (self.traj[1, :] - x[1])**2 + (self.traj[2, :] - x[2])**2
        t = self.t[np.argmin(distances2)]
        
        #t = 0.5

        dg = lambda t: 2*(np.sum((self.f(t) - x) * self.df(t)))
        ddg = lambda t: 2*(np.sum(self.df(t)**2 + (self.f(t) - x) * self.ddf(t)))

        for _ in range(iter):
           t = t - dg(t) / ddg(t)

        return min(max(t, 0), 1)


    def field(self, x, y, z, gamma=0.925):

        # Find minimum distance trajectory point
        t_min = self.find_nearest_t(np.array([x, y, z]))

        # Now compute the normalized distance vector
        distance_vector = (self.f(t_min).flatten() - np.array([x,y,z]))
        distance_norm = np.linalg.norm(distance_vector)

        # Find traj derivative at that index
        derivative = self.df(t_min).flatten()

        if distance_norm > 0:
            gradient_x = (np.exp(-1*distance_norm)*derivative[0] 
                        + (np.exp(10*distance_norm)-1)*distance_vector[0] / distance_norm)
            gradient_y = (np.exp(-1*distance_norm)*derivative[1] 
                        + (np.exp(10*distance_norm)-1)*distance_vector[1] / distance_norm)
            gradient_z = (np.exp(-1*distance_norm)*derivative[2] 
                        + (np.exp(10*distance_norm)-1)*distance_vector[2] / distance_norm)
        else:            
            gradient_x = derivative[0]
            gradient_y = derivative[1]
            gradient_z = derivative[2]

        for collision in self.collisions:
            vec_x = x - collision[0]
            vec_y = y - collision[1]
            vec_z = z - collision[2]

            vec_norm = np.sqrt(vec_x**2 + vec_y**2 + vec_z**2)

            gradient_x += 0.4 / vec_norm**2 * vec_x / vec_norm  
            gradient_y += 0.4 / vec_norm**2 * vec_y / vec_norm  
            gradient_z += 0.4 / vec_norm**2 * vec_z / vec_norm

        # Normalize everything to maintain the same speed everywhere
        gradient_norm = np.sqrt(gradient_x**2 + gradient_y**2 + gradient_z**2)
        gradient_x = gradient_x / gradient_norm * self.speed
        gradient_y = gradient_y / gradient_norm * self.speed
        gradient_z = gradient_z / gradient_norm * self.speed

        return gradient_x, gradient_y, gradient_z