import numpy as np
from scipy.optimize import minimize

class GradientField(object):
    def __init__(self, t, coeff, collisions=[]) -> None:
        self.t = t
        self.coeff = coeff
        self.traj = self.f(t=self.t)
        self.collisions = collisions

    def f(self, t):
        return np.array([np.sum([self.coeff[0, i]*t**i for i in range(len(self.coeff[0]))], axis=0),
                        np.sum([self.coeff[1, i]*t**i for i in range(len(self.coeff[1]))], axis=0)])

    def df(self, t):
        return np.array([np.sum([i * self.coeff[0, i] * t**(i-1) for i in range(1, len(self.coeff[0]))], axis=0),
                         np.sum([i * self.coeff[1, i] * t**(i-1) for i in range(1, len(self.coeff[1]))], axis=0)])
   
    def add_collision(self, collision: tuple) -> None:
        self.collisions.append(collision)
     
    def field(self, x, y, gamma=0.99):

        # Find minimum distance trajectory point
        distances2 = (self.traj[0, :] - x)**2 + (self.traj[1, :] - y)**2
        t_min = self.t[np.argmin(distances2)]

        # Now compute the normalized distance vector
        distance_vector = (self.f(t_min).flatten() - np.array([x,y]))
        distance_norm = np.linalg.norm(distance_vector)
        # Find traj derivative at that index
        derivative = self.df(t_min).flatten()
        
        gradient_x = np.exp(-8 * distance_norm)*derivative[0] + np.exp(0.1 * distance_norm)*distance_vector[0] / distance_norm
        gradient_y = np.exp(-8 * distance_norm)*derivative[1] + np.exp(0.1 * distance_norm)*distance_vector[1] / distance_norm

        gradient_norm = np.sqrt(gradient_x**2 + gradient_y**2)
        ortho_vec = np.array([gradient_y, -gradient_x]) / gradient_norm

        for collision in self.collisions:
            vec_x = x - collision[0]
            vec_y = y - collision[1]
            vec_norm = np.sqrt(vec_x**2 + vec_y**2)

            direction = np.sign(np.dot([vec_x, vec_y], ortho_vec))

            gradient_mag = 3 * np.exp(-gamma * vec_norm)

            gradient_x += direction * ortho_vec[0] * gradient_mag
            gradient_y += direction * ortho_vec[1] * gradient_mag

        return gradient_x, gradient_y