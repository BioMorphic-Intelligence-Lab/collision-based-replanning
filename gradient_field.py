import numpy as np

class GradientField(object):
    def __init__(self, t, coeff, collisions=[]) -> None:
        self.t = t
        self.coeff = coeff
        self.traj = self.generate_traj(t=self.t, coeff=self.coeff)
        self.collisions = collisions
        
    def generate_traj(self, t, coeff):
        return np.array([np.sum([coeff[0, i]*t**i for i in range(len(coeff[0]))], axis=0),
                        np.sum([coeff[1, i]*t**i for i in range(len(coeff[1]))], axis=0)])

    def traj_derivative(self, index):
        # Extract time value
        ti = self.t[index]
        return np.array([np.sum([i * self.coeff[0, i] * ti**(i-1) for i in range(1, len(self.coeff[0]))], axis=0),
                         np.sum([i * self.coeff[1, i] * ti**(i-1) for i in range(1, len(self.coeff[1]))], axis=0)])


    def add_collision(self, collision: tuple) -> None:
        self.collisions.append(collision)

    def field(self, x, y, gamma=0.99):

        # Find minimum distance trajectory point
        distances = (self.traj[0, :] - x)**2 + (self.traj[1, :] - y)**2
        index = np.argmin(distances)

        # Now compute the normalized distance vector
        distance_vector = (self.traj[:, index] - np.array([x,y]))
        distance_norm = np.linalg.norm(distance_vector)
        # Find traj derivative at that index
        derivative = self.traj_derivative(index)
        
        gradient_x = np.exp(-distance_norm)*derivative[0] + distance_vector[0]
        gradient_y = np.exp(-distance_norm)*derivative[1] + distance_vector[1]
        gradient_norm = np.sqrt(gradient_x**2 + gradient_y**2)
        ortho_vec = np.array([gradient_y, -gradient_x]) / gradient_norm

        for collision in self.collisions:
            vec_x = x - collision[0]
            vec_y = y - collision[1]
            vec_norm = np.sqrt(vec_x**2 + vec_y**2)

            direction = np.sign(np.dot([vec_x, vec_y], ortho_vec))

            gradient_mag = np.exp(-gamma * vec_norm)

            gradient_x += direction * ortho_vec[0] * gradient_mag
            gradient_y += direction * ortho_vec[1] * gradient_mag

        return gradient_x, gradient_y
    
    def augment_traj_delta(self, dx, dy, x, y, scale=1):
        grad_x, grad_y = self.field(x, y)
        return dx + scale*grad_x, dy + scale*grad_y