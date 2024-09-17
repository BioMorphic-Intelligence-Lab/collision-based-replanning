import numpy as np
from scipy.spatial.transform import Rotation

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

    def get_orthogonal_unit_vec(self, vec):
        v = np.array([0, 0, 1], dtype=float)
        if np.isclose(vec, [1, 0, 0]).all():
            v = np.array([0, 1, 0])
        elif np.isclose(vec, [0, 1, 0]).all():
            v = np.array([0, 0, 1])
        elif np.isclose(vec, [0, 0, 1]).all():
            v = np.array([1, 0, 0])

        ortho = np.cross(vec, v)

        return ortho / np.linalg.norm(ortho)

    def field(self, x, y, z,
              kappa=10.0):

        # Find minimum distance trajectory point
        t_min = self.find_nearest_t(np.array([x, y, z]))

        # Now compute the normalized distance vector
        distance_vector = (self.f(t_min).flatten() - np.array([x,y,z]))
        distance_norm = np.linalg.norm(distance_vector)

        # Find traj derivative at that index
        derivative = self.df(t_min).flatten()

        gradient_x = 0
        gradient_y = 0
        gradient_z = 0

        if distance_norm > 0:
            gradient_x += (1.0 / (kappa * distance_norm) * derivative[0]
                        + kappa * distance_vector[0])
            gradient_y += (1.0 / (kappa * distance_norm) * derivative[1]
                        + kappa * distance_vector[1])
            gradient_z += (1.0 / (kappa * distance_norm) * derivative[2]
                        + kappa * distance_vector[2])
        elif t_min >= 1:
            gradient_x = kappa * distance_vector[0]
            gradient_y = kappa * distance_vector[1]
            gradient_z = kappa * distance_vector[2]
        else:            
            gradient_x += derivative[0]
            gradient_y += derivative[1]
            gradient_z += derivative[2]

        # Find the point source contribution of the collision point
        coll_contrib_x = 0
        coll_contrib_y = 0
        coll_contrib_z = 0
        for collision in self.collisions:
            vec_x = x - collision[0]
            vec_y = y - collision[1]
            vec_z = z - collision[2]

            vec_norm = np.sqrt(vec_x**2 + vec_y**2 + vec_z**2)

            if vec_norm > 100.0:
                continue

            coll_contrib_x += kappa / vec_norm**3 * vec_x / vec_norm
            coll_contrib_y += kappa / vec_norm**3 * vec_y / vec_norm
            coll_contrib_z += kappa / vec_norm**3 * vec_z / vec_norm

        # Project the point source contribution into the plane defined
        # by the trajectory velocity as its normal. That way we do not interfere
        # with the travelling velocity along the trajectory and only add contributions
        # laterally to that velocity

        # Find the unit vector of the desired velocity
        vel_dir = np.array(derivative) / np.linalg.norm(derivative)
        # Avoid numerical issues if the connection vector and the
        # velocity vector are exactly parallel
        coll_contrib = np.array([coll_contrib_x, coll_contrib_y, coll_contrib_z])
        if np.linalg.norm(coll_contrib) > 0:
            normal_grad = np.dot([coll_contrib_x,
                                  coll_contrib_y,
                                  coll_contrib_z], vel_dir) * vel_dir
            if np.abs(np.dot(coll_contrib, vel_dir) / np.linalg.norm(coll_contrib) - 1) < 1e-5:
                normal_dir = normal_grad / np.linalg.norm(normal_grad)
                ortho_dir_1 = self.get_orthogonal_unit_vec(normal_dir)
                # Randomly rotate the direction to ensure distribution over all orthogonal
                # directions that are possible
                ortho_dir_1 = (
                    Rotation.from_rotvec(
                        normal_dir * np.random.uniform(-np.pi, np.pi)
                    ).as_matrix() @ ortho_dir_1
                )
                coll_contrib_x = np.linalg.norm(coll_contrib) * ortho_dir_1[0]
                coll_contrib_y = np.linalg.norm(coll_contrib) * ortho_dir_1[1]
                coll_contrib_z = np.linalg.norm(coll_contrib) * ortho_dir_1[2]
            else:
                coll_contrib_x = coll_contrib_x - normal_grad[0]
                coll_contrib_y = coll_contrib_y - normal_grad[1]
                coll_contrib_z = coll_contrib_z - normal_grad[2]

        # Add the contribution to the gradient
        gradient_x += coll_contrib_x
        gradient_y += coll_contrib_y
        gradient_z += coll_contrib_z

        # Normalize everything to maintain the same speed everywhere
        gradient_norm = np.sqrt(gradient_x**2 + gradient_y**2 + gradient_z**2)
        gradient_x = gradient_x / gradient_norm * self.speed
        gradient_y = gradient_y / gradient_norm * self.speed
        gradient_z = gradient_z / gradient_norm * self.speed

        return gradient_x, gradient_y, gradient_z