import numpy as np

from gradient_field import GradientField

class SinusoidalGradientField(GradientField):

    def f(self, t):
        """ f(t) = coeff[:, 0] + coeff[:, 1] * sin(coeff[:, 2] t + coeff[:, 3])"""
        return np.array([self.coeff[0, 0] + self.coeff[0, 1] * np.sin(self.coeff[0, 2]  * t + self.coeff[0, 3]),
                         self.coeff[1, 0] + self.coeff[1, 1] * np.sin(self.coeff[1, 2]  * t + self.coeff[1, 3]),
                         self.coeff[2, 0] + self.coeff[2, 1] * np.sin(self.coeff[2, 2]  * t + self.coeff[2, 3])])

    def df(self, t):
        return np.array([
            self.coeff[0, 1] * self.coeff[0, 2] * np.cos(self.coeff[0, 2] * t + self.coeff[0, 3]),
            self.coeff[1, 1] * self.coeff[1, 2] * np.cos(self.coeff[1, 2] * t + self.coeff[1, 3]),
            self.coeff[2, 1] * self.coeff[2, 2] * np.cos(self.coeff[2, 2] * t + self.coeff[2, 3])])
    
    def ddf(self, t):
        return np.array([
            -self.coeff[0, 1] * self.coeff[0, 2]**2 * np.sin(self.coeff[0, 2] * t + self.coeff[0, 3]),
            -self.coeff[1, 1] * self.coeff[1, 2]**2 * np.sin(self.coeff[1, 2] * t + self.coeff[1, 3]),
            -self.coeff[2, 1] * self.coeff[2, 2]**2 * np.sin(self.coeff[2, 2] * t + self.coeff[2, 3])])
