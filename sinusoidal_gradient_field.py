import numpy as np

from gradient_field import GradientField

class SinusoidalGradientField(GradientField):

    def f(self, t):
        """ f(t) = coeff[:, 0] * sin(coeff[:, 1] t + coeff[:, 2])"""
        return np.array([self.coeff[0, 0] * np.sin(self.coeff[0, 1]  * t + self.coeff[0, 2]),
                         self.coeff[1, 0] * np.sin(self.coeff[1, 1]  * t + self.coeff[1, 2]),
                         self.coeff[2, 0] * np.sin(self.coeff[2, 1]  * t + self.coeff[2, 2])])

    def df(self, t):
        return np.array([
            self.coeff[0, 0] * self.coeff[0, 1] * np.cos(self.coeff[0, 1] * t + self.coeff[0, 2]),
            self.coeff[1, 0] * self.coeff[1, 1] * np.cos(self.coeff[1, 1] * t + self.coeff[1, 2]),
            self.coeff[2, 0] * self.coeff[2, 1] * np.cos(self.coeff[2, 1] * t + self.coeff[2, 2])])
