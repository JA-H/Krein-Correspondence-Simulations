import numpy as np
import matplotlib.pyplot as plt

class Krein_Brownian_Killed:
    """
    This class contains the formulas for the extension function and Laplace exponent of a reflected Brownian motion in [0, R]
    killed upon hitting R.
    """
    def __init__(self, R, error = 1E-10):
        self.R = R
        self.error = error
    
    def Extension_Function(self, xi, y):
        if xi <= self.error:
            return 1.0
        else:
            return ( np.exp(-y*np.sqrt(xi)) - np.exp( y*np.sqrt(xi) - 2.0*self.R*np.sqrt(xi) )  )/(1.0 - np.exp(-2.0*self.R*np.sqrt(xi)))

    def Laplace_Exponent(self, xi):
        if xi <= self.error:
            return 1.0/self.R
        else:
            return np.sqrt(xi)*(1 + np.exp(-2.0*self.R*np.sqrt(xi) ))/(1 - np.exp(-2.0*self.R*np.sqrt(xi) ))