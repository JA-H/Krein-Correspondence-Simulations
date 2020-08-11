import numpy as np
import matplotlib.pyplot as plt

class Krein_Brownian_Killed:
    def __init__(self, R):
        self.R = R
    
    def Extension_Function(self, xi, y):
        return ( np.exp(-y*np.sqrt(xi)) - np.exp( y*np.sqrt(xi) - 2.0*self.R*np.sqrt(xi) )  )/(1.0 - np.exp(-2.0*self.R*np.sqrt(xi)))

    def Laplace_Exponent(self, xi):
            return np.sqrt(xi)*(1 + np.exp(-2.0*self.R*np.sqrt(xi) ))/(1 - np.exp(-2.0*self.R*np.sqrt(xi) ))