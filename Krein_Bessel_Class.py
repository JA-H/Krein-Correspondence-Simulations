import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, kv

class Krein_Bessel:
    """
    This class contains the constants and functions that appear in the Krein \
        correspondence for the special case where \lambda \mapsto \lambda^(\alpha/2) \
        for some \alpha in (0, 2).
    """
    def __init__(self, alpha):
        self.alpha = alpha
        self.c_alpha = (2.0**(-alpha))*abs(gamma( -alpha/2. ))/gamma(alpha/2.)
    
    def Laplace_Exponent(self, xi):
        return xi**(self.alpha/2.)
    
    def Extension_Func(self, xi, y):
        if y < 1E-6:
            return 1.0
        else:
            z = ( y*xi**(self.alpha/2.) )/(self.c_alpha)
            varphi = ( 2.0**(1. - (self.alpha/2.0) ) )*(z**0.5)*kv( self.alpha/2.0 , z**(1.0/self.alpha)  )/( gamma(self.alpha/2.0) )  
            return varphi
    
    def Speed_Measure_Density(self, y):
        return (1.0/( (self.alpha**2.0)*( self.c_alpha**(2.0/self.alpha) ) )  )*y**(2.0/self.alpha - 2.0)

def main():
    B = Krein_Bessel(1.0)

    N = 1000
    T = 100.0
    xi = np.arange(0.1, T, (T - 0.1)/N)

    Fractional = B.Laplace_Exponent(xi)
    Fractional_approx = np.zeros(N)

    for i in range(N):
        Fractional_approx[i] = ( 1.0 - B.Extension_Func(xi[i], 0.1) )/0.1
    
    plt.plot(xi, np.exp(-Fractional), xi, np.exp(-Fractional_approx) )
    plt.savefig('test.png', dpi=300)
    plt.show()

main()