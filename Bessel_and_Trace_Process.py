import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

class Bessel:
    def __init__(self, T, dt, dim):
        self.T = T #Length of time interval
        self.dt = dt #Length of time step
        self.Num = round(T/dt) #Number of time steps
        self.dim = dim #Dimension of the Bessel process 
        
        self.alpha = 2.0 - dim #Alpha corresponding to the dimension of the Bessel process
        self.c_alpha = ( (2.0**(-self.alpha) )*np.absolute( gamma(-self.alpha/2.0) ) )/gamma(self.alpha/2.0) \
             #Constant that appears in corresponding speed measure of the rescaled Bessel process

    def Squared_Bessel_Process(self):
        """In order to simulate a Bessel process, we use the algorithm \
            found in 'Makarov, & Glew. Exact simulation of Bessel diffusions'. """
        Q = np.zeros(self.Num) # Memory for Squared Bessel process
        error = 1E-10
    
        for i in range(self.Num - 1):
            if Q[i] <= error:
                Q[i + 1] = np.random.gamma(self.dim/2.0, 2.0*self.dt )  
            else:
                Y = np.random.poisson( Q[i]/(2.0*self.dt)  )   
                Q[i + 1] = np.random.gamma( Y + self.dim/2.0, 2.0*self.dt )     
        return Q
    
    def Bessel_Process(self):
        return np.sqrt( self.Squared_Bessel_Process() )

    def Rescaled_Bessel_Process(self):
        return (self.c_alpha)*(self.Bessel_Process() )**(self.alpha)

    def Local_Time(self, delta, Y):
        """This function takes a time T, a time increment dt, a rescaled Bessel \
            Process Y with dimension dim and a small value delta and returns the \
            approximate local time of the sample path."""
        alpha = self.alpha
        dim = self.dim
        c_alpha = self.c_alpha

        m_delta = ( 1.0/( dim*alpha*(c_alpha)**(2.0/alpha) ) )*delta**( dim/alpha )

        Lt = np.zeros(self.Num)

        for i in range(self.Num - 1):
            if 0.0 <= Y[i] <= delta:
                Lt[i+1] = (1.0/m_delta)*self.dt
            else:
                Lt[i+1] = 0.0
                
        Lt = np.cumsum(Lt)
                
        return Lt

def Brownian_Motion(T, dt):
    """This function takes a time T and a time increment dt and returns an \
        array of the values of the X_t process at these time increments."""
        
    N = round(T/dt) # Number of time-steps
    X = np.random.standard_normal(size = N)
    X = np.cumsum(X)*np.sqrt(dt) 
    return X

def main(T, dt, dim, delta):
    X = Brownian_Motion(T, dt)

    Bes = Bessel(T, dt, dim)

    Y = Bes.Rescaled_Bessel_Process() 
    L = Bes.Local_Time(delta, Y)
    
    #Pair Process Plot
    plt.plot(X, Y, linewidth=0.1)
    plt.xlabel(r'$(X_t)_{t \geq 0}$')
    plt.ylabel(r'$(Y_t)_{t \geq 0}$')
    plt.savefig('./images/Plot_of_Pair_Processes/Plot_of_Pair_Process_alpha=' + str(Bes.alpha) + '.png', dpi=300)
    plt.show()
    plt.close()

    #Subordinated Process Plot
    plt.plot(L, X, linewidth=0.3)
    plt.xlabel(r'$(L^0_t(Y))_{t \geq 0}$')
    plt.ylabel(r'$(X_t)_{t \geq 0}$')
    plt.savefig('./images/Plot_of_Pair_Processes/Plot_of_Trace_Process_alpha='+ str(Bes.alpha) + '.png', dpi=300)
    plt.show()
    plt.close()
    
    return 0

main(T = 10.0**4, dt = 1E-3, dim = 0.8, delta = 1E-3)
main(T = 10.0**4, dt = 1E-3, dim = 1.2, delta = 5E-2) 
