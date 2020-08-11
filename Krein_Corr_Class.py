import numpy as np
import matplotlib.pyplot as plt
import mpmath as mpm
import time

import Krein_Brownian_Killed_Class as BMKill

class Krein_Corr:
    """This class simulates the Krein correspondence for a Krein string given by a sum of 
    weighted Dirac measures, which may be used to approximate any given Krein string.
    To initialise the class, we require two numpy arrays, y and m, where y is a partition 
    of an interval [0, R] (with y[0] = 0 and y[-1] = R) and m corresponds to the Dirac point 
    measure on this partition. For simplicity, we assume that (in the notation of the thesis)
    that L = R < infty so the corresponding gap diffusion is killed upon hitting R.
    """

    def __init__(self, y, m):
        self.y = y # Points where the Krein string is defined
        self.m = m # Krein string which we assume is given by \sum_{y_i \in y} m_i\delta_{y_i}(dy)
        self.R = y[-1] #Endpoint which we assuime is killing
        self.drift_coeff = m[0] # This is forced to be positive due to the form of the Krein string
        
    def Extension_Func(self, xi):
        """
        We use the finite difference approximation of the BVP problem associated with the
        extension function. This BVP is given by,
        f''(y) = xi f(y)m(diff y), f(0) = 1, f(R) = 0,
        for fixed xi in [0, infty). 
        """
        y, m = self.y, self.m
        N = y.size

        #RHS of equation defining the Dirichlet boundary condition
        b = np.zeros(N)
        b[0] = 1.0

        #LHS matrix of the difference equation
        A = np.zeros( (N, N) )
        A[0, 0], A[N-1, N-1] = 1, 1
        
        for i in range(1, N - 1):
            A[i, i - 1] = -1.0
            A[i, i] = (( y[i] - y[i - 1] )/( y[i + 1] - y[i] )) + xi*m[i]*( y[i] - y[i - 1] ) + 1.0
            A[i, i+1] = -(( y[i] - y[i - 1] )/( y[i + 1] - y[i] ))
    
        varphi = np.linalg.solve(A, b)

        return varphi
        
    def Laplace_Exponent(self, xi, method = "CtdFrac"):
        """
        In this function we calculate the Laplace exponent at xi associated with the Krein string m via two 
        different methods: first by calculating the derivative at zero of extension function associated 
        with m, the second by directly calculating the continued fraction representation of 
        the complete Bernstein function. We set the default method to be the continued fraction method
        as the extension method is much slower due to the matrix computations imvolved.  
        """
        if method == "FinDiff":
            phi_approx = self.Extension_Func(xi)
            return (1.0 - phi_approx[1])/self.y[1] + self.m[0]*xi

        elif method == "CtdFrac":
            m, y = self.m, self.y

            A = np.array([1.0, m[0]*xi ])
            B = np.array([ 0.0, 1.0 ])
                
            for i in range(1, y.size - 1):
                # Convergents
                A_2 = ( y[i]-y[i-1] )*A[1] + A[0]
                A_3 = m[i]*xi*A_2 + A[1]
                A = np.array( [A_2, A_3] )
                
                B_2 = ( y[i]-y[i-1] )*B[1] + B[0]
                B_3 = m[i]*xi*B_2 + B[1]
                B = np.array( [B_2, B_3] )
                
                # Renormalisation every 10 iterations 
                if i % 10 == 0:
                    A = A/B[1]
                    B = B/B[1]
                    
            psi_xi = ( (y[-1] - y[-2])*A[1] + A[0] )/( (y[-1] - y[-2])*B[1] + B[0] )
            return psi_xi
        
    def Subordinator_pdf(self, t, T, N): 
        """
        In this function, we employ mpmath library to invert the Laplace transform of exp(-t*psi)
        where psi is the Laplace exponent numerically, giving the pdf of T_t.        
        """
        def Laplace_Trans_of_Sub(eta):
            Log_Lap_of_Sub = self.Laplace_Exponent( eta , method = "CtdFrac" )
            mpmLaplace = mpm.convert( Log_Lap_of_Sub )
            return mpm.exp( -t*mpmLaplace ) 

        times = np.linspace(0.0, T, N)
        sub_dist = np.zeros(N)
        for i in range(N):
            try:
                sub_dist[i] = mpm.invertlaplace( Laplace_Trans_of_Sub, times[i], method = 'talbot' )
            except ZeroDivisionError:
                continue 

        return sub_dist
    
def main():
    #Defining the approxiamtion of BM in [0, 1.0] killed upon hitting 1.0.
    N = int(1E2)
    R = 1.0
    y = np.linspace(0.0, R, N)
    m = (1.0/N)*np.ones(N)
    
    BM_Example = Krein_Corr(y, m)
    BM_Actual = BMKill.Krein_Brownian_Killed(R)

    xi_N = int(1E4)
    xi_max = 100.0
    xi_values = np.linspace(0.0, xi_max, xi_N)

    phi_exact = np.zeros(xi_N)    
    phi_approx = np.zeros(xi_N)
    phi_formula = np.zeros(xi_N)
    
    tic = time.perf_counter()
    for i in range(xi_N):
        phi_approx[i] = BM_Example.Laplace_Exponent(xi_values[i], "FinDiff")        
    toc = time.perf_counter()
    print(f"'FinDiff' took {toc - tic:0.2f} seconds")

    tic = time.perf_counter()
    for i in range(xi_N):
        phi_exact[i] = BM_Example.Laplace_Exponent(xi_values[i], "CtdFrac")        
    toc = time.perf_counter()
    print(f"'CtdFrac' took {toc - tic:0.2f} seconds")

    tic = time.perf_counter()
    for i in range(xi_N):
        phi_formula[i] = BM_Actual.Laplace_Exponent(xi_values[i])        
    toc = time.perf_counter()
    print(f"Exact formula took {toc - tic:0.2f} seconds")

    plt.plot(xi_values, phi_approx, 'r-', label=r"$\psi$ calculated via extension function", linewidth = 0.5)
    plt.plot(xi_values, phi_exact, 'b-', label=r"$\psi$ calculated as continued fraction", linewidth = 0.5)
    plt.plot(xi_values, phi_formula,'g-', label=r"Exact representation of $\psi$", linewidth = 0.5)
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\psi(\lambda)$")
    plt.legend()
    plt.savefig("CBF_comparison.png", dpi = 300)
    plt.close()
    
    
    T = 2.0
    N_t = 100
    times = np.linspace(0.0, T, N_t)
    
    tic = time.perf_counter()      
    sub_05 = BM_Example.Subordinator_pdf(0.5, T, N_t)
    print("Laplace transform to find pdf of T_{0.5} complete.")

    sub_1 = BM_Example.Subordinator_pdf(1.0, T, N_t)
    print("Laplace transform to find pdf of T_{1.0} complete.")
    
    sub_2 = BM_Example.Subordinator_pdf(2.0, T, N_t)
    print("Laplace transform to find pdf of T_{2.0} complete.")
    
    toc = time.perf_counter()
    print(f"Laplace transforms in {toc - tic:0.4f} seconds")

    plt.plot(times, sub_05, "g-", label=r"pdf of $T_{0.5}$")
    plt.plot(times, sub_1, "c-", label=r"pdf of $T_{1.0}$")
    plt.plot(times, sub_2, "m-", label=r"pdf of $T_{2.0}$")
    plt.xlabel(r"Time $t$")
    plt.ylabel(r"pdf of $T_s$ at time $t$")
    plt.legend()
    plt.savefig("subordinator_pdfs.png", dpi = 300)
    plt.show()
    plt.close

    return 0

main()