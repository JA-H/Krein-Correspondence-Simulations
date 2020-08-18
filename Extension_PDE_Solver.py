import numpy as np
import fenics as pde
import matplotlib.pyplot as plt

from Krein_Brownian_Killed_Class import *

# PDE domain is (x_0, x_1)x(0, R) with N_x (resp. N_y) points in x (resp. y)
x_0, x_1 = 0.0, np.pi
R = np.pi
N_x, N_y = 150, 150

test = Krein_Brownian_Killed(R)

#Creating our mesh and test function space
mesh = pde.RectangleMesh( pde.Point(x_0, 0.0), pde.Point(x_1, R), N_x, N_y )
V = pde.FunctionSpace(mesh, "Lagrange", 1 )

#We have a zero Dirichlet boundary around the boundary except at x = 0.
tol = 1E-14

def Outer_Boundary(x, on_boundary):
    return (on_boundary) and (x[0] <= x_0 + tol or x[0] >= x_1 - tol or x[1] >= R - tol)

#We set this to be a zero boundary condition on the boundary defined above.
u0 = pde.Constant( 0.0 )
bc = pde.DirichletBC(V, u0, Outer_Boundary)

#To solve -psi(-d^2)u = f, we define u'(x, 0) = g written in C++
g_str = 'sin(x[0]) + 3.0*sin(3.0*x[0]) + 10.0*sin(10.0*x[0])'

#Solving as in Poisson problem
u = pde.TrialFunction(V)
v = pde.TestFunction(V)
g = pde.Expression( g_str, element = V.ufl_element() )
a = pde.inner( pde.grad(u), pde.grad(v) )*pde.dx
L = g*v*pde.ds

u = pde.Function(V)
pde.solve( a == L, u, bc )
u.set_allow_extrapolation(True)

#Plot of 2D solution
p = pde.plot(u)
vtkfile = pde.File("test_extension.pvd")
vtkfile << u

plt.colorbar(p)
plt.plot()
plt.xlabel(r"$0 \leq x \leq \pi$")
plt.ylabel(r"$0 \leq y \leq \pi$")
plt.savefig("pde.png", dpi = 300)
plt.close()

#Simulated u(x, 0) vs. Actual u(x, 0)
x_bound = np.linspace(x_0, x_1, N_x )
u_bound_val = np.array( [u(x, 0.0) for x in x_bound ] )
act_val = (1.0/test.Laplace_Exponent(1.0**2.0))*np.sin( x_bound ) \
    + (3.0/test.Laplace_Exponent(3.0**2.0))*np.sin( 3.0*x_bound ) \
    + (10.0/test.Laplace_Exponent(10.0**2.0))*np.sin( 10.0*x_bound )

#Plot of u(x, 0)
plt.plot( x_bound, u_bound_val, "-r", label="Simulated boundary values")
plt.plot( x_bound, act_val, "-m", label="Actual boundary values" )
plt.xlabel(r"$0 \leq x \leq \pi$")
plt.ylabel(r"$u(x, 0)$")
plt.legend()
plt.savefig("boundary_values_of_pde.png", dpi = 300)
plt.show()
plt.close()

