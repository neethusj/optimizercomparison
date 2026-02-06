import numpy as np

#----1D functions-----

""" CONVEX FUNCTION
f(x)=x^2
gradient=f'(x)=2x
"""
def quadratic(x):
    return x**2
def grad_quadratic(x):
    return 2*x

""" NON-CONVEX FUNCTION
    f(x)=x^4-3x^3+2
    gradient=f'(x)=4x^3-9x^2
"""

def non_convex(x):
    return x**4-3*x**3+2
def gradient_nonconvex(x):
    return 4*x**3-9*x**2

#----2D Benchmark Functions------

"""
Rosenbrock function[Valley/Banana function]-Unimodal
f(x,y)=(1-x)^2+100(y-x^2)^2
gradient=[dx,dy]
dx=-2(1-x)-400x(y-x^2)
dy=200(y-x^2)
Global minimum at (1,1)
"""

def rosenbrock(x,y):
       return (1-x)**2+100*(y-x**2)**2
def gradient_rosenbrock(x,y):
        dx=-2*(1-x)-400*x*(y-x**2)
        dy=200*(y-x**2)
        # return array of partial derivatives
        return np.array([dx,dy])


"""

Rastrigin Function-Multimodal
global minimum at (0,0)
f(x,y)=20+x^2+y^2-10cos(2.pi.x)-10cos(2.pi.y)
gradient=[dx,dy]
dx=2x+20.pi.sin(2.pi.x)
dy=2y+20.pi.np.sin(2.pi.y)
"""

def rastrigin(x,y):
     pi=np.pi
     return 20+x**2+y**2-10*np.cos(2*pi*x)-10*np.cos(2*pi*y)
def gradient_rastrigin(x,y):
     pi=np.pi
     dx=2*x+20*pi*np.sin(2*pi*x)
     dy=2*y+20*pi*np.sin(2*pi*y)
     return np.array([dx,dy])


"""
Himmelblau function:Have 4 global minima
(3,2), (-2.805,3.131), (-3.779,-3.283), (3.584,-1.848)
f(x,y)=(x^2+y-11)^2+(x+y^2-7)^2
dx=4x(x^2+y-11)+2(x+y^2-7)
dy=2(x^2+y-11)+4y(x+y^2-7)
"""
def himmelblau(x,y):
     return (x**2+y-11)**2+(x+y**2-7)**2
def gradient_himmelblau(x,y):
     dx=4*x*(x**2+y-11)+2*(x+y**2-7)
     dy=2*(x**2+y-11)+4*y*(x+y**2-7)
     return np.array([dx,dy])