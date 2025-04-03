"""
Routines to compute the smoothing kernel in 2D SPH models

AS8003: Computational astrophysics, Department of Astronomy, Stockholm University

"""
from nbody_template import *
import numpy as np 

# ******************************************

def getW(pos, p0, h):
    """
    Computes the smoothing kernel based on cubic splines
    Reference: Rosswog (2009)

    With q = |r| / h:
    W =
       1) C * ((2.0 - q)**3 - 4.0*(1.0-q)**3) if 1>q>=0
       2) C * ((2.0 - q)**3                   if 2>q>=1
       3) 0.0                                 if q>2

    with C = 5/(14 * pi * h**2)
    
    Input parameters:
       pos: positions at which the kernel of the j-th particle must be evaluated ((n_dim, npos))
         p: positions of the j-th particle (n_dim)
         h: smoothing length (scalar)
    
    Coded by J. de la Cruz Rodriguez (ISP-SU, 2025)
    """

    # Init som variables and dimensions
    
    C = 5.0 / (14.0 * np.pi * h**2)
    ndim, npos = pos.shape
    W = np.zeros(npos)

    
    # subtract p0 from all particle positions in pos to compute r - rj

    dr = pos - np.outer(p0, np.ones(npos))


    # compute q = |r-rj| / h

    q = np.sqrt((dr**2).sum(axis=0)) / h
    
    # Case 1
    
    idx = q < 1.0
    qidx = q[idx]
    W[idx] = C * ((2.0 - qidx)**3 - 4.0*(1.0-qidx)**3)

    
    # Case 2
    
    idx = (q<2.0) & (q>=1.0)
    qidx = q[idx]
    W[idx] = C * (2.0 - qidx)**3

    
    # Case 3: the rest are already zero
    
    return W

# ******************************************

def getdW(pos, p0, h):
    """
    See definitions above in getW.
    
    Computes dW/dx and dW/dy:

    dW/d{x,y} = (3 * C / h ) *
            case 1: ((2-q)**2 - 4*(1-q)**2) * pos_{x,y} / r
            case 2: (2-q)**2 * pos_{x,y} / r
            case 3: 0.0

    Coded by J. de la Cruz Rodriguez (ISP-SU, 2025)
    """

    # Init som variables and dimensions
    
    C = 15.0 / (14.0 * np.pi * h**3)
    ndim, npos = pos.shape
    dW = np.zeros((ndim, npos))

    
    # subtract p0 from all particle positions in pos to compute r - rj

    dr = pos - np.outer(p0, np.ones(npos))
    r = np.sqrt((dr**2).sum(axis=0))
    q = r / h
    

    # Case 1
    
    idx = (q < 1.0) & (q>0.0)
    qidx = np.ascontiguousarray(q[idx])
    
    tmp = C * ((2.0 - qidx)**2 - 4.0*(1.0-qidx)**2) / (qidx*h)
    dW[0, idx] =  tmp * dr[0, idx]
    dW[1, idx] =  tmp * dr[1, idx]

    
    # Case 2
    
    idx = (q<2.0) & (q>=1.0)
    qidx = np.ascontiguousarray(q[idx])
    tmp = C * ((2.0 - qidx)**2 - 4.0*(1.0-qidx)**2) / (qidx*h)
    dW[0, idx] = tmp * dr[0,idx]
    dW[1, idx] = tmp * dr[1,idx]

    return dW
    
# ******************************************

def getRho(position, p0, h, mass_particles):  # why do we want p0 as an argument, cant we just take it as an index of position and loop it?

    rho = np.zeros(len(mass_particles))
    W = getW(position, p0, h)
    rho = np.sum(mass_particles * W)  # with probably some axis, probably wrong

    return rho

def getPressure(position, p0, h, mass_particles):  # why do we want p0 as an argument, cant we just take it as an index of position and loop it?

    pressure = np.zeros(len(mass_particles))
    dW = getdW(position, p0, h)
    pressure = -2*kappa*np.sum(mass_particles * dW)  # probably wrong, but a start to vizualise it. What is kappa?

    return pressure


if __name__ == "__main__":

    viscous_term = 0.5
    ngrid = 200
    n_step = 1000
    n_star = 2000
    dt = tau * 0.01
    h = 0.1

    position, velocity, mass_particles = InitialConditions(n_star, ...)

    for index in range(n_star):
        W = getW(position, position[:,index], h)
        dW = getdW(position, position[:,index], h)
        rho = getRho(position, position[:,index], h, mass_particles)
        pressure = getPressure(position, position[:, index], h, mass_particles)

        acc = getAcceleration(position, mass_particles)

        real_acc = - pressure + acc - velocity * viscous_term
        #put in leapfrog here





        #this is the plot I think
        x =
        y = 
        ygr, xgr = np.meshgrid(x,y, indexing="ij")
        pos = np.ascontiguousarray([xgr.flatten(), ygr.flatten()])
        rho = getRho(position, p0, h, mass_particles)
        rho = rho.reshape((ngrid,ngrid))
    
