"""
Routines to compute the smoothing kernel in 2D SPH models

AS8003: Computational astrophysics, Department of Astronomy, Stockholm University

"""  # SAVE GRAPHICS AS PDF, NOT PNG
import numpy as np 
import matplotlib.pyplot as plt; plt.ion()
import numba
# ******************************************
# USEFUL CONSTANTS#
Rg = 9.5E17       
R_sun = 7e5
M_sun = 1.99E30
Mg = 1.54E12 * M_sun 
G = 6.67E-11
tau = (R_sun**3 / (G*M_sun))**0.5 # time unit
n_dim = 2

kappa = 7.E-6
viscous_term = 100
h = 0.02*R_sun

# ******************************************

"""
Routines to compute the smoothing kernel in 2D SPH models

AS8003: Computational astrophysics, Department of Astronomy, Stockholm University

"""

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
    
    idx = ((q<2.0) & (q>=1.0))
    qidx = q[idx]
    W[idx] = C * (2.0 - qidx)**3

    
    # Case 3: the rest are already zero
    
    return W

# ******************************************

@numba.njit(fastmath=True)
def getdW(pi, pj, h):
    """
    pi: coordinates of particle i
    pj: coordinates of particle j (the kernel of this one is projected at location i).
    h: smoothing distance
    
    Computes dW/dx and dW/dy:

    dW/d{x,y} = (3 * C / h ) *
            case 1: ((2-q)**2 - 4*(1-q)**2) * pos_{x,y} / r
            case 2: (2-q)**2 * pos_{x,y} / r
            case 3: 0.0
    Coded by J. de la Cruz Rodriguez (ISP-SU, 2025)
    """

    # Init constant
    
    Norm = - 15.0 / (14.0 * np.pi * h**3)

    
    # get distance vector ri - rj
    
    dr = pi - pj 
    r = np.sqrt((dr**2).sum())

    # define q
    q = r / h

    tmp = 0.0
    
    if(q >= 2.0): # case q >= 2
        tmp = 0.0
        
    elif(q >= 1.0): # case q > 1 and q < 2
        tmp = Norm * ((2.0 - q)**2) / (r)

    elif(q > 0.0): # case q > 0.0 and q<1.0
        tmp = Norm * ((2.0 - q)**2 - 4.0*(1.0-q)**2) / (r)

    return tmp * dr
    
# ******************************************


def mkStar2D(n_particle, Star_radius, Star_mass):

    position  = np.zeros((n_dim, n_particle))  # first row is x position of particle, second row is y position of particle
    velocity  = np.zeros((n_dim, n_particle))  # first row is x velocity of particle, second row is y velocity of particle
    mass_particles = np.zeros(n_particle)

    ## Implement everything here! to fill these arrays
    mass_particles =  Star_mass / n_particle  # get mass for each particle
    
    r = np.abs(np.random.normal(0, Star_radius, n_particle))  # get r position of particle
    theta = np.random.uniform(0, 2*np.pi, n_particle)  # get theta for each particle

    position[0] = r * np.cos(theta)  # get x position of each particle
    position[1] = r * np.sin(theta)  # get y position of each particle 

    return position, velocity, mass_particles


@numba.njit(fastmath=True,parallel=True)
def getAcceleration(position, mass_particles, h, kappa, viscous_term, velocity):

    ndim, nparticle = position.shape
    acc = np.zeros((ndim, nparticle))
    

    # CALCULATED ACCELERATION HERE
    for index in numba.prange(nparticle):

        p_i = position[:,index]
        acc_i = np.zeros(ndim)

        for j in range(nparticle):
            if(index != j):
                p_j = position[:,j]
                dr = p_j - p_i
                distance_sqrt = np.sqrt((dr**2).sum())
                distance_qubed = max(0.01*R_sun,distance_sqrt)**3 
        
                acc_i += G * mass_particles * dr / distance_qubed
                acc_i -= 2 * kappa * mass_particles * getdW(p_i, p_j, h)
        acc[:,index] = acc_i - viscous_term * velocity[:,index]
                                   
    return acc


if __name__ == "__main__":



    n_step = 10000
    n_star = 1000
    dt = tau * 0.0003

    #Initialize the beginning parameters
    position, velocity, mass_particles = mkStar2D(n_star, 0.4*R_sun, M_sun)

    acc = getAcceleration(position, mass_particles, h, kappa, viscous_term, velocity)


    f, ax = plt.subplots(figsize=(6,6))
    
    d0, = ax.plot(position[0]/R_sun, position[1]/R_sun, 'o', color='orangered', ms=2.5, alpha=0.5, linewidth=0, mew=0)
    
    
    # plot labels
    
    ax.set_ylabel("y/Rg")
    ax.set_xlabel("x/Rg")
    ax.set_title("t_step={0}".format(0))

    
    # Fill the limits of your plot!
    
    ax.set_xlim()
    ax.set_ylim()
    f.savefig("bla.pdf")

    #Leapfrog iteration to update them
    for tt in range(1,n_step):

        #No clue if this is how to implement the real acceleration

        velocity = velocity + acc * dt / 2  # calculate new velocity a the half step
        position = position + velocity * dt  # calculate position at the full step
        acc = getAcceleration(position, mass_particles, h, kappa, viscous_term, velocity)  # calculate the acceleration at the full step
        velocity = velocity + acc * dt / 2  # calculate new velocity at the full step

        if(tt%4 == 0):


            d0.set_data(position[0]/R_sun, position[1]/R_sun)

            # Update title with the time step

            ax.set_title("t_step = {0}".format(tt))

            # Force re-drawing the figure
            
            f.canvas.draw()
            f.canvas.flush_events()

    
