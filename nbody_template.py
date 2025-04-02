"""
N-Body simulation of tidal forces in self-gravitating problems

AS8003: Computational astrophysics, Department of Astronomy, Stockholm University
J. de la Cruz Rodriguez (ISP-SU, 2025)


Student name: 
Rikard Lesley
"""

import numpy as np
import numba
import matplotlib.pyplot as plt; plt.ion()
import os

# ******* Some useful quantities ***********

G = 6.67E-11                # N m^2 / kg^2
Msun = 1.99E30              # Kg
Mg = 1.54E12 * Msun         # kg
Rg = 9.5E17                 # m
tau = (Rg**3 / (G*Mg))**0.5 # time unit
n_dim = 2

# ******************************************

def mkGalaxy2D(n_particle, Galaxy_radius, Galaxy_mass, x0=0.0, y0=0.0, vx0=0.0, vy0=0.0):
    """
    Here you should implement a routine that places particles in the XY-plane
    and assigns an initial velocity to each of them. It should return three
    arrays: the positions array ((n_dim, n_particle)), the velocity array ((n_dim, n_particle))
    and the particle mass array (n_particle), where in this case n_dim = 2.

    Input parameters:
        n_particle: number of particles to distribute
                 R: Radius of the galaxy (it is the sigma parameter of a random Gaussian distribution) [m]
                 M: Total mass of the galaxy [kg]
                x0: x-coordinate of the center of the galaxy [m]
                y0: y-coordinate of the center of the galaxy [m]
               vx0: linear velocity component in the x-plane [m/s]
               vy0: linear velocity component in the y-plane [m/s]
    """

    position  = np.zeros((n_dim, n_particle))  # first row is x position of particle, second row is y position of particle
    velocity  = np.zeros((n_dim, n_particle))  # first row is x velocity of particle, second row is y velocity of particle
    mass_particles = np.zeros(n_particle)

    ## Implement everything here! to fill these arrays
    mass_particles = Galaxy_mass / n_particle  # get mass for each particle
    
    r = np.abs(np.random.normal(0, Galaxy_radius, n_particle))  # get r position of particle
    theta = np.random.uniform(0, 2*np.pi, n_particle)  # get theta for each particle

    position[0] = r * np.cos(theta)  # get x position of each particle
    position[1] = r * np.sin(theta)  # get y position of each particle 
    
    velocity[0] = -position[1]  # transpose the position vector and take negative of one to get an orthogonal direction to the position vector, perpendicular direction
    velocity[1] = position[0]  


    denumerator = np.sqrt(np.square(velocity[0, :]) + np.square(velocity[1, :]))

    vrot= np.sqrt(G * Galaxy_mass / r) * 0.05  
    velocity[0, :] = velocity[0, :] / denumerator * vrot
    velocity[1, :] = velocity[1, :] / denumerator * vrot # this is just the direction of the velocity, not the acutal velocity

    
    # Add the global offsets in position and velocity
    
    position[0,:] += x0
    position[1,:] += y0
    
    velocity[0,:] += vx0
    velocity[1,:] += vy0

    return position, velocity, mass_particles

#multiplied the vrot with 0.05 because I have all mass in the center for each particle and not the mass which is within a radius and thus the ones in the middle have too large vrot

# ******************************************

def InitialConditions(n_galaxy1, n_galaxy2):
    """
    Here you should create the initial conditions. You should call mkGalaxy2D
    twice to create two different galaxies at different coordinates
    """

    # place galaxies
    
    position1, velocity1, mass_particles1 = mkGalaxy2D(n_galaxy1, Rg, Mg, 0.0, 0.0, 0.0, 0.0)
    position2, velocity2, mass_particles2 = mkGalaxy2D(n_galaxy2, Rg / 3, Mg*0.01, 8*Rg, 2.5*Rg, -Rg / (2*tau), 0)



    # now concatenate those arrays into one large array
    
    position  = np.zeros((n_dim, n_galaxy1+n_galaxy2))
    velocity  = np.zeros((n_dim, n_galaxy1+n_galaxy2))
    mass_particles = np.zeros(n_galaxy1+n_galaxy2)

    position[:, 0:n_galaxy1] = position1
    position[:, n_galaxy1::] = position2
    velocity[:, 0:n_galaxy1] = velocity1
    velocity[:, n_galaxy1::] = velocity2
    mass_particles[0:n_galaxy1] = mass_particles1
    mass_particles[n_galaxy1::] = mass_particles2


    return position, velocity, mass_particles

# ******************************************

@numba.njit(fastmath = True, parallel = True)
def getAcceleration(position, mass_particles):
    """
    Given an array of particle positions p(n_dim, n_particle),
    calculate the net acceleration at the location of each particle
    from their gravitational interaction with the rest
    """

    ndim, nparticle = position.shape
    acc = np.zeros((ndim, nparticle))

    # CALCULATED ACCELERATION HERE
    for index in numba.prange(nparticle):

        i = position[:,index]  # get element of one column (particle)

        distance = np.sqrt(np.square(position -  np.outer(i, np.ones(nparticle))).sum(axis=0))   # calculate the distance by using matrices instead of more loops
        distance_third = np.power(distance + 0.1*Rg, 3)  # adds a small value to not divide by 0 later on
        distance_vec = position - np.outer(i, np.ones(nparticle))  # calculate the vector

        acceleration = G * np.sum( mass_particles * distance_vec / distance_third, axis = 1)  # calculate the acceleration vector
        acc[0, index] = acceleration[0]  # store it in the acc array so acceleration in x is in first row, and acceleration in y is in second row
        acc[1, index] = acceleration[1]  # use index to store the acceleration in the same index spot as the particles position

    return acc

#multipled 0.08 with galaxy radius because the particles may be too heavy 
    
# ******************************************


if __name__ == "__main__":  # main program, not executed if you import this file


    # simulation parameters (feel free to change them)
    
    n_step = 2000
    n_galaxy_1 = 6000
    n_galaxy_2 = 1000
    dt = tau * 0.01


    # Initialize the positions and velocities
    
    position, velocity, mass_particles = InitialConditions(n_galaxy_1, n_galaxy_2)


    # Initialize the acceleration using the initial positions

    acc = getAcceleration(position, mass_particles)



    # Initialize plots using matplotlib, scale the positions
    # with the initial radius of the galaxy

    f, ax = plt.subplots(figsize=(6,6))

    
    # Plot galaxy 1, all particles from 0:n_galaxu_1
    
    d0, = ax.plot(position[0,0:n_galaxy_1]/Rg, position[1,0:n_galaxy_1]/Rg, 'o', color='orangered', ms=2.5, alpha=0.5, linewidth=0, mew=0)

    
    # Plot galaxy 2 using a different color: particles from n_galaxy_1:(n_galaxu_1+n_galaxy2)
    
    d1, = ax.plot(position[0,n_galaxy_1::]/Rg, position[1,n_galaxy_1::]/Rg, 'o', color='navy', ms=2.5, alpha=0.5, linewidth=0, mew=0)
    #                                          changed from n_galaxy_1 as it wanted same dimensions, also changed alpha to make it less transparent and ms to make it bigger, and color
    
    # plot labels
    
    ax.set_ylabel("y/Rg")
    ax.set_xlabel("x/Rg")
    ax.set_title("t_step={0}".format(0))

    
    # Fill the limits of your plot!
    
    ax.set_xlim(-5,10)
    ax.set_ylim(-3,4)


    # Iterate the positions for all time steps  # save images with if tt = (timestep) savefig in a

    for tt in range(1, n_step):

        #IMPLEMENTED LEAPFROG HERE!!!!
        velocity = velocity + acc * dt / 2  # calculate new velocity a the half step
        position = position + velocity * dt  # calculate position at the full step
        acc = getAcceleration(position, mass_particles)  # calculate the acceleration at the full step
        velocity = velocity + acc * dt / 2  # calculate new velocity at the full step

        #Bottleneck is most definitely here, as getAcceleration has a loop 
        # meaning that we will have a nested loop. I should probably change the loop in getAcceleration with NumBa to use the cores to speed up the process


        # Update the plots with the new particle positions
        # to save some time, only update the plot every 4
        # iterations

        if(tt%2 == 0):

            # update the data points in each of the plots,
            # keeping all labels and axes exactly the same
            
            d0.set_data(position[0,0:n_galaxy_1]/Rg, position[1,0:n_galaxy_1]/Rg)
            d1.set_data(position[0,n_galaxy_1::]/Rg, position[1,n_galaxy_1::]/Rg)

            # Update title with the time step

            ax.set_title("t_step = {0}".format(tt))
            #put what was in flush events to this set_title
            # Force re-drawing the figure
            
            f.canvas.draw()
            f.canvas.flush_events()
            #f.canvas.flush_events("t_step={0}".format(tt))
                                  


            
