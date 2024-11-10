from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

### Part 1 A
#From calculating the Equation of state for a degenerate, non-relativistic electron gas with Y_e = 0.5 we get K = 8.55*10**17
def EOS(density): return 8.55e17 * density**(5/3)

def white_dwarf(radius, y):  # set up out white dwarf, assigning the mass and pressure which are then calculated due to the density and radius
    mass = y[0]
    pressure = y[1]

    density = (pressure / 8.55e17) ** (3/5)  # flip the EOS to get the density, want to just put it in through parameter but hten return has to contain three things

    dmdr = 4 * np.pi * radius**2 * density  # the mass equation
    dPdr = -6.67259e-8 * mass * density / radius**2  # the hydrostatic equilibrium equation

    return [dmdr, dPdr]

def stop_integration(radius, y): return y[1]  # an "event" which tells the solve_ivp to stop integrating when y[1] (the pressure) reaches zero

stop_integration.terminal = True  # stops the integration when this is true


radius = []  # store the radius result
mass = []  # store the mass result

density_values = np.logspace(7, 10, 20)  # central density ranges from 10^7 g/cm^3 to 10^10/cm^3

for rho_c in density_values:  # looping over every density value 
    P_c = EOS(rho_c)
    initial_conditions = [0, P_c]  # set the start conditions to no mass and the start pressure
    sol = solve_ivp(white_dwarf, [1e-5, 1e10], initial_conditions, events = stop_integration, dense_output = True, rtol = 1e-10, first_step = 1e4)  # calculate the diff equations
    print(f"Central density: {rho_c}")
    print(f"t_events: {sol.t_events}")
    print(f"Final pressure: {sol.y[1, -1]}")
    print(f"Final mass: {sol.y[0,-1] / 1.989e33 } M_sun\n")
    #print(rho_c, sol.y[0][-1], sol.t_events)
    #radius.append(sol.t[0] / 6.955e10)  # convert to solar radii
    #mass.append(sol.y[0] / 1.989e33)  # convert to solar mass
print(sol.t)
"""""
###plotting the results
plt.plot(radius, mass)

sirius_b_mass = 1.018  # solar mass
sirius_b_radius = 0.008098  # solar radii
plt.errorbar(sirius_b_radius, sirius_b_mass, xerr=0.000046, yerr=0.011, fmt = 'o')
plt.xlabel("Radius")
plt.ylabel("Mass")
plt.grid(True)
plt.show()
"""""