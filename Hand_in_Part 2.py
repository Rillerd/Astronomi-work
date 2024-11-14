import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

C = 3e10  # lightspeed [cgs]
G = 6.7e-8  # graviational constant [cgs]
K = 5.4e9  # derived K constant [cgs]
M_sun = 2e33  # solar mass [cgs]
R_sun = 7e10  # solar radius [cgs]
km = 1e5
"""
### Part A
def neutron_star_structure(radius, y, C, G, K):
    pressure, mass = y

    if pressure <= 0:  # make sure we don't get a negative pressure
        return [0,0]
    
    rho = (pressure / K) ** (3/5)  # invert EOS to get rho
    epsilon = (3/2) * (pressure / rho)  # solve the gamma equation to get epsilon
    
    dP_dr = -( G * (mass + 4 * np.pi * radius**3 * pressure / C**2) * 
             (rho + rho * epsilon / C**2 + pressure / C**2)) / (radius**2 * (1 - (2 * G * mass) / (radius * C**2)))
    dm_dr = 4 * np.pi * radius**2 * (rho + rho*epsilon / C**2)

    return [dP_dr, dm_dr]

def solve_neutron_star(rho_c, K, G, C):
    P_c = K * rho_c ** (5/3)
    intial_conditions = [P_c, 0]
    solution = solve_ivp(neutron_star_structure, [1e3, 2e8], intial_conditions, 
                         dense_output = True, args = (C, G, K), rtol = 1e-10, first_step = 100)
    radiuses = solution.t
    pressures, masses = solution.y

    surface_idx = np.argmax(pressures <= 1e-8)
    if pressures[surface_idx] > 1e-8:
        surface_idx = -1

    return radiuses[:surface_idx], masses[:surface_idx], pressures[:surface_idx]

central_densities = np.logspace(13, 15, 30)

plt.figure(figsize=(10, 6))

for rho_c in central_densities:
    radiuses, masses, pressures = solve_neutron_star(rho_c, K, G, C)
    plt.plot(radiuses / km, masses / M_sun)
    print(f"{pressures} \n")

plt.xlabel("Radius [km]")
plt.ylabel("Mass [$M_\\odot$]")
plt.title("Mass-Radius Relationship for Neutron Star")
plt.legend()
plt.grid(True)
plt.show()

"""
### Part B

central_densities = np.logspace(13, 16, 50)

###Epsilon and pressure values from CompOSE, cold neutron star website

epsilon_values = np.array([9.77e18, 9.95e18, 1.013e19, 1.03e19, 1.05e19, 1.07e19, 1.09e19, 1.115e19, 
    1.138e19, 1.1633e19, 1.1909e19, 1.221e19, 1.255e19, 1.29e19, 1.327e19, 1.372e19, 
    1.424e19, 1.485e19, 1.56e19, 1.648e19, 1.757e19, 1.891e19, 2.058e19, 2.2777e19, 2.5687e19,
    2.969e19, 3.5321e19, 4.328e19, 5.454e19, 7.020e19, 9.161e19, 1.201e20, 1.572e20,
    2.044e20, 2.631e20, 3.349e20, 4.216e20, 5.249e20, 6.466e20, 7.892e20, 9.552e20, 
    1.147e21, 1.369e21, 1.626e21, 1.919e21, 2.575e21, 2.645e21, 3.0885e21, 3.5968e21, 4.1787e21])

real_pressures_c = np.array([1.35e31, 1.61e31, 1.907e31, 2.266e31, 2.706e31, 3.239e31, 3.895e31, 4.714e31, 
                             5.76e32, 7.12e31, 8.95e31, 1.14e32, 1.45e32, 1.68e32, 2.22e32, 
                             2.99e32, 4.067e32, 5.575e32, 7.686e32, 1.067e33, 1.499e33, 2.141e33, 3.133e33, 
                             4.727e33, 7.358e33, 1.1755e34, 1.907e34, 3.104e34, 
                             5.008e34, 7.94e34, 1.231e5, 1.862e35, 2.751e35, 3.975e35, 5.634e35,
                             7.856e35, 1.0809e36, 1.471e36, 1.9855e36, 2.6616e36, 3.5495e36, 4.715e36,
                             6.2443e36, 8.251e36, 1.0884e37, 1.434e37, 1.8877e37, 2.4834e37,
                             3.2659e37, 4.294e37])


def neutron_star_structure(radius, y, C, G, epsilon_values, real_pressures_c, central_densities):
    pressure, mass = y

    if pressure <= 0:  # make sure we don't get a negative pressure
        return [0,0]
    
    epsilon = np.interp(pressure, real_pressures_c, epsilon_values)  # calc the epsilon for a given pressure
    rho = np.interp(pressure, real_pressures_c, central_densities)  # calc the density for a given pressure
    

    dP_dr = -( G * (mass + 4 * np.pi * radius**3 * pressure / C**2) * 
             (rho + (rho * epsilon / C**2) + pressure / C**2)) / (radius**2 * (1 - (2 * G * mass) / (radius * C**2)))
    dm_dr = 4 * np.pi * radius**2 * (rho + rho*epsilon / C**2)

    return [dP_dr, dm_dr]

def solve_neutron_star(C, G, P_c, epsilon_values, real_pressures_c, central_densities):

    intial_conditions = [P_c, 0]
    solution = solve_ivp(neutron_star_structure, [1e-5, 2e7], intial_conditions,
                         dense_output = True, args = (C, G, epsilon_values, real_pressures_c, central_densities), rtol = 1e-6, first_step = 10)
    radiuses = solution.t
    pressures, masses = solution.y

    surface_idx = np.argmax(pressures <= 1e-8)
    if pressures[surface_idx] > 1e-8:
        surface_idx = -1

    return radiuses[:surface_idx], masses[:surface_idx], pressures[:surface_idx]

plt.figure(figsize=(10, 6))

for P_c in real_pressures_c:
    radiuses, masses, pressures = solve_neutron_star(C, G, P_c, epsilon_values, real_pressures_c, central_densities)
    plt.plot(masses / M_sun, radiuses /km,)
    print(masses[-1] / M_sun, radiuses[-1] / km, pressures[-1])

plt.xlabel("Mass [$M_\\odot$]")
plt.ylabel("Radius [km]")
plt.title("Mass-Radius Relationship for Neutron Star")
#plt.legend()
plt.grid(True)
plt.show()
#"""

### Part C
# https://www.nature.com/articles/s41550-020-1014-6 url that says 1.4M_sun neutron star has R = 11 +0.9 -0.6 km
