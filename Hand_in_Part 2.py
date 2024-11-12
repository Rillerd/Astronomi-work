import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

C = 3e10  # lightspeed [cgs]
G = 6.7e-8  # graviational constant [cgs]
K = 5.4e9  # derived K constant [cgs]
M_sun = 2e33  # solar mass [cgs]
R_sun = 7e10  # solar radius [cgs]

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

def solve_neutron_star(rho_c, K, C):
    P_c = K * rho_c ** (5/3)
    intial_conditions = [P_c, 0]
    solution = solve_ivp(neutron_star_structure, [1e-5, 1e7], intial_conditions, 
                         dense_output = True, args = (C, G, K), rtol = 1e-10, first_step = 10)
    radiuses = solution.t
    pressures, masses = solution.y

    surface_idx = np.argmax(pressures <= 1e-20)
    if pressures[surface_idx] > 1e-20:
        surface_idx = -1

    return radiuses[:surface_idx], masses[:surface_idx], pressures[:surface_idx]

central_densities = np.logspace(13, 15, 10)

plt.figure(figsize=(10, 6))

for rho_c in central_densities:
    radiuses, masses, pressures = solve_neutron_star(rho_c, K, C)
    plt.plot(masses[-1] / M_sun, radiuses[-1] / R_sun, label=f'rho_c = {rho_c:.1e} g/cm³')
    print(f"{masses / M_sun} M_sun, {radiuses / R_sun} R_sun, {pressures} \n")

plt.ylabel("Radius [$R_\\odot$]")
plt.xlabel("Mass [$M_\\odot$]")
plt.title("Mass-Radius Relationship for Neutron Star")
plt.legend()
plt.grid(True)
plt.show()
