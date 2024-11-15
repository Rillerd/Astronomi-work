import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
"""
###Part A
def equations(radius, y):
    pressure, mass = y
    if pressure <= 0:
        return [0, 0]  # Stop integration at the surface where pressure drops
    rho = (pressure / 3.1e12) ** (3/5)
    dP_dr = -6.7e-8 * mass * rho / radius**2
    dm_dr = 4 * np.pi * radius**2 * rho
    return [dP_dr, dm_dr]

def solve_white_dwarf_structure(rho_c):
    P_c = 3.1e12 * rho_c**(5/3)
    initial_conditions = [P_c, 0]
    solution = solve_ivp(equations, [1e-5, 1e9], initial_conditions,
                         dense_output=True, rtol=1e-10, first_step=1e4)
    radiuses = solution.t
    pressures, masses = solution.y
    
    # Adjust surface detection to handle cases where the pressure never reaches 1e-8
    surface_idx = np.argmax(pressures <= 1e-8)  # Find first index where pressure <= threshold
    if pressures[surface_idx] > 1e-8:
        surface_idx = -1  # If threshold isn't met, use the last valid index
    
    return radiuses[:surface_idx], masses[:surface_idx], pressures[:surface_idx]

# Range of central densities
central_densities = np.logspace(5, 7, 10)  # g/cm^3

# Plot Mass-Radius Relationship
plt.figure(figsize=(10, 6))
for rho_c in central_densities:
    radius, masses, pressures = solve_white_dwarf_structure(rho_c)
    plt.plot(radius / 7e10, masses / 2e33, label=f'rho_c = {rho_c:.1e} g/cm³')
    print(radius[-1] / 7e10, masses[-1] / 2e33, pressures[-1])

# Observational data for Sirius B
plt.errorbar(0.008098, 1.018, xerr=0.000046, yerr=0.011, fmt='o', label='Sirius B')

# Plot configuration
plt.ylabel("Mass [$M_\\odot$]")
plt.xlabel("Radius [$R_\\odot$]")
plt.title("Mass-Radius Relationship for White Dwarfs")
plt.legend()
plt.grid(True)
plt.show()


###Part B
def equations(radius, y):
    pressure, mass = y
    if pressure <= 0:
        return [0, 0]  # Stop integration at the surface where pressure drops
    rho = (pressure / 5e14) ** (3/4)
    dP_dr = -6.7e-8 * mass * rho / radius**2
    dm_dr = 4 * np.pi * radius**2 * rho
    return [dP_dr, dm_dr]

def solve_white_dwarf_structure(rho_c):
    P_c = 5e14 * rho_c**(4/3)
    initial_conditions = [P_c, 0]
    solution = solve_ivp(equations, [1e-5, 1e9], initial_conditions,
                         dense_output=True, rtol=1e-10, first_step=1e4)
    radiuses = solution.t
    pressures, masses = solution.y
    
    # Adjust surface detection to handle cases where the pressure never reaches 1e-8
    surface_idx = np.argmax(pressures <= 1e-8)  # Find first index where pressure <= threshold
    if pressures[surface_idx] > 1e-8:
        surface_idx = -1  # If threshold isn't met, use the last valid index
    
    return radiuses[:surface_idx], pressures[:surface_idx], masses[:surface_idx]

# Range of central densities
central_densities = np.logspace(5, 10, 10)  # g/cm^3

# Plot Mass-Radius Relationship
plt.figure(figsize=(10, 6))
for rho_c in central_densities:
    radius, pressures, masses  = solve_white_dwarf_structure(rho_c)
    plt.plot(radius / 7e10, masses / 2e33, label=f'rho_c = {rho_c:.1e} g/cm³')
    print(radius / 7e10, masses / 2e33, pressures)

# Observational data for Sirius B
plt.errorbar(0.008098, 1.018, xerr=0.000046, yerr=0.011, fmt='o', label='Sirius B')

# Plot configuration
plt.ylabel("Mass [$M_\\odot$]")
plt.xlabel("Radius [$R_\\odot$]")
plt.title("Mass-Radius Relationship for White Dwarfs")
plt.legend()
plt.grid(True)
plt.show()
""" 

### Part C
pf_values = np.linspace(1e-17, 4.7e-16, 30)
x_values = pf_values / 2.7e-17  # get the x-values from the fermi momentums


def full_EOS(x_values):  # the full equation of state with the fermi-values derived from Part A and B and checking realistic densities and number densities for electrons
    pressures = []

    for x in x_values: 
        pressure = 5.8e22 * (x * (1 + x**2) ** (1/2) * (2 * x**2 / 3 - 1) + np.log(x + (1 + x**2) ** (1/2)))
        pressures.append(pressure)

    array_pressures = np.array(pressures)

    return array_pressures

P_e_values = full_EOS(x_values)

def interpolated_x(P_e): return np.interp(P_e, P_e_values, x_values)  # interpolated x for a given P_e
def interpolated_P_e(x): return np.interp(x, x_values, P_e_values)  # interpolate P_e for a given x

def interpolated_rho(P_e):  # interpolated density for given P_e and subsequently x
    x = interpolated_x(P_e)
    return 1.96e6 * x**3  # 1.96e6 comes from evaluating the constants when going from number density of electrons to density of electrons 

def equations(radius, y):

    pressure, mass = y

    if pressure <= 0:
        return [0, 0]  # Stop integration at the surface where pressure becomes 0
    
    rho = interpolated_rho(pressure)  # get the density for the given pressure

    dP_dr = -6.7e-8 * mass * rho / radius**2
    dm_dr = 4 * np.pi * radius**2 * rho

    return [dP_dr, dm_dr]

def solve_white_dwarf(P_c):

    initial_conditions = [P_c, 0]
    solution = solve_ivp(equations, [1e-5, 1e9], initial_conditions,
                         dense_output=True, rtol=1e-10, first_step=1e4)
    radiuses = solution.t
    pressures, masses = solution.y
    
    # Adjust surface detection to handle cases where the pressure never reaches 1e-8
    surface_idx = np.argmax(pressures <= 1e-8)  # Find first index where pressure <= threshold
    if pressures[surface_idx] > 1e-8:
        surface_idx = -1  # If threshold isn't met, use the last valid index
    
    return radiuses[:surface_idx], pressures[:surface_idx], masses[:surface_idx]

plt.figure(figsize=(10, 6))
for P_c in P_e_values:
    radiuses, pressures, masses = solve_white_dwarf(P_c)
    plt.plot(radiuses / 7e10, masses / 2e33)
    print(radiuses[-1] / 7e10, masses[-1] / 2e33, pressures[-1])

plt.scatter(0.008098, 1.018, color="blue", label="Sirius B (Observed)")
plt.xlabel("Radius [$R_\\odot$]")
plt.ylabel("Mass [$M_\\odot$]")
plt.title("Mass-Radius Relationship for White Dwarfs")
plt.legend()
plt.grid(True)
plt.show()
