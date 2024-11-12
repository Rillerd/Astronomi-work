import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
"""
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
"""
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
    
    return radiuses[:surface_idx], masses[:surface_idx], pressures[:surface_idx]

# Range of central densities
central_densities = np.logspace(5, 10, 10)  # g/cm^3

# Plot Mass-Radius Relationship
plt.figure(figsize=(10, 6))
for rho_c in central_densities:
    radius, masses, pressures = solve_white_dwarf_structure(rho_c)
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
def full_EOS():  # the full equation of state with the fermi-values derived from Part A and B and checking realistic densities and number densities for electrons
    pf_values = np.linspace(1e-17, 4.7e-16, 30)
    pressures = []
    for p_f in pf_values: 
        x = p_f / 2.7e-17
        pressure = 5.8e22 * (x * (1 + x**2) ** (1/2) * (2 * x**2 / 3 - 1) + np.log(x +(1 + x**2) ** (1/2)))
        pressures.append(pressure)
    return np.array(pressures), pf_values

pressures, pf_values = full_EOS()

def interpolated_P_e(p_f): return np.interp(p_f, pf_values, pressures)  # interpolate P_e for a given fermi energy

def equations(radius, y):
    pressure, mass = y
    if pressure <= 0:
        return [0, 0]  # Stop integration at the surface where pressure becomes 0
    rho = (pressure / 5e14) ** (3/4)

    dP_dr = -6.7e-8 * mass * rho / radius**2
    dm_dr = 4 * np.pi * radius**2 * rho

    return [dP_dr, dm_dr]
"""

