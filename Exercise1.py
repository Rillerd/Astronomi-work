import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

### Constants
T_HII = 1e4  # K
n_H = 1e4  # cm^-3 (number density of hydrogen, n_H = n_H0 + n_H+) n_e = n_H+
alpha_B = 2.59e-13  # cm^3 / s
nu_0 = 3.3e15  # Hz, ionization freauency for hydrogen
c = 3e10  # cm / s
h = 6.6e-27  # erg / s
k_B = 1.38e-16  # erg / K
sigma_0 = 6.3e-18  # cm^2

### O3 star
L_03 = 4e39  # erg/s 
T_03 = 4e4  #K


def blackbody(T, nu): return (2 * h * nu**3) / (c**2 * np.exp( (h * nu) / (k_B * T) ) - 1)


def normalization_constant(T, L):  # calculate the normalization factor
    integrand = lambda nu: blackbody(T, nu)
    flux, _ = quad(integrand, 0, 1e19)  # integrating blackbody function over all relevant frequencies, the blackbody ends at 1e16 so 1e19 is close enough to infinity
    return L / flux  # return the normalization factor


def photon_ionization_rate(T, norm_const):
    integrand = lambda nu: ((norm_const * blackbody(T, nu)) / (h * nu))  # spectral luminosity L_nu / h * nu
    Q_0, _ = quad(integrand, nu_0, 1e19)  # same reason to 1e19 as above instead of infinity
    return Q_0


def stromgren_radius(Q_0): return ((3 * Q_0) / (4 * np.pi * n_H**2 * alpha_B))**(1/3) # cm


def mean_energy(T, norm_const, Q_0):
    integrand = lambda nu:  (norm_const * blackbody(T, nu))
    flux, _ = quad(integrand, nu_0, 1e19)  # mean energy is the ratio between energy flux and photon flux, integral from nu0 to infinity l_nu
    return flux / Q_0 * 6.24e11  # this is the mean ionization energy, make this to eV


#################################################################################################################################################################
### The 3 functions below are required for part B

def AGN_luminosity(energy): return 1e41 * (energy / 13.6) ** (-3/2)


def AGN_Q_0():
    integrand = lambda energy: AGN_luminosity(energy) / energy
    Q_0, _ = quad(integrand, 13.6, np.inf)  # integrate to get photo_ionization rate which is integral from E_0 to E_max over the luminosity divided by energy
    return Q_0


def AGN_mean_energy(Q_0):
    integrand = lambda energy: AGN_luminosity(energy)  # the ionizing luminosity
    L_ionizing, _ = quad(integrand, 13.6, np.inf)
    return L_ionizing / Q_0

#################################################################################################################################################################


def cross_section(E_mean): return sigma_0 * (E_mean / 13.6 ) ** (-3)  # cross section is 2.3e-18 cm^2 which is less than for ionization potential which it should be 

"""
We set x = n(H+)/ n_H = n_e / n_H, y = r / R_SO, assume low neutral fraction and thus n_H = 1-x << 1 and leads to 
1 - x ~ (3 * y^2 * Q(0)) / (Q(r) * n_H * sigma * R_SO) << 1. And now assume large ionized fraction gives x approx 1, then
Q(r) / Q_0 ~ 1-y^3 = 1- (r/R_SO)^3. Equate the two equations for Q(r) and we get n_H0 = 1 - x = 3 * 
"""

def hydrogen_number_density(r, R_SO, sigma):
    n_H_neutral = (3 * (r/R_SO) ** 2) / ((1 - (r/R_SO) ** 3) * n_H * sigma * R_SO)
    n_H_plus = 1 - (3 * (r/R_SO) ** 2) / ((1 - (r/R_SO) ** 3) * n_H * sigma * R_SO)
    return n_H_neutral, n_H_plus


def optical_depth(radius): return n_H * sigma_0 * radius  # at the ionization threshold for hydrogen, our cross-section is the standard sigma_0


#def mean_free_path(): return 1 / (n_H * sigma * 3.1e18)  # in pc
def radius_values(R_SO): return np.linspace(0, 1.000001 * R_SO, 100000)


def densities_and_optical_depth(radiuses, R_SO, sigma):

    density_neutral_values = []
    density_ion_values = []
    optical_depth_values = []

    for radius in radiuses:
        density_neutral, density_ion = hydrogen_number_density(radius, R_SO, sigma)
        tau = optical_depth(radius)
        density_neutral_values.append(density_neutral)
        density_ion_values.append(density_ion)
        optical_depth_values.append(tau)
    
    
    pc_radiuses = radiuses / (3.1e18)
    
    return pc_radiuses, density_neutral_values, density_ion_values, optical_depth_values


def plotting(pc_radius, density_neutral_values, density_ion_values, optical_depth_values, R_SO):

    x_min = R_SO / 3.1e18 - 0.01
    x_max = R_SO / 3.1e18 + 0.01

    plt.plot(pc_radius, density_neutral_values, label = "HI")
    plt.plot(pc_radius, density_ion_values, label = "HII")
    plt.xlim()                             #                 plt.xlim(0.26, 0.265)
    plt.ylim(0,1)
    plt.xlabel("Distance [pc]")
    plt.ylabel("Ionization fraction")
    plt.legend()
    plt.show()

    plt.plot(pc_radius, optical_depth_values, label = "$\\tau$")
    plt.xlabel("Distance [pc]")
    plt.ylabel("Optical depth")
    plt.legend()
    plt.show()


def main():

    norm_const = normalization_constant(T_03, L_03)
    Q_0 = photon_ionization_rate(T_03, norm_const)
    E_mean = mean_energy(T_03, norm_const, Q_0)
    R_SO = stromgren_radius(Q_0)
    sigma = cross_section(E_mean)
    radiuses = radius_values(R_SO)
    pc_radiuses, density_neutral_values, density_ion_values, tau = densities_and_optical_depth(radiuses, R_SO, sigma)

    print(f"\nO3 supernova\nR_SO = {R_SO / 3.1e18:.2f} pc, Q_0 = {Q_0:.1e}, <E> = {E_mean:.1f} eV, cross-section = {sigma:.1e} cm^2")
    #plotting(pc_radiuses, density_neutral_values, density_ion_values, tau, R_SO)
 
    Q_0 = AGN_Q_0()
    E_mean = AGN_mean_energy(Q_0)
    R_SO = stromgren_radius(Q_0)
    sigma = cross_section(E_mean)
    radiuses = radius_values(R_SO)
    pc_radiuses, density_neutral_values, density_ion_values, tau = densities_and_optical_depth(radiuses, R_SO, sigma)

    print(f"\nAGN: \nR_SO = {R_SO / 3.1e18:.5f} pc, Q_0 = {Q_0:.1e}, <E> = {E_mean:.1f} eV, cross-section = {sigma:.1e} cm^2")
    plotting(pc_radiuses, density_neutral_values, density_ion_values, tau, R_SO)



if __name__ == "__main__":
    main()
