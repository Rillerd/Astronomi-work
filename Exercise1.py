import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


### Constants
T_HII = 1e4  # K
n_H = 1e4  # cm^-3 (number density of hydrogen, n_H = n_H0 + n_H+) n_e = n_H+
alpha_B = 2.59e-13  # cm^3 / s
nu_0 = 3.3e15  # Hz, ionization frequency for hydrogen
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


def mean_energy(T, norm_const, Q_0):
    integrand = lambda nu:  (norm_const * blackbody(T, nu))
    flux, _ = quad(integrand, nu_0, 1e19)  # mean energy is the ratio between energy flux and photon flux, integral from nu0 to infinity l_nu
    return flux / Q_0 * 6.24e11  # this is the mean ionization energy, make this to eV


#######################################################################################################################################################################################
###################################                              Exercise 1B                 ############################################################################################

def AGN_luminosity(energy): return 1e41 * (energy / 13.6) ** (-1.5)


def AGN_Q_0():
    integrand = lambda energy: AGN_luminosity(energy) / energy
    Q_0, _ = quad(integrand, 13.6, np.inf)  # integrate to get photo_ionization rate which is integral from E_0 to E_max over the luminosity divided by energy. in units of erg / s / eV
    return Q_0 / (1.6e-12)  # to get in units of erg / s / erg = 1 / s
### divide result with eV

def AGN_mean_energy(Q_0):
    integrand = lambda energy: AGN_luminosity(energy)  # the ionizing luminosity
    L_ionizing, _ = quad(integrand, 13.6, np.inf)  # this is in eV, want to get to ergs
    return L_ionizing * (1.6e12) / Q_0  # L_ion was in erg/s/eV, need to multiply bu 1.6e12 to get consistent units.

#######################################################################################################################################################################################
#######################################################################################################################################################################################


def cross_section(E_mean): return sigma_0 * (E_mean / 13.6 ) ** (-3)  # cross section is 2.3e-18 cm^2 which is less than for ionization potential which it should be 


def stromgren_radius(Q_0): return ((3 * Q_0) / (4 * np.pi * n_H**2 * alpha_B))**(1/3) # cm


def hydrogen_number_density(r, R_SO, sigma):
    n_H_neutral_ratio = (3 * (r/R_SO) ** 2) / ((1 - (r/R_SO) ** 3) * n_H * sigma * R_SO)
    n_H_plus_ratio = 1 - n_H_neutral_ratio
    return n_H_neutral_ratio, n_H_plus_ratio


def optical_depth(radius, n_H_neutral): return n_H_neutral * sigma_0 * radius  # at the ionization threshold for hydrogen, our cross-section is the standard sigma_0


def radius_values(R_SO): return np.linspace(0, 1.2 * R_SO, 1000)


####################################################################################################################################################################################
                                  ############################## Exercise 2   ##############################

def heating(T, ion_ratio): 
    n_H_plus = ion_ratio * n_H
    alpha_B = 2.59e-13 * (T / 1e4) ** (-0.833-0.035 * np.log(T / 1e4))
    return n_H_plus * n_H * alpha_B * (3 / 2) * k_B * T # change alpha_B to temp dependent


def recombination_cooling(T, ion_ratio):   
    n_H_plus = ion_ratio * n_H
    alpha_B = 2.59e-13 * (T / 1e4) ** (-0.833-0.035 * np.log(T / 1e4))
    return n_H_plus ** 2 *  alpha_B * (0.684 - 0.0416 * np.log(T/1e4)) * k_B * T


def free_free_cooling(T, ion_ratio): 
    n_H_plus = ion_ratio * n_H
    return 0.54 * n_H_plus ** 2 * alpha_B * (T / 1e4) ** 0.37 * k_B * T


def collisional_cooling(T, ion_ratio, neutral_ratio): 

    n_H_plus = ion_ratio * n_H
    n_H_neutral = neutral_ratio * n_H

    A_21 = 4.69e8  # s^-1
    A_2gamma = 8.23  # s^-1  dont know if want the forbidden one also
    energy_difference = 10.23 * 1.6e-12  # erg
    omega_12 = 0.69  # for T = 2e4 K
    omega_gamma2 = 0.35  # for T = 2e4 K 
    stat_weight_S = 2
    stat_weight_P = 6

    q_22P  = (8.629e-6 * omega_12) / (T ** (1/2) * stat_weight_P )  * np.exp(-energy_difference / (k_B * T))  # collision excitation rate coefficient for 2^2P
    q_22S  = (8.629e-6 * omega_gamma2) / ( T ** (1/2) * stat_weight_S) *  np.exp(-energy_difference / (k_B * T))  # collision excitation rate coefficient for 2^2S

    return n_H_plus * n_H_neutral * (A_2gamma * q_22S +  A_21 * q_22P) * energy_difference  


def total_cooling(T, ion_ratio, neutral_ratio): return free_free_cooling(T, ion_ratio) + recombination_cooling(T, ion_ratio) + collisional_cooling(T, ion_ratio, neutral_ratio)

def plotting_all_rates():

    ff_values = []
    rr_values = []
    ce_values = []
    heating_values = []
    cooling_values = []
    temperatures = np.logspace(3,5, 100)

    
    neutral_ratio = 1e-4  #  neutral
    ion_ratio = 1 - neutral_ratio  # ion
    norm_factor = ion_ratio * n_H ** 2 # division by n_e * n_H for the plot

    for T in temperatures:  # calculate the cooling and heating for the different temperatures and then plot
        
        ff = free_free_cooling(T, ion_ratio)
        rr = recombination_cooling(T, ion_ratio)
        ce = collisional_cooling(T, ion_ratio, neutral_ratio)
        heat = heating(T, ion_ratio)
        cooling = total_cooling(T, ion_ratio, neutral_ratio)

        ff_values.append(ff / norm_factor)
        rr_values.append(rr / norm_factor)
        ce_values.append(ce / norm_factor)
        heating_values.append(heat / norm_factor)
        cooling_values.append(cooling / norm_factor)

    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, heating_values, label="Heating", color="orange")
    plt.plot(temperatures, rr_values, label="Recombination Cooling", color="blue")
    plt.plot(temperatures, ff_values, label="Free-Free Cooling", color="green")
    plt.plot(temperatures, ce_values, label="Collisional Cooling", color="red")
    plt.plot(temperatures, cooling_values, label="Total Cooling", color="purple", linestyle="--")

    # Labels and legend
    plt.xlabel("Temperature [K]")
    plt.ylabel("Rate per n_e n_H [erg cm^3 s^-1]")
    #plt.xlim(1000, 9000)
    #plt.ylim(1e-26, 1e-24)
    plt.yscale("log")
    plt.title("Normalized Heating and Cooling Rates vs Temperature")
    plt.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    plt.legend()
    plt.show()
    
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################

def densities_and_optical_depth(radiuses, R_SO, sigma):

    density_neutral_ratio_values = []
    density_ion_ratio_values = []
    optical_depth_values = []

    for radius in radiuses:
        density_neutral_ratio, density_ion_ratio = hydrogen_number_density(radius, R_SO, sigma)
        tau = optical_depth(radius, density_neutral_ratio)
        density_neutral_ratio_values.append(density_neutral_ratio)
        density_ion_ratio_values.append(density_ion_ratio)
        optical_depth_values.append(tau)
    
    
    pc_radiuses = radiuses / (3.1e18)
    
    return pc_radiuses, density_neutral_ratio_values, density_ion_ratio_values, optical_depth_values


def plotting(pc_radius, density_neutral_values, density_ion_values, optical_depth_values, title):

    plt.plot(pc_radius, density_neutral_values, label = "HI")
    plt.plot(pc_radius, density_ion_values, label = "HII")
    plt.ylim(0,1)
    plt.xlabel("Distance [pc]")
    plt.ylabel("Ionization fraction [X / n_H]")
    plt.title(title)
    plt.legend()
    plt.show()

    plt.plot(pc_radius, optical_depth_values, label = "$\\tau$")
    plt.xlabel("Distance [pc]")
    plt.ylabel("Optical depth")
    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.show()


def main():

    norm_const = normalization_constant(T_03, L_03)
    Q_0 = photon_ionization_rate(T_03, norm_const)
    E_mean = mean_energy(T_03, norm_const, Q_0)
    R_SO = stromgren_radius(Q_0)
    sigma = cross_section(E_mean)
    radiuses = radius_values(R_SO)
    pc_radiuses, density_neutral_ratio_values, density_ion_ratio_values, tau = densities_and_optical_depth(radiuses, R_SO, sigma)
    plotting_all_rates()
    print(f"\nO3 supernova\nR_SO = {R_SO / 3.1e18:.2f} pc, Q_0 = {Q_0:.1e}, <E> = {E_mean:.1f} eV, cross-section = {sigma:.1e} cm^2")
    #plotting(pc_radiuses, density_neutral_ratio_values, density_ion_ratio_values, tau, "O3 star")
    
    Q_0 = AGN_Q_0()
    E_mean = AGN_mean_energy(Q_0)
    R_SO = stromgren_radius(Q_0)
    sigma = cross_section(E_mean)
    radiuses = radius_values(R_SO)
    pc_radiuses, density_neutral_ratio_values, density_ion_ratio_values, tau = densities_and_optical_depth(radiuses, R_SO, sigma)

    

    print(f"\nAGN: \nR_SO = {R_SO / 3.1e18:.5f} pc, Q_0 = {Q_0:.1e}, <E> = {E_mean:.1f} eV, cross-section = {sigma:.1e} cm^2")
    plotting(pc_radiuses, density_neutral_ratio_values, density_ion_ratio_values, tau, "AGN")
    


if __name__ == "__main__":
    main()
