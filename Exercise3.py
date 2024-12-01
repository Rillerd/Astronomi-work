import matplotlib.pyplot as plt
import numpy as np


### Constants ###
c = 3e10  # cm / s
h = 6.6e-27  # erg / s
k_B = 1.38e-16  # erg / K
erg = 1e-12  # convert from eV to erg

### Atomic constants for O III transitions ###
omega_1S_3P = 1.18  # Omega_24
omega_1S_1D = 0.62  # Omega_34
omega_1D_3P = 2.17  # Omega_23
weight_1S = 1  # n = 4
weight_1D = 5  # n = 3
weight_3P2 = 5  # n = 2
weight_3P1 = 3  # n = 1
A_1S_3P = 2.15e-1  # s^-1  A_41
A_1S_1D = 1.71  # s^-1  A_43
A_1D_3P2 = 1.81e-2  # S^-1  A_32
A_1D_3P1  = 6.21e-3  # s^-1  A_31
energy_difference_1S_3P = 5.34 * erg  # erg    E_41 = E_4 - E_1
energy_difference_1S_1D = 2.84 * erg # erg    E_43 = E_4- E_3
energy_difference_1D_3P2 = 2.48 * erg # erg   E_32 = E_3 - E_2
energy_difference_1D_3P1 = 2.5 * erg # erg    E_31 = E_3 - E_1


def collisional_excitation_rate(omega, weight, energy_difference, T, n_e): 
    return 8.629e-8 * (T/1e4)**(-1/2) * (omega / weight) * np.exp(-energy_difference / (k_B * T)) * n_e


def emissivity_1S_1D(n0, T):
    
    n_e = 2 * n0

    ### Collision excitaion rates ###
    C_14 = collisional_excitation_rate(omega_1S_3P, weight_3P1, energy_difference_1S_3P, T, n_e)
    C_34 = collisional_excitation_rate(omega_1S_1D, weight_1D, energy_difference_1S_1D, T, n_e)

    ### Collision deexcitation rates ###
    C_41 = (weight_3P1 / weight_1S) * C_14 * np.exp(energy_difference_1S_3P / (k_B * T))
    C_43 = (weight_1D / weight_1S) * C_34 * np.exp(energy_difference_1S_1D / (k_B * T))

    total_population_of_S_state = n0 * C_14
    probability_of_radiative_emission = A_1S_1D / (A_1S_1D + A_1S_3P + C_41 + C_43)

    return energy_difference_1S_1D * total_population_of_S_state * probability_of_radiative_emission



def emissivity_1S_3P(n0, T):

    n_e = 2 * n0

    ### Collisional excitation rates ###
    C_14 = collisional_excitation_rate(omega_1S_3P, weight_3P1, energy_difference_1S_3P, T, n_e)
    C_34 = collisional_excitation_rate(omega_1S_1D, weight_1D, energy_difference_1S_1D, T, n_e)

    ### Collisional de-excitation rates ###
    C_41 = (weight_3P1 / weight_1S) * C_14 * np.exp(energy_difference_1S_3P / (k_B * T))
    C_43 = (weight_1D / weight_1S) * C_34 * np.exp(energy_difference_1S_1D / (k_B * T))

    total_population_of_S_state = n0 * C_14
    probability_of_radiative_emission = A_1S_3P / (A_1S_3P + A_1S_1D + C_43 + C_41)

    return energy_difference_1S_3P * total_population_of_S_state * probability_of_radiative_emission



def emissivity_1D_3P2(n0, T):

    n_e = 2 * n0

    ### Collisional excitation rates ###
    C_14 = collisional_excitation_rate(omega_1S_3P, weight_3P1, energy_difference_1S_3P, T, n_e)
    C_13 = collisional_excitation_rate(omega_1D_3P, weight_3P2, energy_difference_1D_3P2, T, n_e)
    C_34 = collisional_excitation_rate(omega_1S_1D, weight_1D, energy_difference_1S_1D, T, n_e)
    C_23 = collisional_excitation_rate(omega_1D_3P, weight_3P2, energy_difference_1D_3P2, T, n_e)

    ### Collisional de-excitation rates ###
    C_41 = (weight_3P1 / weight_1S) * C_14 * np.exp(energy_difference_1S_3P / (k_B * T))
    C_43 = (weight_1D / weight_1S) * C_34 * np.exp(energy_difference_1S_1D / (k_B * T))
    C_32 = (weight_3P2 / weight_1D) * C_23 * np.exp(energy_difference_1D_3P2 / (k_B * T))
    C_31 = (weight_3P1 / weight_1D) * C_13 * np.exp(energy_difference_1D_3P1 / (k_B * T))

    total_population_of_D_state = n0 * (C_13 + C_14 * (C_43 + A_1S_1D) / (A_1S_1D + A_1S_3P + C_43 + C_41))
    probability_of_radiative_emission = A_1D_3P2 / (A_1D_3P2 + A_1D_3P1 + C_32 + C_31) 

    return energy_difference_1D_3P2 * total_population_of_D_state * probability_of_radiative_emission


def emissivity_1D_3P1(n0, T):
    
    n_e = 2 * n0

    ### Collisional excitation rates ###
    C_14 = collisional_excitation_rate(omega_1S_3P, weight_3P1, energy_difference_1S_3P, T, n_e)
    C_13 = collisional_excitation_rate(omega_1D_3P, weight_3P1, energy_difference_1D_3P1, T, n_e)
    C_23 = collisional_excitation_rate(omega_1D_3P, weight_3P2, energy_difference_1D_3P2, T, n_e)
    C_34 = collisional_excitation_rate(omega_1S_1D, weight_1D, energy_difference_1S_1D, T, n_e)

    ### Collisional de-excitation rates ###
    C_41 = (weight_3P1 / weight_1S) * C_14 * np.exp(energy_difference_1S_3P / (k_B * T))
    C_32 = (weight_3P2 / weight_1D) * C_23 * np.exp(energy_difference_1D_3P2 / (k_B * T))
    C_31 = (weight_3P1 / weight_1D) * C_13 * np.exp(energy_difference_1D_3P1 / (k_B * T))
    C_43 = (weight_1D / weight_1S) * C_34 * np.exp(energy_difference_1S_1D / (k_B * T))
     
    total_population_of_D_state =  n0 * (C_13 + C_14 *  (A_1S_1D + C_43) / (A_1S_1D + A_1S_3P + C_41 * C_43))
    probability_of_radiative_emission = A_1D_3P1 / (A_1D_3P1 + A_1D_3P2 + C_32 + C_31)
                
    return energy_difference_1D_3P1 * total_population_of_D_state * probability_of_radiative_emission


def emissivity_values(n0):

    j_1S_1D_values = []
    j_1S_3P_values = []
    j_1D_3P2_values = []
    j_1D_3P1_values = []
    emissivity_fraction = []

    temperatures = np.linspace(100, 1e5, 1000)
    for T in temperatures:
        j_1S_1D_values.append(emissivity_1S_1D(n0, T))
        j_1S_3P_values.append(emissivity_1S_3P(n0, T))
        j_1D_3P2_values.append(emissivity_1D_3P2(n0, T))
        j_1D_3P1_values.append(emissivity_1D_3P1(n0, T))
        emissivity_fraction.append(emissivity_1S_1D(n0, T) / (emissivity_1D_3P2(n0, T) + emissivity_1D_3P1(n0, T)))

    return temperatures, emissivity_fraction, j_1S_1D_values, j_1S_3P_values, j_1D_3P2_values, j_1D_3P1_values


def main():

    density_values = [1, 1e4, 1e6, 1e8]  # cm^-3

    for density in density_values:
        temp, emissivity_fraction, _,_,_,_ = emissivity_values(density)
        plt.plot(temp, emissivity_fraction, label =f"n0 = {density:.0e} cm⁻³" )

    plt.yscale("log")
    plt.ylabel("Emissivity fraction")
    plt.xlabel("T [K]")
    plt.ylim(1e-3, 0.05)
    plt.xlim(2e3, 7e3)
    plt.legend()
    plt.title("Emissivity fraction of OIII 4363$\\lambda$ / (4959$\\lambda$+5007$\\lambda$)")
    plt.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    plt.show()


if __name__ == "__main__":
    main()

