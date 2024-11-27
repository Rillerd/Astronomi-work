import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

h = 6.6e-27  # erg / s
k_B = 1.38e-16  # erg / K
G = 6.7e-8
M_BH = 21.2 * 2e33
M_star = 40.6*2e33
R_star = 22.3 * 7e10
v_wind = 2.1e8
period = 483840
M_star_loss = 7e-6 * 2e33 / 3.1e7
m_p = 1.67e-24
sigma_T = 6.65e-25
c = 3e10
f = 1/2
year = 60 * 60 * 24 * 365
sigma_SB = 5.7e-5


a = ((G * period**2 * (M_star + M_BH)) / (4 * np.pi**2)) ** (1/3)  # semi-major axis

M_BH_acc = (G**2 * M_BH**2 * M_star_loss / (v_wind ** 4 * a **2 ))  # blackhole accretion rate g / s

M_BH_acc_edd = 8 * np.pi * G * M_BH * m_p / (sigma_T * c)  # blackhole accretion rate at eddington limit g / s

v_orb = np.sqrt((G * (M_star + M_BH)) / a )  # relative orbital velocity of the binary system

R_crit = 2 * G * M_BH / v_wind**2  # critical radius for hte blackhole accretion

j = f * v_orb * R_crit  # angular momentum (spin) for the system

r_keppler = j**2 / (G * M_BH)  # keppler radius

r_ISCO = 2 * G * M_BH / c**2  # ISCO radius for the blackhole
print(R_crit, r_keppler, r_ISCO)


def temperature(radius): return (G * M_BH * M_BH_acc / (8 * np.pi * radius**3 * sigma_SB)) ** (1/4)


def blackbody(nu, radius): 
    T = temperature(radius)
    return 2 * h * nu ** 3 / (c**2 * np.exp((h * nu) / (k_B * T)) - 1) 


def specific_luminosity(nu):
    integrand = lambda radius: 2 * np.pi * radius * np.pi * blackbody(nu, radius)
    L_nu, _ = quad(integrand, r_ISCO, r_keppler)
    return L_nu

def total_luminosity():  # stefan-boltzmans law
    integrand = lambda radius: 2 * np.pi * radius * sigma_SB * temperature(radius)
    L_tot, _ = quad(integrand, r_ISCO, r_keppler)
    return L_tot

def plotting():

    wavelength_values = np.logspace(9, 18.8, 100)
    luminosities = []

    for nu in wavelength_values:
        luminosities.append(specific_luminosity(nu))
        
    plt.loglog(wavelength_values, luminosities)
    plt.title("Specific Luminosity")
    plt.xlabel("$\\nu$ [Hz]")
    plt.ylabel("$L_\\nu$   [erg / s / Hz]")
    plt.show()


def main():
    
    plotting()
    print(f"Total luminosity: {total_luminosity():.1e} erg / s ")


if __name__ == "__main__":
    main()
