import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

# Define Constants

N = np.arange(1,11)
N_0 = 0
C_S = 1E-19
C_D = 10E-19
C_G = 12E-18
C = C_S + C_D + C_G
e = 1.6E-19
E_C = (e ** 2) / C
E_N = 0.1 * E_C

# Define a 1D array for the values for the voltages
V_SD = np.linspace(-0.010, 0.010, 2000)
V_G = np.linspace(0.005, 0.10, 2000)

# Generate 2D array to represent possible voltage combinations

V_SD_grid, V_G_grid = np.meshgrid(V_SD, V_G)

# Define the potentials energies of the source and drain

mu_D = 0  # drain (convention for it to equal zero - grounded)
mu_S = - e * V_SD  # source

I_tot = np.zeros(V_SD_grid.shape)  # Define the total current

def electricPotential(n, V_SD_grid, V_G_grid):
    """
    Function to compute the electric potential of the QDot.
    :param n: the number of electrons in the dot
    :param V_SD_grid: the 2d array of source-drain voltage values
    :param V_G_grid: the 2d array of gate voltage values
    :return: The Electric Potential for adding the nth electron to the dot
    """
    return (n - N_0 - 1/2) * E_C - (E_C / e) * (C_S * V_SD_grid + C_G * V_G_grid) + E_N

def currentChecker(mu_N):
    """
    Function to determne region where current is alowed to flow and where there is a blockade.
    :param mu_N: The electric potential to add the Nth electron to the system.
    :return: The Total allowed current across the grid of voltages. It is either 0, 1, or 2 (units and additive effects of different levels not considered)
    """
    # the algorithm below looks contrived but it removes the need for for loops increasing runtime
    condition1 = mu_N > 0
    condition2 = mu_N < mu_S
    condition3 = V_SD < 0
    condition4 = mu_N < 0
    condition5 = mu_N > mu_S
    condition6 = V_SD > 0

    # Consider both scenarios where mu_D < mu_N < mu_S and mu_S < mu_N < mu_D
    I_1 = (condition1 & condition2 & condition3).astype(int)
    I_2 = (condition4 & condition5 & condition6).astype(int)
    return I_2 + I_1  # combine the result of these possibilities



fig = plt.figure()

for n in N:
    mu_N = electricPotential(n, V_SD_grid, V_G_grid) # get the electric potential energy for ground-state
    allowed_indices = currentChecker(mu_N) # this also equals the current due to ground-state but haven't considered units
    # compute energies for excited states
    mu_N_excited1 = mu_N + n * 0.1 * 0.4 * (E_C / 10)
    mu_N_excited1 = np.multiply(mu_N_excited1, allowed_indices) # does element-wise multiplication with allowed_indices. Ensures current only flows if nth energy state is free.
    mu_N_excited2 = mu_N + n * 0.1 * 2 * (E_C / 10)
    mu_N_excited2 = np.multiply(mu_N_excited2, allowed_indices)

    I_tot += currentChecker(mu_N_excited1) + currentChecker(mu_N_excited2)


I_tot_filter = gaussian_filter(I_tot, sigma=5)


# Plot diamonds

contour = plt.contourf(V_G_grid,V_SD_grid, I_tot_filter, cmap="seismic")
plt.ylabel("$V_{SD}$ (V)")
plt.xlabel("$V_{G}$ (V)")
cbar1 = fig.colorbar(contour)
cbar1.ax.set_ylabel("$I$ (A)", rotation=270)
plt.show()


