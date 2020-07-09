import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.util import random_noise
from random import seed
from random import random

# seed random number generator
seed()

# Define Constants

N = range(1,7)
N_0 = 0
C_S = 10E-19
C_D = 10E-19
C_G = 12E-18
C = C_S + C_D + C_G
e = 1.6E-19
E_C = (e ** 2) / C

# Define a 1D array for the values for the voltages
V_SD = np.linspace(-0.03, 0.03, 2000)
V_G = np.linspace(0.00, 0.10, 2000)

# Generate 2D array to represent possible voltage combinations

V_SD_grid, V_G_grid = np.meshgrid(V_SD, V_G)

# Define the potential energies of the source and drain

mu_D = 0  # drain potential energy (convention for it to equal zero - grounded)
mu_S = - e * V_SD  # source potential energy

I_tot = np.zeros(V_SD_grid.shape)  # Define the total current


def electricPotential(n, V_SD_grid, V_G_grid):

    """
    Function to compute the electric potential of the QDot.
    :param n: the number of electrons in the dot
    :param V_SD_grid: the 2d array of source-drain voltage values
    :param V_G_grid: the 2d array of gate voltage values
    :return: The Electric Potential for adding the nth electron to the dot
    """

    E_N = E_C * random() / n

    return (n - N_0 - 1/2) * E_C - (E_C / e) * (C_S * V_SD_grid + C_G * V_G_grid) + E_N

def currentChecker(mu_N):
    """
    Function to determne region where current is allowed to flow and where there is a blockade.
    Finds indexes corresponding to values of V_SD and V_G for which current can flow from source-drain or drain-source
    :param mu_N: The electric potential to add the Nth electron to the system.
    :return: The Total allowed current across the grid of voltages. It is either 0, 1, or 2 (units and additive effects of different levels not considered)
    """
    # the algorithm below looks contrived but it removes the need for for loops increasing runtime
    # it checks whether the potential energy of the electron state is between the source and drain
    condition1 = mu_N > 0
    condition2 = mu_N < mu_S
    condition3 = V_SD < 0
    condition4 = mu_N < 0
    condition5 = mu_N > mu_S
    condition6 = V_SD > 0

    # Consider both scenarios where mu_D < mu_N < mu_S and mu_S < mu_N < mu_D
    I_1 = (condition1 & condition2 & condition3).astype(int)
    I_2 = (condition4 & condition5 & condition6).astype(int)
    return I_2 + I_1  # combine the result of these possibilities.


fig = plt.figure()

offsets = np.array([[E_C * 1 / 10,  E_C * 2 / 10],
                       [E_C * 1 / 10,  E_C * 2 / 10]]) # If doing a random implementation that changes
                                                                # for each n need to cache values

'''offsets:
       [ES(N), LS(N)],
       [ES(N-1), LS(N-1)] represents the height of each level about the ground state of that level'''

for n in N:
    # potential energy of  ground to ground transition GS(N-1) -> GS(N)
    mu_N = electricPotential(n, V_SD_grid, V_G_grid)

    allowed_indices = currentChecker(mu_N)

    if n != 1:

        # The transitions from this block are to/from excited states

        # potential energy of  ground to excited transition GS(N-1) -> ES(N)
        mu_N_transition1 = mu_N + offsets[0,0]
        mu_N_transition1 = np.multiply(mu_N_transition1, allowed_indices)
        '''This does element-wise multiplication
         with allowed_indices. Ensures current only flows / transition occurs only if ground state is free'''

        # potential energy of excited to ground transition GS(N-1) -> LS(N)
        mu_N_transition2 = mu_N + offsets[0, 1]
        mu_N_transition2 = np.multiply(mu_N_transition2, allowed_indices)

        # potential energy of excited to ground transition ES(N-1) -> GS(N)
        mu_N_transition3 = mu_N - offsets[1,0]
        mu_N_transition3 = np.multiply(mu_N_transition3, allowed_indices)

        # potential energy of excited to ground transition ES(N-1) -> ES(N)
        mu_N_transition4 = mu_N - offsets[1, 0] + offsets[0, 0]
        mu_N_transition4 = np.multiply(mu_N_transition4, allowed_indices)

        # potential energy of excited to ground transition ES(N-1) -> LS(N)
        mu_N_transition5 = mu_N - offsets[1, 0] + offsets[0, 1]
        mu_N_transition5 = np.multiply(mu_N_transition5, allowed_indices)

        # potential energy of excited to ground transition LS(N-1) -> GS(N)
        mu_N_transition6 = mu_N - offsets[1,1]
        mu_N_transition6 = np.multiply(mu_N_transition6, allowed_indices)

        # potential energy of excited to ground transition LS(N-1) -> ES(N)
        mu_N_transition7 = mu_N - offsets[1,1] + offsets[0,0]
        mu_N_transition7 = np.multiply(mu_N_transition7, allowed_indices)

        # potential energy of excited to ground transition LS(N-1) -> LS(N)
        mu_N_transition8 = mu_N - offsets[1, 1] + offsets[0, 1]
        mu_N_transition8 = np.multiply(mu_N_transition8, allowed_indices)

        I_tot += currentChecker(mu_N_transition1) + currentChecker(mu_N_transition2) \
        + currentChecker(mu_N_transition3) + currentChecker(mu_N_transition4) + currentChecker(mu_N_transition5) \
        + currentChecker(mu_N_transition6) + currentChecker(mu_N_transition7) + currentChecker(mu_N_transition8)

      # If statement is used as only transition to ground state is allowed for N = 1 from ground state

    I_tot += currentChecker(mu_N)
    print(I_tot)

I_tot_filter = gaussian_filter(I_tot, sigma=0)  # Apply Gaussian Filter. The greater sigma the more blur.
#I_tot_filter = random_noise(I_tot_filter, mode='s&p',amount=0.01)

# Plot diamonds

contour = plt.contourf(V_G_grid,V_SD_grid, I_tot_filter, cmap="seismic")
plt.ylabel("$V_{SD}$ (V)")
plt.xlabel("$V_{G}$ (V)")
cbar1 = fig.colorbar(contour)
cbar1.ax.set_ylabel("$I$ (A)", rotation=270)
plt.show()


