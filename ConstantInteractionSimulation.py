import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from random import seed # generates seed for random number generator
from random import random  # random generates a random number between 0 and 1
from random import uniform # generates random float between specified range
from datetime import datetime
from skimage.util import random_noise



# Define Constants

N = range(1,8)
N_0 = 0
e = 1.6E-19

# Define a 1D array for the values for the voltages
V_SD_max = 0.1
V_G_min = 0.0
V_G_max = 0.5
V_SD = np.linspace(- V_SD_max, V_SD_max, 1500)
V_G = np.linspace(V_G_min, V_G_max, 1500)

# Generate 2D array to represent possible voltage combinations

V_SD_grid, V_G_grid = np.meshgrid(V_SD, V_G)

# Define the potential energies of the source and drain

mu_D = 0  # drain potential energy (convention for it to equal zero - grounded)
mu_S = - e * V_SD  # source potential energy

I_tot = np.zeros(V_SD_grid.shape)  # Define the total current
I_ground = np.zeros(V_SD_grid.shape)  # Define the ground transition current
E_N_previous = 0 # stores previous E_N value
V_G_start = 0 # start of current diamond
diamond_starts = np.zeros((1,len(N))) # numpy array to store the store positions of each diamond along x-axis

def electricPotential(n, V_SD_grid, V_G_grid):

    """
    Function to compute the electric potential of the QDot.
    :param n: the number of electrons in the dot
    :param V_SD_grid: the 2d array of source-drain voltage values
    :param V_G_grid: the 2d array of gate voltage values
    :return: The Electric Potential for adding the nth electron to the dot
    """

    E_N = E_C*(((n)**2-(n-1)**2)/n*5+random()/9*n)  # arbitrary random formula used to increase diamond width as more electrons are added

    return E_N, (n - N_0 - 1/2) * E_C - (E_C / e) * (C_S * V_SD_grid + C_G * V_G_grid) + E_N

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
    return I_1 + I_2  # combine the result of these possibilities.

for k in range(1,50):

    # seed random number generator
    seed(datetime.now())  # use current time as random number seed

    C_S = 10E-19 * uniform(0.5, 1)  # Uniform used for some random variation
    C_D = 10E-19 * uniform(0.5, 1)
    C_G = 12E-18 * uniform(0.4, 1)
    C = C_S + C_D + C_G
    E_C = (e ** 2) / C

    fig1 = plt.figure()

    Estate_height_previous = 0  # stores previous various excited energy height above ground level

    for n in N:
        Estate_height = uniform(0.1, 0.5) * E_C
        Lstate_height = uniform(0.5, 0.8) * E_C

        # potential energy of ground to ground transition GS(N-1) -> GS(N)
        E_N, mu_N = electricPotential(n, V_SD_grid, V_G_grid)

        # Indices where current can flow for  GS(N-1) -> GS(N) transitions
        allowed_indices = current_ground = currentChecker(mu_N)
        delta_E_N = E_N - E_N_previous  # Not sure on exact definition yet
        delta_V_G = e/C_G + delta_E_N * C /(e *C_G) # Width of current diamond

        if n ==1:
            V_G_start = (e/C_G) * (E_N / E_C + 1/2)  # start of first diamond / start of current diamond
            diamond_starts[0,n-1] = V_G_start
            # potential energy of ground to excited transition GS(N-1) -> ES(N)
            mu_N_transition1 = mu_N + Estate_height

            mu_N_transition1 = np.multiply(mu_N_transition1, allowed_indices)
            '''This does element-wise multiplication
                    with allowed_indices. Ensures current only flows / transition occurs only if ground state is free'''

            current_transition1 = currentChecker(mu_N_transition1)  # additional check if current can flow
            random_current_transition1 = current_transition1 * uniform(0.5, 2)
            '''random_current_transition1 adds some randomness to the current value'''

            I_tot += random_current_transition1

        elif n != 1:
            V_G_start += delta_V_G  # update so start of current diamond
            diamond_starts[0, n-1] = V_G_start
            # The transitions from this block are to/from excited states

            # potential energy of  ground to excited transition GS(N-1) -> ES(N)
            mu_N_transition1 = mu_N + Estate_height
            mu_N_transition1 = np.multiply(mu_N_transition1, allowed_indices)
            current_transition1 = currentChecker(mu_N_transition1)  # additional check if current can flow
            random_current_transition1 = current_transition1 * uniform(0.2, 2)
            '''This does element-wise multiplication
             with allowed_indices. Ensures current only flows / transition occurs only if ground state is free'''

            # potential energy of excited to ground transition GS(N-1) -> LS(N)
            mu_N_transition2 = mu_N + Lstate_height
            mu_N_transition2 = np.multiply(mu_N_transition2, allowed_indices)
            current_transition2 = currentChecker(mu_N_transition2)  # additional check if current can flow
            random_current_transition2 = current_transition2 * uniform(0.2, 2)

            # potential energy of excited to ground transition ES(N-1) -> GS(N)
            mu_N_transition3 = mu_N - Estate_height_previous
            mu_N_transition3 = np.multiply(mu_N_transition3, allowed_indices)
            current_transition3 = currentChecker(mu_N_transition3)  # additional check if current can flow
            random_current_transition3 = current_transition3 * uniform(0.2, 2)

            # potential energy of excited to ground transition ES(N-1) -> ES(N)
            mu_N_transition4 = mu_N - Estate_height_previous + Estate_height
            mu_N_transition4 = np.multiply(mu_N_transition4, allowed_indices)
            current_transition4 = currentChecker(mu_N_transition4)  # additional check if current can flow
            random_current_transition4 = current_transition4 * uniform(0.2, 2)

            # potential energy of excited to ground transition ES(N-1) -> LS(N)
            mu_N_transition5 = mu_N - Estate_height_previous + Lstate_height
            mu_N_transition5 = np.multiply(mu_N_transition5, allowed_indices)
            current_transition5 = currentChecker(mu_N_transition5)  # additional check if current can flow
            random_current_transition5 = current_transition5 * uniform(0.2, 2)

            # potential energy of excited to ground transition LS(N-1) -> GS(N)
            mu_N_transition6 = mu_N - Lstate_height_previous
            mu_N_transition6 = np.multiply(mu_N_transition6, allowed_indices)
            current_transition6 = currentChecker(mu_N_transition6)  # additional check if current can flow
            random_current_transition6 = current_transition6 * uniform(0.2, 2)

            # potential energy of excited to ground transition LS(N-1) -> ES(N)
            mu_N_transition7 = mu_N - Lstate_height_previous + Estate_height
            mu_N_transition7 = np.multiply(mu_N_transition7, allowed_indices)
            current_transition7 = currentChecker(mu_N_transition7)  # additional check if current can flow
            random_current_transition7 = current_transition7 * uniform(0.2, 2)

            # potential energy of excited to ground transition LS(N-1) -> LS(N)
            mu_N_transition8 = mu_N - Lstate_height_previous + Lstate_height
            mu_N_transition8 = np.multiply(mu_N_transition8, allowed_indices)
            current_transition8 = currentChecker(mu_N_transition8)  # additional check if current can flow
            random_current_transition8 = current_transition8 * uniform(0.2, 2)

            I_tot += random_current_transition1 + random_current_transition2 + random_current_transition3 + \
                     random_current_transition4 + random_current_transition5 + random_current_transition6 + \
                     random_current_transition7 + random_current_transition8

        # If statement is used as only transition to ground state is allowed for N = 1 from ground state

        I_tot += current_ground
        I_ground += current_ground

        # update 'previous' variables to previous values
        E_N_previous = E_N
        Estate_height_previous = Estate_height
        Lstate_height_previous = Lstate_height

    #print("Start of each diamond along x-axis (V): " + str(diamond_starts))
    I_tot = I_tot / np.max(I_tot) # scale current values


    I_tot_filter = random_noise(I_tot, mode='gaussian')
    I_tot_filter = gaussian_filter(I_tot, sigma=5)  # Apply Gaussian Filter. The greater sigma the more blur.

    # Plot diamonds

    contour = plt.contourf(V_G_grid,V_SD_grid, I_tot_filter, cmap="seismic", levels = np.linspace(0,1,100)) # draw contours of diamonds
    '''The extra diamonds arose out of the fact that there was a small number of contour levels added in 
    levels attribute to fix this so 0 current was grouped with the small current values '''


    #plt.ylabel("$V_{SD}$ (V)")
    #plt.xlabel("$V_{G}$ (V)")
    plt.ylim([-V_SD_max, V_SD_max])
    plt.xlim([V_G_min, V_G_max])
    #cb = fig1.colorbar(contour)
    #cb.ax.set_ylabel("$I$ (arb. units)", rotation=270, labelpad=20)
    #plt.title("Quantum Dot Simulation")
    plt.axis("off")

    plt.savefig("./Training_Input/input_{0}.png".format(k),bbox_inches='tight', pad_inches=0.0)

    plt.close()
    # Compute negative and positive slopes of diamonds for drawing edges

    positive_slope = C_G / (C_G + C_D)
    negative_slope = - C_G / C_S

    fig2 = plt.figure()

    for i in range(len(N)-1):  # need -1 as block would attempt to access index N otherwise and it doesn't exist
        # positive grad. top-left
        x_final = (positive_slope * diamond_starts[0, i] - negative_slope * diamond_starts[0, i + 1]) / (positive_slope - negative_slope)  # analytical formula derived by equating equations of lines
        x_values = [diamond_starts[0, i], x_final]
        y_final = positive_slope * (x_final - diamond_starts[0, i])
        y_values = [0, y_final]
        plt.plot(x_values, y_values, '-k')

        # negative grad. top-right
        x_values = [x_final, diamond_starts[0, i + 1]]
        y_values = [y_final, 0]
        plt.plot(x_values, y_values, '-k')

        # positive grad. bottom-right
        x_final = (positive_slope * diamond_starts[0, i + 1] - negative_slope * diamond_starts[0, i]) / (positive_slope - negative_slope)
        x_values = [diamond_starts[0, i + 1], x_final]
        y_final = positive_slope * (x_final - diamond_starts[0, i + 1])
        y_values = [0, y_final]
        plt.plot(x_values, y_values, '-k')

        # negative grad. bottom-left
        x_values = [x_final, diamond_starts[0, i]]
        y_values = [y_final, 0]
        plt.plot(x_values, y_values, '-k')


    plt.ylim([-V_SD_max, V_SD_max])
    plt.xlim([V_G_min, V_G_max])
    plt.axis("off")

    plt.savefig("./Training_Output/output_{0}.png".format(k),bbox_inches='tight', pad_inches=0.0)

    plt.close()






