import numpy as np
import scipy as sci
from scipy.integrate import odeint
from matplotlib import pyplot as plt

# REAL PARAMETERS
G_real = 6.6742e-20  # Real Gravitational Constant
m1 = 5.974e+24  # Mass of Earth in kg
m2 = 7.348e+22  # Mass of Moon in kg
R_earth = 6378.0  # Radius of Earth in km
R_moon = 1737  # Radius of Moon in km
r12_real = 384400  # Actual distance between Earth and Moon
#--------------------------------------------

# NORMALIZED PARAMETERS
G = 1  # Normalized Gravitational Constant
r12 = 1  # Normalized distance between Earth-to-Moon
M = 1  # Normalized total mass of system
M1 = m1 / (m1 + m2)  # Normalized mass of the Earth
M2 = 1 - M1  # Normalized mass of the Moon
x1 = -M2
x2 = 1 - M2
Omega = np.sqrt((G * (M1 + M2)) / (r12 ** 3))  # Normalized Angular Velocity of rotating system = 1 rad/s
#----------------------------------------------

# PARAMETERS TO CONVERT FROM NORMALIZED TO NON-NORMALIZED SYSTEM
x1_real = -M2 * r12_real  # Same as x1 above, just no longer non-dimensional
x2_real = M1 * r12_real  # Same as x2 above, just no longer non-dimensional
Omega_real = np.sqrt((G_real * (m1 + m2)) / r12_real ** 3)  # Actual Angular Velocity of rotating Earth-Moon system
mu1 = G_real * m1  # Real gravitational parameter of Earth
mu2 = G_real * m2  # Real gravitational parameter of Moon
#----------------------------------------------

# Set initial conditions
L1_point = 0.8369 * r12_real  # Real location of L1 point (km)
x = 0.75 * r12_real  # Initial x-position os simulation
y = 0  # Initial y-position os simulation

C_constant = 3.1725  # <--- Actual Jacobi Constant goes here

Jacobi_constant = -(C_constant / 2) - 0.07775  # Jacbi Constant for Simulation; This value is used for the equations. However, it crresponds to the value of C used to plot the ZVC  = -2C
r1_real = np.sqrt((x + M2 * r12_real) ** 2 + y ** 2)  # Earth-to-satellite vector in km
r2_real = np.sqrt((x - M1 * r12_real) ** 2 + y ** 2)  # Moon-to-satellite vector in km

velocity0 = np.sqrt((Omega_real ** 2) * (x ** 2 + y ** 2) + ((2 * mu1) / r1_real) + ((2 * mu2) / r2_real) + (2 * Jacobi_constant))  # initial velocity in based on position and Jacobi Constant

vx = velocity0 * np.cos(15 * (np.pi / 180))  # Arbitrarily split velocity into x-component
vy = velocity0 * np.sin(15 * (np.pi / 180))  # Arbitrarily split velocity into y-component

X0 = np.array([x, y, vx, vy])  # Initial condition array

# TIME FOR SIMULATION
seconds_in_day = 86400
t_initial = 0  # Initial time of simulation
t_final = 100 * seconds_in_day  # Total simulation time
dt = 5  # Individual time step
steps = t_final / dt  # Total steps in simulation, equally separated
tspan = np.linspace(t_initial, t_final, steps)

print("Jacobi Constant: " + str(C_constant))
print("Initial velocity from Jacobi Constant: " + str(velocity0) + " km/s")


def get_x_dot(X0, t_span):  # Get state space to be integrated: [Vx, Vy, Accel_X, Accel_Y]
    r1_real = np.sqrt((X0[0] + M2 * r12_real) ** 2 + X0[1] ** 2)  # Updated Earth-to-satellite vector in km
    r2_real = np.sqrt((X0[0] - M1 * r12_real) ** 2 + X0[1] ** 2)  # Updated Moon-to-satellite vector in km
    x_dot = X0[2]  # We already have x_dot --> preiously calculated vx
    y_dot = X0[3]  # We already have y_dot --> previously calculated vy
    x_ddot = (2 * Omega_real * X0[3]) + (Omega_real ** 2 * X0[0]) - ((mu1 * (X0[0] - x1_real)) / r1_real ** 3) - ((mu2 * (X0[0] - x2_real)) / r2_real ** 3)  # X-component of 3-Body gravitational acceleration; from 3-Body Equation of motion
    y_ddot = -(2 * Omega_real * X0[2]) + (Omega_real ** 2 * X0[1]) - ((mu1 / r1_real ** 3) + (mu2 / r2_real ** 3)) * X0[1]  # Y-component of 3-Body gravitational acceleration; from 3-Body Equation of motion
    X_dot = np.array([x_dot, y_dot, x_ddot, y_ddot])  # return new state space
    return X_dot


x_axis = np.linspace(-1.5, 1.5, 1001)  # Create grid
y_axis = np.linspace(-1.5, 1.5, 1001)  # Create grid


def CR3BP_Potential(M1, M2, Omega, x1, x2, x_axis, y_axis):  # Calculate pseudo potential
    U = np.zeros((len(x_axis), len(y_axis)))
    x_counter = 0
    y_counter = 0
    for i in x_axis:
        for j in y_axis:
            r1 = np.sqrt((i - x1) ** 2 + j ** 2)
            r2 = np.sqrt((i - x2) ** 2 + j ** 2)
            U[y_counter, x_counter] = -((Omega ** 2) / 2) * (i ** 2 + j ** 2) - (M1 / r1) - (M2 / r2)
            y_counter = y_counter + 1
        y_counter = 0
        x_counter = x_counter + 1
    return U


sol = odeint(get_x_dot, X0, tspan)  # INTEGRATE STATE SPACE = INTEGRATE 3-BODY EQUATIONS OF MOTION TO GET TRAJECTORY


def plot_trajectory():
    C = C_constant  # Jacobi constant for ZVC
    U = CR3BP_Potential(M1, M2, Omega, x1, x2, x_axis, y_axis)  # Calculate Potential
    plt.contour(y_axis, x_axis, U, [-C / 2, C], colors=('k'), linewidths=0.75, linestyles='solid')  # Plot ZVC, Nondimesional

    ax = plt.gca()
    ax.plot(sol[:, 0] / r12_real, sol[:, 1] / r12_real, color='blue', linewidth=0.75)  # Normalize our trajectory to the non-dimensional coordinate system by dividing by r12_real. Plot
    ax.plot(x1, 0, '*')  # Earth, Nondimesional
    ax.plot(x2, 0, '*')  # Moon, Nondimesional
    circle1 = plt.Circle((x2, 0), R_moon / r12_real, color='r', fill=False)  # Circle around Moon representing its surface, Nondimesional
    circle2 = plt.Circle((x1, 0), R_earth / r12_real, color='b', fill=False)  # Circle around Earth representing its surface, Nondimesional
    ax.plot(L1_point / r12_real, 0, '.', c='k')  # L1 point, Nondimesional
    ax.plot(1.146765, 0, '.', c='k')  # L2 point, Nondimesional
    ax.plot(-1.004167, 0, '.', c='k')  # L3 point, Nondimesional
    ax.plot(0.5 + x1, np.sqrt(3) / 2.0, '.', c='k')  # L4 point, Nondimesional
    ax.plot(0.5 - x1, -np.sqrt(3) / 2.0, '.', c='k')  # L5 Point, Nondimesional
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    plt.title("Trajectory of Satellite within ZVC with a Jacobi Constant C = " + str(C))
    plt.grid()
    plt.show()


plot_trajectory()

# Get Closest approaches and see if Jacobi constant actually is constant
moon_position = x2_real
earth_position = x1_real
distance_to_moon = np.zeros((len(sol[:, 0])))
distance_to_earth = np.zeros((len(sol[:, 0])))
new_jacobi = np.zeros((len(sol[:, 0])))
for n in range(len(sol[:, 0])):
    distance_to_moon[n] = np.sqrt(((sol[n, 0]) - moon_position) ** 2 + ((sol[n, 1]) ** 2)) - R_moon  # Calculate distance of each point from the surface of the Moon
    distance_to_earth[n] = np.sqrt(((sol[n, 0]) - earth_position) ** 2 + ((sol[n, 1]) ** 2)) - R_earth  # Calculate distance of each point from the surface of the Earth
    velo = np.sqrt(sol[n, 2] ** 2 + sol[n, 3] ** 2)  # Get velocity of at each point
    r1_jacobi = np.sqrt((sol[n, 0] + M2 * r12_real) ** 2 + sol[n, 1] ** 2)  # Updated Earth-to-satellite vector in km
    r2_jacobi = np.sqrt((sol[n, 0] - M1 * r12_real) ** 2 + sol[n, 1] ** 2)  # Updated Moon-to-satellite vector in km
    new_jacobi[n] = -((velo ** 2) - ((Omega_real ** 2) * (sol[n, 0] ** 2 + sol[n, 1] ** 2)) - ((2 * mu1) / (r1_jacobi)) - ((2 * mu2) / (r2_jacobi))) - 2 * 0.07775  # Calculate Jacobi constant for each point
jacobi_tolerance = np.max(new_jacobi) - np.min(new_jacobi)
moon_closest_approach = np.min(distance_to_moon)
earth_closest_approach = np.min(distance_to_earth)
print("Closest approach to Moon's surface: " + str(moon_closest_approach) + " km")
print("Closest approach to Earth's surface: " + str(earth_closest_approach) + " km")
print("Jacobi Constant is constant to a tolerance of " + str(jacobi_tolerance))


def plot_Jacobi():  # plot the Jacobi constant for each point in time
    plt.plot(tspan, new_jacobi[:])
    plt.xlabel("Time (s)")
    plt.ylabel("Jacobi Constant")
    plt.grid()
    plt.show()


plot_Jacobi()
