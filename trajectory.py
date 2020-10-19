import numpy as np
import scipy as sci
import scipy.integrate
from matplotlib import pyplot as plt

G_real = 6.6742e-20
m1 = 5.974e+24  # Mass of Earth in kg
m2 = 7.348e+22  # Mass of Moon in kg
M = 1  # Normalized total mass of system
M1 = m1 / (m1 + m2)  # Normalized mass of the Earth
M2 = 1 - M1  # Normalized mass of the Moon
r12_real = 384400  # Actual distance between Earth and Moon
x1 = -M2
x2 = 1 - M2

# Set initial conditions
R_earth = 6378.0  # km
d0 = 200  # km
r0 = R_earth + d0
phi = -90  # degrees
gamma = 19  # degrees

# Create auxillery variables
x1_real = -M2 * r12_real
x2_real = M1 * r12_real
Omega_real = np.sqrt((G_real * (m1 + m2)) / r12_real ** 3)
mu1 = G_real * m1 + x1_real
mu2 = G_real * m2
#print(x1_real, x2_real, Omega_real, mu1, mu2)

x = r0 * np.cos(phi) + x1_real
y = r0 * np.sin(phi)

r1_real = np.sqrt((x + M2 * r12_real) ** 2 + y ** 2)
r2_real = np.sqrt((x - M1 * r12_real) ** 2 + y ** 2)

Jacobi_constant = 3.1725

velocity0 = np.sqrt((Omega_real ** 2) * (x ** 2 + y ** 2) + ((2 * mu1) / r1_real) + ((2 * mu2) / r2_real) + (2 * Jacobi_constant))  # initial velocity in km/s

print("Jacobi Constant: " + str(Jacobi_constant))
print("Initial velocity from Jacobi Constant: " + str(velocity0))

vx = velocity0 * (np.sin(gamma) * np.cos(phi) - np.cos(gamma) * np.sin(phi))
vy = velocity0 * (np.sin(gamma) * np.sin(phi) + np.cos(gamma) * np.cos(phi))

X0 = np.array([x, y, vx, vy])

t_initial = 0

t_final = 1.5 * 86400  # Days * seconds/day
dt = 5


def get_x_dot(X0, t_initial, dt, x1_real, x2_real, Omega_real, mu1, mu2, r12_real, M1, M2):
    r1_real = np.sqrt((X0[0] + M2 * r12_real) ** 2 + X0[1] ** 2)
    r2_real = np.sqrt((X0[0] - M1 * r12_real) ** 2 + X0[1] ** 2)
    y1_dot = X0[2]
    y2_dot = X0[3]

    y3_dot = -(2 * Omega_real * X0[3]) + (Omega_real ** 2 * X0[0]) - ((mu1 * (X0[0] - x1_real)) / r1_real ** 3) - ((mu2 * (X0[0] - x2_real)) / r2_real ** 3)

    y4_dot = -(2 * Omega_real * X0[2]) + (Omega_real ** 2 * X0[1]) - ((mu1 / r1_real ** 3) + (mu2 / r2_real ** 3)) * X0[1]
    X_dot = np.array([y1_dot, y2_dot, y3_dot, y4_dot])
    # print(X_dot)
    return X_dot


def RK4_algorithm(X0, t_initial, dt, x1_real, x2_real, Omega_real, mu1, mu2, r12_real, M1, M2):
    h = dt
    k1 = get_x_dot(X0, t_initial, dt, x1_real, x2_real, Omega_real, mu1, mu2, r12_real, M1, M2)
    k2 = get_x_dot(X0 + h * k1 / 2, t_initial + h / 2, dt, x1_real, x2_real, Omega_real, mu1, mu2, r12_real, M1, M2)
    k3 = get_x_dot(X0 + h * k2 / 2, t_initial + h / 2, dt, x1_real, x2_real, Omega_real, mu1, mu2, r12_real, M1, M2)
    k4 = get_x_dot(X0 + h * k3, t_initial + h, dt, x1_real, x2_real, Omega_real, mu1, mu2, r12_real, M1, M2)

    X_update = X0 + ((h / 6) * (k1 + 2 * k2 + 2 * k3 + k4))
    t_update = t_initial + h

    return X_update, t_update


def RK4(X0, t_final, t_initial, dt, x1_real, x2_real, Omega_real, mu1, mu2, r12_real, M1, M2):
    steps = t_final / dt
    trajectory = np.zeros((int(steps), 4))
    for i in range(int(steps)):
        X_star, t_update = RK4_algorithm(X0, t_initial, dt, x1_real, x2_real, Omega_real, mu1, mu2, r12_real, M1, M2)
        X0 = X_star
        t_initial = t_update
        trajectory[i, 0] = X_star[0]
        trajectory[i, 1] = X_star[1]
        trajectory[i, 2] = X_star[2]
        trajectory[i, 3] = X_star[3]
    X_New = X_star
    return X_New, t_update, trajectory


def rkf45(tspan, y0):
    a = np.array([0, 1. / 4., 3. / 8., 12. / 13., 1, 1. / 2.])

    b = np.array([[0, 0, 0, 0, 0], [1. / 4., 0, 0, 0, 0], [3. / 32., 9. / 32., 0, 0, 0], [1932. / 2197., -7200. / 2197., 7296. / 2197., 0, 0], [439. / 216., -8, 3680. / 513., -845. / 4104., 0], [-8. / 27., 2, -3544. / 2565., 1859. / 4104., -11. / 40]])

    c4 = np.array([[25. / 216, 0, 1408. / 2565., 2197. / 4104., -1. / 5., 0]])

    c5 = np.array([[16. / 135., 0, 6656. / 12825., 28561. / 56430., -9. / 50., 2. / 55.]])

    t0 = tspan[0]
    tf = tspan[1]
    t = t0
    y_loop = y0
    tout = np.array([t])
    yout = np.array([np.transpose(y_loop)])
    h = 10  # assumed initial time step
    tolerance = 1e-8
    f = np.zeros((4, 6))
    counter = 0
    while t < tf:
        hmin = 16 * np.spacing(t)
        ti = t
        yi = y_loop
        # Evaluate the time derivative(s) at 6 points within the interval
        for i in range(5):
            t_inner = ti + a[i] * h
            y_inner = yi
            for j in range(i - 1):
                y_inner = y_inner + h * b[i, j] * f[:, j]
            f[:, i] = get_x_dot(y_inner, t_inner, dt, x1_real, x2_real, Omega_real, mu1, mu2, r12_real, M1, M2)
        # compute the maximum truncation error:
        c4c5 = np.transpose(c4) - np.transpose(c5)
        te = 0.01 * np.dot(f, c4c5)
        # difference between 4th and 5th order solutions
        te_max = np.max(np.abs(te))
        # compute the allowable truncation error:
        ymax = np.max(np.abs(y_loop))
        te_allowed = tolerance * ymax
        # compute the fractional change in step size:
        delta = (te_allowed / (te_max + np.spacing(t))) ** (1. / 5.)
        # If the truncation error is in bounds, then update the solution:
        diff = te_max - te_allowed
        if -100000 < diff < 0.1:
            t = t + h
            fc5 = np.dot(f, np.transpose(c5))
            fc5.resize((4))
            y_loop = yi + h * fc5
            tout = np.append(tout, t)
            yout = np.vstack([yout, np.transpose(y_loop)])
            counter = counter + 1
        # Update time step:
        h = delta + h
        t = t + h
        if h < hmin:
            print('\n\n Warning: Step size fell below its minimum\n allowable value' + str(hmin) + ' at time ' + str(t))

    return tout, yout


#[t, f] = rkf45([t_initial, t_final], X0)
# print(f)
#x_trajectory = f[:, 0]
#y_trajectory = f[:, 1]
#vx_trajectory = f[:, 2]
#vy_trajectory = f[:, 3]

X_New, t_update, Trajectory = RK4(X0, t_final, t_initial, dt, x1_real, x2_real, Omega_real, mu1, mu2, r12_real, M1, M2)
Traj_X = Trajectory[:, 0]
Traj_Y = Trajectory[:, 1]


#df = np.sqrt((-Trajectory[-1, 1] - x2_real) ** 2 + (Trajectory[-1, 0] ** 2)) - 1737
#vf = np.sqrt((Trajectory[-1, 2]) ** 2 + Trajectory[-1, 3] ** 2)
#print(df, vf)
#print(-Trajectory[-1, 1])

ax = plt.gca()
ax.set_aspect('equal')
#ax.plot(-y_trajectory, x_trajectory)
ax.plot(Traj_X, Traj_Y)
ax.plot(x1_real, 0, '*')
ax.plot(x2_real, 0, '*')
circle1 = plt.Circle((x2_real, 0), 1737, color='r', fill=False)
circle2 = plt.Circle((x1_real, 0), 6378, color='b', fill=False)
ax.add_artist(circle1)
ax.add_artist(circle2)
plt.grid()
plt.show()
