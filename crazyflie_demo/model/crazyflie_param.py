import numpy as np

g = 9.80665 # [m/s^2]

# Physical Parameters of the Crazyflie 2.0
m = 0.035         # Crazyflie 2.0 mass w/ 4 Vicon markers [g]
Ct = 3.1582e-10   # Thrust coeff [N/rpm^2]
Cd = 7.9379e-12   # Drag coeff [Nm/rpm^2]
Ixx = 1.395e-5    # Moment of inertia [Kg x m^2]
Iyy = 1.436e-5    # Moment of inertia [Kg x m^2]
Izz = 2.173e-5    # Moment of inertia [Kg x m^2]
d = 39e-3         # Crazyflie 2.0 arm length [m]
# omega_e = 16073.0 # Hover RPM [RPM]
omega_e = 16486.0

# Initial Conditions, state vector formed following:
# https://arxiv.org/pdf/1608.05786.pdf
x0 = 0.0     # X position [m]
y0 = 0.0     # Y position [m]
z0 = 0.0     # Z position [m]
psi0 = 0.0   # Yaw angle [deg]
theta0 = 0.0 # Pitch angle [deg]
phi0 = 0.0   # Roll angle [deg]
u0 = 0.0     # Body frame X linear velocity [m/s]
v0 = 0.0     # Body frame Y linear velocity [m/s]
w0 = 0.0     # Body frame Z linear velocity [m/s]
r0 = 0.0     # Body frame yaw angular velocity [deg/s]
q0 = 0.0     # Body frame pitch angular velocity [deg/s]
p0 = 0.0     # Body frame roll angular velocity [deg/s]

# Simulation Parameters
t_start = 0.0 # [s]
t_end = 10.0  # [s]
Ts = 0.01     # [s]

freq_rate_ctrl = 500.0
t_rate = 1/freq_rate_ctrl 

freq_attitude_ctrl = 250.0
t_att = 1/freq_attitude_ctrl

freq_off_board = 100.0
t_ob = 1/freq_off_board

freq_phys = 30.0
t_phys = 1/freq_phys

freq_plot = 10.0
t_plot = 1/freq_plot

# Prop RPM saturation limits
rpm_max = 21666.4
input_limits = np.full((4, 1), rpm_max)

# Outer loop saturation Limits
phi_max = 30.0       # [rad]
theta_max = 30.0     # [rad]
r_max = 200.0        # [rad/s]
thrust_max = 60000.0 # mapped to PWM output [N/A]
ol_input_limits = np.array([
    [phi_max],
    [theta_max],
    [r_max],
    [thrust_max],
])

# Construction of the necessary state-space matrices
N = 12
M = 4
A = np.zeros((N, N))

A = np.zeros((N, N))
A[0, 6] = 1.
A[1, 7] = 1.
A[2, 8] = 1.
A[3, 9] = 1.
A[4, 10] = 1.
A[5, 11] = 1.
A[6, 4] = g
A[7, 5] = -g

B = np.zeros((N, M))
B[8, 0] = 2. * (Ct/m)
B[8, 1] = 2. * (Ct/m)
B[8, 2] = 2. * (Ct/m)
B[8, 3] = 2. * (Ct/m)
B[9, 0] = -2. * (Cd/Izz)
B[9, 1] = 2. * (Cd/Izz)
B[9, 2] = -2. * (Cd/Izz)
B[9, 3] = 2. * (Cd/Izz)
B[10, 0] = -np.sqrt(2) * d * (Ct/Iyy)
B[10, 1] = np.sqrt(2) * d * (Ct/Iyy)
B[10, 2] = np.sqrt(2) * d * (Ct/Iyy)
B[10, 3] = -np.sqrt(2) * d * (Ct/Iyy)
B[11, 0] = -np.sqrt(2) * d * (Ct/Ixx)
B[11, 1] = -np.sqrt(2) * d * (Ct/Ixx)
B[11, 2] = np.sqrt(2) * d * (Ct/Ixx)
B[11, 3] = np.sqrt(2) * d * (Ct/Ixx)

# C = np.eye(12)
