import numpy as np
import matplotlib.pyplot as plt 

from crazyflie_dynamics import CrazyflieDynamics
from data_plotter import DataPlotter
import crazyflie_param as P

# self.state = np.array([
#     [P.x0],     # 0
#     [P.y0],     # 1
#     [P.z0],     # 2
#     [P.psi0],   # 3
#     [P.theta0], # 4
#     [P.phi0],   # 5
#     [P.u0],     # 6
#     [P.v0],     # 7
#     [P.w0],     # 8
#     [P.r0],     # 9
#     [P.q0],     # 10
#     [P.p0],     # 11
# ])

class RateController:
    def __init__(self, t=P.t_rate, kp_p=1.0, kp_q=1.0, kp_r=70.0, ki_r=16.7):
        self.kp_q = kp_q    # Pitch Rate Proportional Gain
        
        self.kp_p = kp_p    # Roll Rate Proportional Gain
        
        self.kp_r = kp_r    # psi Rate Proportional Gain
        self.ki_r = ki_r    # psi Rate Integral Gain
        self.e_r_hist = 0.0 # Initialize Historical Error

        self.t = t

    def update(self, q_c, p_c, r_c, state):
        q = 57.2958*state.item(10); p = 57.2958*state.item(11)

        del_theta = self.kp_q * (q_c - q)

        del_phi = self.kp_p * (p_c - p)

        e_r = r_c - state.item(9)
        self.e_r_hist += (e_r * self.t)
        del_psi = self.kp_r * (r_c - state.item(9)) + (self.ki_r * self.e_r_hist)

        return del_phi, del_theta, del_psi
        # used in control mixer

class AttitudeController:
    # TODO: integrator makes unstable
    def __init__(self, t=P.t_att, kp=3.5, ki=2.0, kd=0.0, cap=0.2):
        self.kp_phi = kp      # Roll Attitude Proportional Gain
        self.ki_phi = ki      # Roll Attitude Integral Gain
        self.kd_phi = kd
        self.e_phi_hist = 0.0
        self.e_phi_prev = 0.0
        
        self.kp_theta = kp    # Pitch Attitude Proportional Gain
        self.ki_theta = ki    # Pitch Attitude Integral Gain
        self.kd_theta = kd
        self.e_theta_hist = 0.0
        self.e_theta_prev = 0.0

        self.t = t

        self.cap = cap

    def update(self, theta_c, phi_c, state): # phi controls neg y, theta controls pos x
        # Values from the state are in rad - convert to deg for control purposes
        theta = 57.2958*state.item(4); phi = 57.2958*state.item(5)

        # Calculate errors
        e_theta = theta_c - theta
        self.e_theta_hist += (e_theta * self.t)
        e_theta_der = (e_theta - self.e_theta_prev) / self.t
        self.e_theta_prev = e_theta

        q_c = (self.kp_theta * e_theta) + (self.ki_theta * self.e_theta_hist) \
            + (self.kd_theta * e_theta_der)

        print('theta {} theta_c {} e_theta {} q_c {}'.format(theta, theta_c, e_theta, q_c))

        e_phi = phi_c - phi
        self.e_phi_hist += (e_phi * self.t)
        e_phi_der = (e_phi - self.e_phi_prev) / self.t
        self.e_phi_prev = e_phi

        p_c = (self.kp_phi * e_phi) + (self.ki_phi * self.e_phi_hist) +\
            (self.kd_phi * e_phi_der)
        
        # if np.abs(q_c) > self.cap:
        #     q_c = self.cap * (np.sign(q_c))
        # if np.abs(p_c) > self.cap:
        #     p_c = self.cap * (np.sign(p_c))
   
        return q_c, p_c
        # used in the rate controller

class ControlMixer:
    def __init__(self):
        self.temp = 0.0

    # pos theta, pos x
    # pos phi, neg y
    def update(self, omega_cap, del_phi, del_theta, del_psi):
        u_pwm = np.array([
            [omega_cap - del_phi/2 - del_theta/2 - del_psi],
            [omega_cap - del_phi/2 + del_theta/2 + del_psi],
            [omega_cap + del_phi/2 + del_theta/2 - del_psi],
            [omega_cap + del_phi/2 - del_theta/2 + del_psi], 
        ])
        return u_pwm

class AltitudeController:
    def __init__(self, t=P.t_ob, ff=46241.0, kp=11000.0, ki=1000.0, kd=2000.0):
        # 44705
        self.ff = ff # Feedforward from Eq. 3.1.8 not used currently
        self.kp = kp
        self.ki = ki
        self.e_hist = 0.0
        self.kd = kd
        self.e_prev = 0.0

        self.t = t

    def update(self, z_c, z):
        e = z_c - z
        # print("commanded pos is {}, actual pos is {}".format(z_c, z))
        self.e_hist += (e * self.t)
        e_der = (e - self.e_prev) / self.t # dirty derivative error
        self.e_prev = e
        # print("error: {}, der error: {}, hist error {}".format((e * self.kp), (e_der * self.kd), (self.e_hist * self.ki)))
        del_omega_cap = self.ff + (self.kp * e) + (self.ki * self.e_hist) + (self.kd * e_der)
        # del_omega_cap = (self.kp * e) + (self.ki * self.e_hist) + (self.kd * e_der)
        del_omega_cap = self.saturate(del_omega_cap)
        return del_omega_cap
    
    def saturate(self, del_omega_cap):
        # using 10000 - 60000 PWM as per crazyflie_ros linear.z msg
        if del_omega_cap > 15000 + self.ff:
            del_omega_cap = 15000 + self.ff
        elif del_omega_cap < -20000 + self.ff:
            del_omega_cap = -20000 + self.ff
        return del_omega_cap

class XYController:
    def __init__(self, t=P.t_ob, kp=30.0, ki=2.0, cap=0.524):
        self.kp = kp
        self.ki = ki
        self.cap = cap
        self.x_b_prev = 0.0
        self.y_b_prev = 0.0
        self.xe_b_hist = 0.0
        self.ye_b_hist = 0.0
        self.t = t
    
    def update(self, x_c, y_c, state):
        x = state.item(0); y = state.item(1); psi = state.item(3)
        u = state.item(6); v = state.item(7)

        xe = x_c - x; ye = y_c - y # Get position error

        # print('xe {}\nye {}'.format(xe, ye))
        # print('x_c {} x {}'.format(x_c, x))

        # x_b = x * np.cos(psi) + y * np.sin(psi) # Get x in body frame
        # u = (x_b - self.x_b_prev) / self.t # u is x-vel in body frame
        # self.x_b_prev = x_b # Reset previous val

        # y_b = -(x * np.sin(psi)) + y * np.cos(psi) # Get y in body frame
        # v = (y_b - self.y_b_prev) / self.t # v is y-vel in body frame
        # self.y_b_prev = y_b # Reset previous val

        xe_b = xe * np.cos(psi) + ye * np.sin(psi) # Get errors in body frame
        ye_b = -(xe * np.sin(psi)) + ye * np.cos(psi)

        self.xe_b_hist += ((xe_b - u) * self.t) # Accumulate and store histroical error
        self.ye_b_hist += ((ye_b - v) * self.t)

        # print('xe {} u {} ye {} u {}'.format(xe, u, ye, v))

        # Control law - angles are in radians
        theta_c = ((xe_b - u) * ( self.kp)) + (self.xe_b_hist * ( self.ki)) # Eq. 3.1.11 and Eq. 3.1.12
        phi_c   = ((ye_b - v) * (-self.kp)) + (self.ye_b_hist * (-self.ki))

        # 0.00116355*
        # print('theta_c {} phi_c {}'.format(theta_c, phi_c))

        # Cap roll (y) and pitch (x) to prevent unstable maneuvers
        if np.abs(phi_c) >= self.cap:
            phi_c =  np.sign(phi_c) * self.cap

        if np.abs(theta_c) >= self.cap:
            theta_c = np.sign(theta_c) * self.cap
        
        return theta_c, phi_c

class YawController:
    def __init__(self, kp=0.1, cap=200.0):
        self.kp = kp
        self.cap = cap
    
    def update(self, psi_c, psi):
        """
        Off-Board global psi angle controller

        Parameters
        ----------
        psi_c = psi angle setpoint [deg]
        psi   = current psi angle [deg]

        Returns
        -------
        r_c = commanded yaw rate [deg/s]
        """
        psi = 57.2958*psi # convert from [rad] to [deg]
        psi_e = psi_c - psi
        r_c = self.kp * psi_e

        if np.abs(r_c) >= self.cap:
            r_c = np.sign(r_c) * self.cap

        return r_c

class XYTrajController:
    def __init__(self, kp=100.0, kd=100.0, k_ff=10.0, cap=30.0):
        self.kp = kp
        self.kd = kd # adding damping
        self.k_ff = k_ff
        self.cap = cap

        self.r_prev = np.array([0.0, 0.0])

        self.t = P.t_phys
        self.g = 9.80665

        self.time_now = 0.0
        self.time_list = []
        self.x_list = []
        self.x_t_list = []
        self.y_list = []
        self.y_t_list = []
        self.xd_list = []
        self.yd_list = []
        self.xd_t_list = []
        self.yd_t_list = []
    
    def normalize(self, a):
        """
        Return the normal unit vector.
        param a: vector ([float])
        """
        normal = np.empty_like(a)
        normal[0] = -a[1]
        normal[1] = a[0]
        normal = normal / np.linalg.norm(normal)
        return normal

    def update(self, r_t, rd_t, r_t_vect, state, psi_c, rdd_t, is_pickling=False):
        """
        Off-Board trajectory PID controller
        Parameters
        ----------
        r_t      = traj pos
        rd_t     = traj vel
        r_t_vect = vector from current to next traj point
        r        = actual drone pos
        psi_c    = yaw setpoint
        rdd_t    = pre-computed feedforward
        is_pickling = save out data into 'cf_data' file
        Returns
        -------
        theta_c = commanded pitch angle which results in pos x movement
        phi_c   = commanded roll angle which results in neg y movement
        """
        # # Calculate unit vectors used tabulate components of error
        # t_unit = r_t_vect / np.linalg.norm(r_t_vect)
        # n_unit = self.normalize(t_unit)

        # Calculate all position error

        x = state.item(0); y = state.item(1)
        r = np.array([x, y])

        e_p = r_t - r

        # # Calculate velocity error component
        # rd = (r - self.r_prev) / self.t
        # self.r_prev = r

        # obtain u and v from state
        u = state.item(6); v = state.item(7)
        rd = np.array([u, v])

        e_v = rd_t - rd
        
        rdd_t = self.kp * e_p + self.kd * e_v \
            # + self.k_ff * rdd_t # optional feedforward component
        # print("total rdd_t {}".format(rdd_t))

        theta_c = 1.0/self.g * (rdd_t[0] * np.sin(psi_c) - \
            rdd_t[1] * np.cos(psi_c)) # equivalent to movement in -y direction
        phi_c   = 1.0/self.g * (rdd_t[0] * np.cos(psi_c) + \
            rdd_t[1] * np.sin(psi_c)) # equivalent to movement in +x direction

        # Cap roll (y) and pitch (x) to prevent unstable maneuvers
        if np.abs(phi_c) >= self.cap:
            phi_c =  np.sign(phi_c) * self.cap
        if np.abs(theta_c) >= self.cap:
            theta_c = np.sign(theta_c) * self.cap

        return phi_c, theta_c