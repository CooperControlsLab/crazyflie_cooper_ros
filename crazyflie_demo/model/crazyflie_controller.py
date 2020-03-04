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
    def __init__(self, kp_p=70.0, kp_q=70.0, kp_r=70.0, ki_r=16.7):
        self.kp_p = kp_p    # Roll Rate Proportional Gain
        
        self.kp_q = kp_q    # Pitch Rate Proportional Gain
        
        self.kp_r = kp_r    # psi Rate Proportional Gain
        self.ki_r = ki_r    # psi Rate Integral Gain
        self.e_r_hist = 0.0 # Initialize Historical Error

    def update(self, p_c, q_c, r_c, state):
        del_phi = self.kp_p * (p_c - state.item(11))
        
        del_theta = self.kp_q * (q_c - state.item(10))

        e_r = r_c - state.item(9)
        self.e_r_hist += e_r
        del_psi = self.kp_r * (r_c - state.item(9)) + (self.ki_r * self.e_r_hist)

        return del_phi, del_theta, del_psi
        # used in control mixer

class AttitudeController:
    # TODO: integrator makes unstable
    def __init__(self, kp=10.0, ki=0.0, kd=0.0, cap=100.0):
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

        self.t_phys = 1/30.0

        self.cap = cap

    def update(self, phi_c, theta_c, state): # phi controls neg y, theta controls pos x
        # Calculate errors
        e_phi = phi_c - state.item(5)
        self.e_phi_hist += (e_phi * self.t_phys)
        e_phi_der = (e_phi - self.e_phi_prev) / self.t_phys
        self.e_phi_prev = e_phi

        p_c = (self.kp_phi * e_phi) + (self.ki_phi * self.e_phi_hist) +\
            (self.kd_phi * e_phi_der)
        
        e_theta = theta_c - state.item(4)
        self.e_theta_hist += (e_theta * self.t_phys)
        q_c = (self.kp_theta * e_theta) + (self.ki_theta * self.e_theta_hist)

        if np.abs(q_c) > self.cap:
            q_c = self.cap * (np.sign(q_c))
        if np.abs(p_c) > self.cap:
            p_c = self.cap * (np.sign(p_c))
   
        return p_c, q_c
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
    def __init__(self, kp=20000.0, ki=35.0, kd=11000.0):
        self.omega_cap_e = 44705 # Feedforward from Eq. 3.1.8
        self.kp = kp
        self.ki = ki
        self.e_hist = 0.0
        self.kd = kd
        self.e_prev = 0.0
        self.t_ob = P.t_ob

    def update(self, z_c, z):
        e = z_c - z
        # print("commanded pos is {}, actual pos is {}".format(z_c, z))
        # self.e_hist += (e * self.t_ob) # historical error
        self.e_hist += e
        e_der = (e - self.e_prev) / self.t_ob # dirty derivative error
        self.e_prev = e
        # print("error: {}, der error: {}, hist error {}".format((e * self.kp), (e_der * self.kd), (self.e_hist * self.ki)))
        del_omega_cap = (self.kp * e) + (self.ki * self.e_hist) + (self.kd * e_der)
        # del_omega_cap = self.saturate(del_omega_cap)
        return del_omega_cap
    
    def saturate(self, del_omega_cap):
        # using 10000 - 60000 PWM as per crazyflie_ros linear.z msg
        if del_omega_cap > 60000:
            del_omega_cap = 60000
        elif del_omega_cap < 10000:
            del_omega_cap = 10000
        return del_omega_cap

class XYController:
    def __init__(self, kp=20.0, ki=2.0, cap=0.2):
        self.kp = kp
        self.ki = ki
        self.cap = cap
        self.x_b_prev = 0.0
        self.y_b_prev = 0.0
        self.xe_b_hist = 0.0
        self.ye_b_hist = 0.0
    
    def update(self, x_c, x, y_c, y, psi, t):
        xe = x_c - x; ye = y_c - y # Get position error
        # print('xe {}\nye {}'.format(xe, ye))

        x_b = x * np.cos(psi) + y * np.sin(psi) # Get x in body frame
        u = (x_b - self.x_b_prev) / t # u is x-vel in body frame
        self.x_b_prev = x_b # Reset previous val

        y_b = -(x * np.sin(psi)) + y * np.cos(psi) # Get y in body frame
        v = (y_b - self.y_b_prev) / t # v is y-vel in body frame
        self.y_b_prev = y_b # Reset previous val

        xe_b = xe * np.cos(psi) + ye * np.sin(psi) # Get errors in body frame
        ye_b = -(xe * np.sin(psi)) + ye * np.cos(psi)

        self.xe_b_hist += ((xe_b - u) * t) # Accumulate and store histroical error
        self.ye_b_hist += ((ye_b - v) * t)

        phi_c   = ((xe_b - u) * ( self.kp)) + (self.xe_b_hist * ( self.ki)) # Eq. 3.1.11 and Eq. 3.1.12
        theta_c = ((ye_b - v) * (-self.kp)) + (self.ye_b_hist * (-self.ki))

        # Cap roll (y) and pitch (x) to prevent unstable maneuvers
        if np.abs(phi_c) >= self.cap:
            phi_c =  np.sign(phi_c) * self.cap

        if np.abs(theta_c) >= self.cap:
            theta_c = np.sign(theta_c) * self.cap
        
        return theta_c, phi_c

class YawController:
    def __init__(self, kp=-20.0):
        self.kp = kp
    
    def update(self, psi_c, psi):
        """
        Off-Board global psi angle controller

        Parameters
        ----------
        psi_c = psi angle setpoint
        psi   = current psi angle

        Returns
        -------
        psie_tot = total psi angle error which maps to commanded psi rate
        """
        psi_e = psi_c - psi
        psi_e_tot = self.kp * psi_e
        return psi_e_tot