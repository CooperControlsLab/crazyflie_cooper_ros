import numpy as np
import crazyflie_param as P
import random

class CrazyflieDynamics:
    def __init__(self, init_pos=np.array([0.0, 0.0, 0.0])):
        """
        Class that contains functions for updating the state of the Crazyflie
        following the equations of motion

        Parameters:
        -----------
        init_pos = optional - initial starting position of drone in inertial frame
        """
        # Initial state condition
        self.state = np.array([
            [init_pos[0]],     # 0
            [init_pos[1]],     # 1
            [init_pos[2]],     # 2
            [P.psi0],          # 3
            [P.theta0],        # 4
            [P.phi0],          # 5
            [P.u0],            # 6
            [P.v0],            # 7
            [P.w0],            # 8
            [P.r0],            # 9
            [P.q0],            # 10
            [P.p0],            # 11
        ])
        
        self.Ts = P.Ts # Time stept_phys
        self.g = P.g # Gravitational acceleration
        self.omega_e = P.omega_e # Hover RPM

        # Parameters of a system are never fully known
        # Add some uncertainty such that model is more robust
        alpha = 0.2  # Uncertainty parameter
        self.m = P.m * (1.+alpha*(2.*np.random.rand()-1.)) # Crazyflie 2.0 mass
        self.A = P.A * (1.+alpha*(2.*np.random.rand()-1.)) # State-space representation A matrix
        self.B = P.B * (1.+alpha*(2.*np.random.rand()-1.)) # State-space representation B matrix
        
        self.input_limits = P.input_limits # PWM Actuation Limits

    def update(self, u):
        """
        External method that takes outer loop control commands
        and updates the state
        """
        u = self.saturate(u, self.input_limits) # saturate the inputs based on what is physically possible
        self._rk4_step(u) # propagate the state by one time sample
        mu = 0.0; sigma = 0.00007 # random noise parameters
        self.state += random.gauss(mu, sigma) # simulate sensor noise
        y = self._h() # return the corresponding output
        return y
    
    def _state_dot(self, state, u):
        """
        Uses state-space equations to calculate 
        linearized time-derivative of state vector

        Parameters:
        -----------
        state = 12-variable state vector of cf
        u     = 4-varibale control input vector of cf

        Returns:
        --------
        xdot  = time-derivative of the state vector
        """
        xdot = np.matmul(self.A, self.state) + self.omega_e * np.matmul(self.B, u)
        return xdot
    
    # def _state_dot_nonlinear(self, state, u):
    #     """
    #     Uses state-space equations provided on pg. 15 to calculate 
    #     nonlinear time-derivative of state vector

    #     Parameters:
    #     -----------
    #     state = 12-variable state vector of cf
    #     u     = 4-varibale control input vector of cf

    #     Returns:
    #     --------
    #     xdot  = time-derivative of the state vector
    #     """
    #     xdot = np.matmul(self.A, self.state) + self.omega_e * np.matmul(self.B, u)
    #     return xdot
    
    def _h(self):
        """
        Finds position and orientation from state and combines into the output vector
        
        Returns:
        --------
        y = 6-variable output vector
        """
        y = np.array([
            [self.state.item(0)], # x
            [self.state.item(1)], # y
            [self.state.item(2)], # z
            [self.state.item(3)], # psi
            [self.state.item(4)], # theta
            [self.state.item(5)], # pi
        ])
        return y

    def _euler_step(self, u):
        # Integrate ODE using Euler's method
        xdot = self._state_dot(self.state, u)
        self.state += self.Ts * xdot

    def _rk4_step(self, u):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        F1 = self._state_dot(self.state, u)
        F2 = self._state_dot(self.state + self.Ts / 2 * F1, u)
        F3 = self._state_dot(self.state + self.Ts / 2 * F2, u)
        F4 = self._state_dot(self.state + self.Ts * F3, u)
        self.state += self.Ts / 6 * (F1 + 2 * F2 + 2 * F3 + F4)

    def saturate(self, u, limit):
        # saturate control input vector to provided caps
        for idx in range(u.shape[0]):
            if abs(u[idx,0]) > limit[idx,0]:
                u[idx,0] = limit[idx,0]*np.sign(u[idx,0])
        return u

    def pwm_to_rpm(self, u_pwm):
        """
        Takes PWM signal sent to motors by the controller 
        and converts to propellor RPM
        """
        u = np.empty_like(u_pwm)
        for idx in range(u.shape[0]):
            u[idx] = (0.2685 * u_pwm[idx] + 4070.3) - self.omega_e # Eq. 2.6.1 subtracted from equillibrium point
        return u