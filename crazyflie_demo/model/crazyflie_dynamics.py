import numpy as np
import crazyflie_param as P

class CrazyflieDynamics:
    def __init__(self, init_pos=np.array([0.0, 0.0, 0.0])):
        # Initial state conditions
        self.state = np.array([
            [init_pos[0]],     # 0
            [init_pos[1]],     # 1
            [init_pos[2]],     # 2
            [P.psi0],   # 3
            [P.theta0], # 4
            [P.phi0],   # 5
            [P.u0],     # 6
            [P.v0],     # 7
            [P.w0],     # 8
            [P.r0],     # 9
            [P.q0],     # 10
            [P.p0],     # 11
        ])
        # self.state[0] = init_pos[0]; self.state[1] = init_pos[1]; self.state[2] = init_pos[2]
        # Time stept_phys
        self.Ts = P.Ts
        # Crazyflie 2.0 mass
        self.m = P.m
        # Gravitation acceleration
        self.g = P.g
        # Hover RPM
        self.omega_e = P.omega_e

        # Hover state space representation 
        # A and B matrix
        self.A = P.A
        self.B = P.B
        
        # Outer Loop Actuator limits
        self.ol_input_limits = P.ol_input_limits
        # Prop RPM Actuator Limits
        self.input_limits = P.input_limits

        # TODO inner loop rate controller 
        # IN: pc, qc, rc, p, q, r 
        # OUT: del_phi, del_theta, del_gamma

    def update(self, u):
        """
        External method that takes outer loop control commands
        and updates the state
        """
        u = self.saturate(u, self.input_limits)
        self.rk4_step(u) # propagate the state by one time sample
        # self.euler_step(u)
        y = self.h() # return the corresponding output
        return y
    
    def state_dot(self, state, u):
        # Returns the deivative of the state vector
        # Uses state-space equations provided on pg. 15
        # xdot = np.matmul(self.A, self.state) + np.matmul(self.B, u)
        xdot = np.matmul(self.A, self.state) + self.omega_e * np.matmul(self.B, u)
        return xdot
    
    def h(self):
        # Returns y = h(x) - Finds position and orientation
        # from state and combines into the output vector
        y = np.array([
            [self.state.item(0)], # x
            [self.state.item(1)], # y
            [self.state.item(2)], # z
            [self.state.item(3)], # psi
            [self.state.item(4)], # theta
            [self.state.item(5)], # pi
        ])
        return y
    
    def pwm_to_rpm(self, u_pwm):
        # Takes PWM signal sent to motors by the controller and converts to propellor RPM
        u = np.empty_like(u_pwm)
        for idx in range(u.shape[0]):
            u[idx] = 0.2685 * u_pwm[idx] + 4070.3 # Eq. 2.6.1
        return u

    def euler_step(self, u):
        # Integrate ODE using Euler's method
        xdot = self.state_dot(self.state, u)
        self.state += self.Ts * xdot

    def rk4_step(self, u):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        F1 = self.state_dot(self.state, u)
        F2 = self.state_dot(self.state + self.Ts / 2 * F1, u)
        F3 = self.state_dot(self.state + self.Ts / 2 * F2, u)
        F4 = self.state_dot(self.state + self.Ts * F3, u)
        self.state += self.Ts / 6 * (F1 + 2 * F2 + 2 * F3 + F4)

    def saturate(self, u, limit):
        for idx in range(u.shape[0]):
            if abs(u[idx,0]) > limit[idx,0]:
                u[idx,0] = limit[idx,0]*np.sign(u[idx,0])
        return u

# Run some tests to see functionality
if __name__ == "__main__":
    cf1 = CrazyflieDynamics()

    # Symmetric input
    u = np.array([
        [25000],
        [10000],
        [10000],
        [10000],
    ])

    # # Test euler integrator with symmetric input [RPM] 
    # print("Euler's method\n", cf1.state)
    # for _ in range(3):
    #     cf1.euler_step(u)
    #     print(cf1.state)

    # # Test rk4 integrator with symmetric input [RPM]
    # cf2 = CrazyflieDynamics()
    # print("RK4\n", cf2.state)
    # for _ in range(3):
    #     cf2.rk4_step(u)
    #     print(cf2.state)

    # Test saturate
    for _ in range(3):
        print(u)
        y = cf1.update(u)
        print(u)