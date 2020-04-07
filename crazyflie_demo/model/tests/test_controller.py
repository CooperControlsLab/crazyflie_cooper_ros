import numpy as np
import matplotlib.pyplot as plt 

# Import crazyflie model modules
import sys
sys.path.append("../")
from data_plotter import DataPlotter
from crazyflie_dynamics import CrazyflieDynamics
from crazyflie_controller import RateController, AttitudeController, ControlMixer, AltitudeController, XYController, YawController
import crazyflie_param as P
from crazyflie_animation import CrazyflieAnimation

def test_altitude(z_c, show_anim=True, save_plot=False):
    cf = CrazyflieDynamics()
    
    if show_anim:
        plot = DataPlotter()
        anim = CrazyflieAnimation()

    rate_ctrl = RateController()
    attitude_ctrl = AttitudeController(P.t_att)
    ctrl_mixer = ControlMixer()
    altitiude_ctrl = AltitudeController()

    # off-borad controller input values
    u_ob = np.array([
        [0.0], # roll
        [0.0], # pitch
        [0.0], # yaw rate
        [0.0], # thrust
    ])

    # reference values
    r = np.array([
        [0.0], # x
        [0.0], # y
        [z_c], # z
        [0.0], # psi
    ])

    r_c = 0.0
    phi_c = 0.0; theta_c = 0.0
    del_phi = 0.0; del_theta = 0.0; del_psi = 0.0

    t = P.t_start

    if save_plot:
        fig, ax = plt.subplots()
        ax.set_title("CF Altitude Simulation")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("z [m]")
        x_list = []
        y_list = []
        ref_list = []

    while t < P.t_end: # plotter can run slowly
        t_next_plot = t + P.t_plot
        
        while t < t_next_plot: # offboard controller is slowest at 100 hz
            t_next_ob = t + P.t_ob
            u_ob[3,0] = altitiude_ctrl.update(r.item(2), cf.state.item(2))
            print("thrust:, \n", u_ob.item(3))

            u = ctrl_mixer.update(u_ob.item(3), del_phi, del_theta, del_psi) # output is PWM signal
            # print("u_pwm: \n", u_pwm.item(0))
            
            y = cf.update(u) # rpm is used in cf state update equations
            t = t + P.t_ob

            if save_plot:
                x_list.append(t)
                y_list.append(cf.state.item(2))
                ref_list.append(r.item(2))
        
        if show_anim:
            plot.update(t, r, cf.state, u)
            anim.update(cf.state)
            plt.pause(0.0000001)

    if save_plot:
        ax.plot(x_list, y_list, c='r')
        ax.plot(x_list, ref_list, c='b')
        fig.savefig("hover_alt_1m_sim")

    if show_anim:
        print('Press key to close')
        plt.waitforbuttonpress()
        plt.close()

def test_ctrl_mixer():
    cf = CrazyflieDynamics()
    plot = DataPlotter()

    ctrl_mixer = ControlMixer()

    t = P.t_start

    thrust = 0.0
    del_phi = 0.0 # move in neg y
    del_theta = 0.1 # move in pos x
    del_psi = 0.0 # move in pos yaw

    # reference values
    r = np.array([
        [0.0], # x
        [0.0], # y
        [0.0], # z
        [0.0], # psi
        [0.0], # theta
        [0.0], # phi
    ])

    while t < P.t_end: # plotter can run the slowest
        t_next_plot = t + P.t_plot

        while t < t_next_plot: # rate controller is the fastest running at 500 hz
            u = ctrl_mixer.update(thrust, del_phi, del_theta, del_psi)
            y = cf.update(u)
            t = t + P.t_rate
    
        plot.update(t, r, cf.state, u)
        plt.pause(0.5)
    
    print('Press key to close')
    plt.waitforbuttonpress()
    plt.close()

def test_rate_ctrl():
    cf = CrazyflieDynamics()
    plot = DataPlotter()

    ctrl_mixer = ControlMixer()
    rate_ctrl = RateController()


    t = P.t_start

    # pos roll rate should make pos phi and move in neg y
    p_c = 0.0 # roll rate moves in neg theta and neg x

    # pos pitch rate should make pos theta and move in pos x
    q_c = 0.0 # pitch rate move in pos phi and pos x
    
    thrust = 0.0

    # off-borad controller input values
    u_ob = np.array([
        [0.0], # roll
        [0.0], # pitch
        [0.0], # yaw rate, moves in move psi
        [0.0], # thrust
    ])

    # reference values
    r = np.array([
        [0.0], # x
        [0.0], # y
        [0.0], # z
        [0.0], # psi
        # [0.0], # theta
        # [0.0], # phi
    ])
    
    while t < P.t_end: # plotter can run the slowest
        t_next_plot = t + P.t_plot

        while t < t_next_plot: # rate controller is the fastest running at 500 hz
            del_phi, del_theta, del_psi = rate_ctrl.update(p_c, q_c, u_ob.item(2), cf.state)
            print("del_phi {}\ndel_theta {}\ndel_psi {}".format(del_phi, del_theta, del_psi))
            
            u = ctrl_mixer.update(thrust, del_phi, del_theta, del_psi)
            y = cf.update(u)
            t = t + P.t_rate
    
        plot.update(t, r, cf.state, u)
        plt.pause(0.5)
    
    print('Press key to close')
    plt.waitforbuttonpress()
    plt.close()

def test_attitude_ctrl(phi_c, theta_c):
    cf = CrazyflieDynamics()
    plot = DataPlotter()

    ctrl_mixer = ControlMixer()
    rate_ctrl = RateController()
    attitude_ctrl = AttitudeController(P.t_att)

    t = P.t_start
    
    # off-borad controller input values
    u_ob = np.array([
        [phi_c], # phi, roll angle, neg y
        [theta_c], # theta, pitch angle, pos x
        [0.0], # yaw rate, moves in move psi
        [0.0], # thrust
    ])

    # reference values
    r = np.array([
        [0.0], # x
        [0.0], # y
        [0.0], # z
        [0.0], # psi
        # [0.0], # theta
        # [0.0], # phi
    ])
    
    while t < P.t_end: # plotter can run the slowest
        t_next_plot = t + P.t_plot

        while t < t_next_plot: # attitude controller runs at 250 hz
            t_next_att = t + P.t_att
            # phi controls x, theta controls y
            p_c, q_c = attitude_ctrl.update(u_ob.item(0), u_ob.item(1), cf.state)
            print('p_c (roll rate) {}\nq_c (pitch rate) {}'.format(p_c, q_c))

            while t < t_next_att: # rate controller is the fastest running at 500 hz
                del_phi, del_theta, del_psi = rate_ctrl.update(p_c, q_c, u_ob.item(2), cf.state)
                print("del_phi {}\ndel_theta {}\ndel_psi {}".format(del_phi, del_theta, del_psi))
                
                u = ctrl_mixer.update(u_ob.item(3), del_phi, del_theta, del_psi)
                y = cf.update(u)
                t = t + P.t_rate
        
        plot.update(t, r, cf.state, u)
        plt.pause(0.5)
    
    print('Press key to close')
    plt.waitforbuttonpress()
    plt.close()

def test_xy(x_c, y_c, z_c, psi_c, save_plot=False):
    cf = CrazyflieDynamics()
    plot = DataPlotter()
    traj = np.array([x_c, y_c, z_c])
    anim = CrazyflieAnimation(traj)

    # Create class objects
    rate_ctrl = RateController()
    attitude_ctrl = AttitudeController(t=P.t_att, kp=3.5, ki=2.0, kd=1.0)
    rate_ctrl = RateController()
    ctrl_mixer = ControlMixer()
    altitiude_ctrl = AltitudeController(ff=41000.0)
    xy_ctrl = XYController(t=P.t_ob, kp=30.0, ki=2.0, cap=20)
    yaw_ctrl = YawController()

    # off-borad controller input values
    u_ob = np.array([
        [0.0], # pitch (phi -> x)  - 0
        [0.0], # roll (theta -> y) - 1
        [0.0], # yaw rate          - 2
        [0.0], # thrust            - 3
    ])

    # reference values
    r = np.array([
        [x_c], # x                 - 0
        [y_c], # y                 - 1
        [z_c], # z                 - 2
        [psi_c], # psi             - 3
    ]) 

    t = P.t_start

    while t < P.t_end: # plotter can run the slowest
        t_next_plot = t + P.t_plot
        
        while t < t_next_plot: # offboard controller is slowest at 30 hz
            t_next_ob = t + P.t_ob
            
            # Altitude off-board controller update
            u_ob[3,0] = altitiude_ctrl.update(r.item(2), cf.state.item(2))

            # XY off-borad controller update
            # phi_c  , theta_c                    x_c      , x
            u_ob[0,0], u_ob[1,0] = xy_ctrl.update(r.item(0), cf.state.item(0), \
                # y_c    , y
                r.item(1), cf.state.item(1), \
                0.0,
                # cf.state.item(3)
                )

            while t < t_next_ob: # attitude controller runs at 250 hz
                t_next_att = t + P.t_att

                # Conduct attitude control
                # phi controls x, theta controls y
                p_c, q_c = attitude_ctrl.update(u_ob.item(0), u_ob.item(1), cf.state)

                while t < t_next_att: # rate controller is the fastest running at 500 hz
                    t = t + P.t_rate

                    # Conduct rate control
                    del_phi, del_theta, del_psi = rate_ctrl.update(p_c, q_c, u_ob.item(2), cf.state)
                    
                    # Update state of model
                    u = ctrl_mixer.update(u_ob.item(3), del_phi, del_theta, del_psi)
                    y = cf.update(u)

        plot.update(t, r, cf.state, u)
        anim.update(cf.state)
        plt.pause(0.01)
    
    print('Press key to close')
    plt.waitforbuttonpress()
    plt.close()

def test_all(x_c, y_c, z_c, psi_c, show_anim=True, save_plot=False):
    cf = CrazyflieDynamics(init_pos=np.array([0.0, 0.0, 0.0]))
    # cf = CrazyflieDynamics()

    if show_anim:
        plot = DataPlotter()
        traj = np.array([x_c, y_c, z_c])
        anim = CrazyflieAnimation(traj)

    # Create class objects
    rate_ctrl = RateController(kp_q=1.0)
    attitude_ctrl = AttitudeController(kp=0.1)
    ctrl_mixer = ControlMixer()
    # altitiude_ctrl = AltitudeController(t=P.t_phys, ff=41000.0, \
        # kp=11000.0, ki=0.0, kd=1000.0) # Phys values
    altitiude_ctrl = AltitudeController()
    xy_ctrl = XYController(kp=1.0, ki=0.0, cap=10.0)
    # xy_ctrl = XYController(kp=1.0, ki=0.067, cap=30.0)
    yaw_ctrl = YawController(kp=0.1, cap=200.0)

    # off-borad controller input values
    u_ob = np.array([
        [0.0], # pitch (theta -> x) - 0 [deg]
        [0.0], # roll  (phi   -> y) - 1 [deg]
        [0.0], # thrust             - 2 [deg]
        [0.0], # yaw rate           - 3 [deg]
    ])

    # reference values
    r = np.array([
        [x_c], # x                 - 0 [m]
        [y_c], # y                 - 1 [m]
        [z_c], # z                 - 2 [m]
        [psi_c], # psi             - 3 [deg]
    ]) 

    t = P.t_start

    if save_plot:
        fig, ax = plt.subplots()
        ax.set_title("CF Yaw Simulation")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("psi [rad]")
        ax.grid()
        x_list = []
        y_list = []
        ref_list = []

    while t < P.t_end: # plotter can run the slowest
        t_next_plot = t + P.t_plot
        
        while t < t_next_plot: # offboard controller is slowest at 100 hz
            t_next_ob = t + P.t_ob
            # t_next_phys = t + P.t_phys
            
            # Altitude off-board controller update
            u_ob[2,0] = altitiude_ctrl.update(r.item(2), cf.state.item(2))

            # XY off-borad controller update
            # theta_c, phi_c                      x_c      , y_c
            u_ob[0,0], u_ob[1,0] = xy_ctrl.update(r.item(0), r.item(1), cf.state)

            # print("theta_c {} phi_c {}".format(u_ob[0,0], u_ob[1,0]))

            # Yaw rate off-board controller update
            u_ob[3,0] = yaw_ctrl.update(r.item(3), cf.state.item(3))

            # For plotter
            ctrl = np.array([
                [u_ob[0,0]], # [deg]
                [u_ob[1,0]], # [deg]
                [u_ob[2,0]], # [PWM]
                [u_ob[3,0]], # [deg/s]
            ])

            while t < t_next_ob: # attitude controller runs at 250 hz
            # while t < t_next_phys:
                t_next_att = t + P.t_att

                # Conduct attitude control
                # phi controls x, theta controls y
                q_c, p_c = attitude_ctrl.update(u_ob.item(0), u_ob.item(1), cf.state)

                # print("p_c {} q_c {}".format(p_c, q_c))

                while t < t_next_att: # rate controller is the fastest running at 500 hz
                    t = t + P.t_rate

                    # Conduct rate control
                    del_phi, del_theta, del_psi = rate_ctrl.update(q_c, p_c, u_ob.item(3), cf.state)
                    
                    # print("del phi {} del theta {} del psi {}".format(del_phi, del_theta, del_psi))

                    # Update state of model
                    u_pwm = ctrl_mixer.update(u_ob.item(2), del_phi, del_theta, del_psi)
                    
                    # print('u_pwm', u_pwm)
                    
                    u = cf.pwm_to_rpm(u_pwm)

                    # print('u', u)

                    y = cf.update(u)
                    
                    if save_plot:
                        x_list.append(t)
                        y_list.append(57.2958*cf.state.item(3))
                        ref_list.append(r.item(3))
                    
                    # if show_anim:
                    #     plot.update(t, r, cf.state, ctrl)
                    #     anim.update(cf.state)
                    #     plt.pause(0.0000001)

                # if show_anim:
                #     plot.update(t, r, cf.state, ctrl)
                #     anim.update(cf.state)
                #     plt.pause(0.0000001)

            # if show_anim:
            #     plot.update(t, r, cf.state, ctrl)
            #     anim.update(cf.state)
            #     plt.pause(0.0000001)
                
        # Worst animation granularity, very fast
        if show_anim:
            plot.update(t, r, cf.state, ctrl)
            anim.update(cf.state)
            plt.pause(0.1)

    if save_plot:
        ax.plot(x_list, y_list, c='r')
        ax.plot(x_list, ref_list, c='b')
        fig.savefig("../plots/hover_yaw_1rad_sim")

    if show_anim:
        print('Press key to close')
        plt.waitforbuttonpress()
        plt.close()

if __name__ == "__main__":
    # Fly to simultaneous x, y, z, and yaw setpoints
    x_c = 0.1     # [m]
    y_c = 0.0     # [m]
    z_c = 0.0     # [m]
    psi_c = 0.0 # [deg]  
    test_all(x_c, y_c, z_c, psi_c, show_anim=True, save_plot=False) # works! 04/02/2020