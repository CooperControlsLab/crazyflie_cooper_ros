import numpy as np
import matplotlib.pyplot as plt 

import sys
sys.path.append("../")
sys.path.append("../../scripts")

from a_traj_generator import StandingWaveGenerator, TrajGenerator
from data_plotter import DataPlotter
from crazyflie_dynamics import CrazyflieDynamics
from crazyflie_controller import RateController, AttitudeController, ControlMixer, AltitudeController, XYController, YawController, XYTrajController
import crazyflie_param as P
from crazyflie_animation import CrazyflieAnimation

def test_traj(traj, show_anim=True, save_plot=False):
    init_pos = np.array([traj[0][0], traj[0][3], traj[0][6]])
    cf = CrazyflieDynamics(init_pos=init_pos)

    if show_anim:
        plot = DataPlotter(is_type="traj")
        anim = CrazyflieAnimation(traj)

    # Create class objects
    rate_ctrl = RateController(kp_p=70.0, \
        kp_q=70.0, kp_r=70.0, ki_r=16.7)
    attitude_ctrl = AttitudeController(kp=3.5, ki=2.0, kd=0.0)
    ctrl_mixer = ControlMixer()
    altitiude_ctrl = AltitudeController()
    # xy_ctrl = XYController(t=P.t_ob, kp=20.0, ki=2.0, cap=0.1396)
    xy_traj_ctrl = XYTrajController(kp=100.0, kd=100.0)
    yaw_ctrl = YawController(kp=0.1, cap=200.0)

    # off-borad controller input values
    u_ob = np.array([
        [0.0], # pitch (phi -> x)  - 0
        [0.0], # roll (theta -> y) - 1
        [0.0], # yaw rate          - 2
        [0.0], # thrust            - 3
    ])

    z_c = 0.0
    psi_c = 0.0

    t = P.t_start

    if save_plot:
        fig, ax = plt.subplots()
        ax.set_title("CF Altitude Simulation")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("z [m]")
        ax.grid()
        x_list = []
        y_list = []
        ref_list = []

    # while t < P.t_end: # plotter can run the slowest
    
    # for i in range(10):
    #     t_next_plot = t + P.t_plot

    # while t < t_next_plot: # offboard controller is slowest at 100 hz
    for i in range(traj.shape[0] - 1):
        t_next_ob = t + P.t_ob

        # reference values
        ref = np.array([
            [traj[i, 0]], # x
            [traj[i, 1]], # xd
            [traj[i, 3]], # y
            [traj[i, 4]], # yd
        ]) 

        # Altitude off-board controller update
        u_ob[3,0] = altitiude_ctrl.update(ref.item(2), cf.state.item(2))

        # Yaw rate off-board controller update
        u_ob[3,0] = yaw_ctrl.update(ref.item(3), cf.state.item(3))

        r_t      = np.array([traj[i, 0], traj[i, 3]]) # traj pos values
        r_t_vect = np.array([traj[i+1, 0], traj[i+1, 3]]) - r_t # vector from current pos to next pos in traj
        rd_t     = np.array([traj[i, 1], traj[i, 4]]) # traj vel values
        rdd_t    = np.array([traj[i, 2], traj[i, 5]])

        # X-Y off-board controller update
        u_ob[0,0], u_ob[1,0] = xy_traj_ctrl.update(r_t, rd_t, r_t_vect, cf.state, psi_c, rdd_t)

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
                
                if save_plot:
                    x_list.append(t)
                    y_list.append(cf.state.item(2))
                    ref_list.append(ref.item(2))

                # if show_anim:
                #     plot.update(t, ref, cf.state, u, is_type="traj")
                #     anim.update(cf.state)
                #     plt.pause(0.0000001)
            
            # if show_anim:
            #     plot.update(t, ref, cf.state, u, is_type="traj")
            #     anim.update(cf.state)
            #     plt.pause(0.0000001)

        if show_anim:
            plot.update(t, ref, cf.state, u, is_type="traj")
            anim.update(cf.state)
            plt.pause(0.00000001)    

        if save_plot:
            ax.plot(x_list, y_list, c='r')
            ax.plot(x_list, ref_list, c='b')
            fig.savefig("../plots/hover_alt_1m_sim")

    if show_anim:
        print('Press key to close')
        plt.waitforbuttonpress()
        plt.close()

if __name__ == "__main__":
    # Circle Traj
    z_c = 0.0
    y_c = 0.0
    x_c = 0.5
    psi_c = 0.0

    traj_gen = TrajGenerator()
    x_center = 0.0; y_center = 0.0
    omega = 1.0
    no_osc = 2.0
    circle_traj = traj_gen.genCircleTraj(x_c, y_c, x_center, y_center, \
        omega, no_osc, CCW=False)

    # print(circle_traj.shape[1])
    test_traj(circle_traj)
