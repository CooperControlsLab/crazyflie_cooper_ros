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

def test_traj(traj, traj_type='circ', show_anim=True, save_plot=False, \
    plot_type='xy'):
    init_pos = np.array([traj[0][0], traj[0][3], traj[0][6]])
    cf = CrazyflieDynamics(init_pos=init_pos)

    if show_anim:
        plot = DataPlotter(is_type="traj")
        anim = CrazyflieAnimation(traj)

    # Create class objects
    rate_ctrl = RateController()
    attitude_ctrl = AttitudeController()
    ctrl_mixer = ControlMixer()
    altitiude_ctrl = AltitudeController()
    xy_traj_ctrl = XYTrajController()
    yaw_ctrl = YawController()

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
        if plot_type == 'xy':
            fig, ax = plt.subplots()
            ax.set_title("Circular Trajectory XY Data Sim")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.grid()
            x_list = []
            y_list = []
            xref_list = []
            yref_list = []

        elif plot_type == 'comp':
            fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10,12))

            ax[0].set_ylabel('x [m]'); ax[0].set_title('CF Sim Data')
            ax[1].set_ylabel('xd [m/s]')
            ax[2].set_ylabel('y [m]')
            ax[3].set_xlabel('t [s]'); ax[3].set_ylabel('yd [m/s]')

            t_list = []
            x_list = []
            xref_list = []

            xd_list = []
            xdref_list = []
            
            y_list = []
            yref_list = []

            yd_list = []
            ydref_list = []

    # Use for loop to ensure at correct point in trajectory
    for i in range(traj.shape[0] - 1):
        # t_next_phys = t + P.t_phys
        t_next_ob = t + P.t_ob

        # reference values
        if traj_type == 'circ':
            ref = np.array([
                [traj[i, 0]], # x
                [traj[i, 1]], # xd
                [traj[i, 3]], # y
                [traj[i, 4]], # yd
            ]) 
        elif traj_type == 'sw':
            ref = np.array([
                [traj[i, 0]], # x
                [traj[i, 1]], # xd
                [0.0], # y
                [0.0], # yd
            ]) 

        # Altitude off-board controller update
        u_ob[3,0] = altitiude_ctrl.update(ref.item(2), cf.state.item(2))

        # Yaw rate off-board controller update
        u_ob[3,0] = yaw_ctrl.update(ref.item(3), cf.state.item(3))
        
        if traj_type == 'circ':
            r_t      = np.array([traj[i, 0], traj[i, 3]]) # traj pos values
            rd_t     = np.array([traj[i, 1], traj[i, 4]]) # traj vel values
        elif traj_type == 'sw':
            r_t      = np.array([traj[i, 0], 0.0])
            rd_t     = np.array([traj[i, 1], 0.0])

        r_t_vect = np.array([traj[i+1, 0], traj[i+1, 3]]) - r_t # vector from current pos to next pos in traj
        rdd_t    = np.array([traj[i, 2], traj[i, 5]])

        # X-Y off-board controller update
        u_ob[0,0], u_ob[1,0] = xy_traj_ctrl.update(r_t, rd_t, r_t_vect, cf.state, psi_c, rdd_t)

        # while t < t_next_phys: # attitude controller runs at 250 hz
        while t < t_next_ob:
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
            if plot_type == 'xy':
                x_list.append(cf.state.item(0)) # x
                y_list.append(cf.state.item(1)) # y
                xref_list.append(ref.item(0)) # xref
                yref_list.append(ref.item(2)) # yref
            elif plot_type == 'comp':
                t_list.append(t)
                x_list.append(cf.state.item(0))
                xref_list.append(ref.item(0))

                xd_list.append(cf.state.item(6))
                xdref_list.append(ref.item(1))
                
                y_list.append(cf.state.item(1))
                yref_list.append(ref.item(2))

                yd_list.append(cf.state.item(7))
                ydref_list.append(ref.item(3))

        if show_anim:
            # plot.update((100.0/30.0)*t, ref, cf.state, u, is_type="traj")
            plot.update(t, ref, cf.state, u, is_type="traj")
            anim.update(cf.state)
            plt.pause(0.00000001)

    if save_plot:
        if plot_type == 'xy':
            ax.plot(x_list, y_list, c='r')
            ax.plot(xref_list, yref_list, c='b')
            fig.savefig("../plots/traj_circ_sim_2omega_20200408")
        elif plot_type == 'comp':
            ax[0].plot(t_list, x_list, c='r')
            ax[0].plot(t_list, xref_list, c='b')
            ax[1].plot(t_list, xd_list, c='r')
            ax[1].plot(t_list, xdref_list, c='b')
            ax[2].plot(t_list, y_list, c='r')
            ax[2].plot(t_list, yref_list, c='b')
            ax[3].plot(t_list, yd_list, c='r')
            ax[3].plot(t_list, ydref_list, c='b')
            fig.savefig("../plots/traj_sim_20200411")

if __name__ == "__main__":
    # Circle Traj
    z_c = 0.0
    y_c = 0.0
    x_c = 0.5
    psi_c = 0.0

    traj_gen = TrajGenerator(hz=P.freq_off_board)
    x_center = 0.0; y_center = 0.0
    omega = 1.0
    no_osc = 3.0
    circle_traj = traj_gen.genCircleTraj(x_c, y_c, x_center, y_center, \
        omega, no_osc, CCW=True)

    test_traj(circle_traj, traj_type='circ', show_anim=False, \
        save_plot=True, plot_type='comp')