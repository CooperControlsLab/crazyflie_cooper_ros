import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import crazyflie model modules
import sys
sys.path.append("../")
from data_plotter import DataPlotter
from crazyflie_dynamics import CrazyflieDynamics
from crazyflie_controller import RateController, AttitudeController, ControlMixer, AltitudeController, XYController, YawController
import crazyflie_param as P
from crazyflie_animation import CrazyflieAnimation

def test_all(x_c, y_c, z_c, psi_c, show_anim=True, save_plot=False, plot_type='x'):
    cf = CrazyflieDynamics(init_pos=np.array([0.0, 0.0, 0.0]))

    if show_anim:
        plot = DataPlotter()
        traj = np.array([x_c, y_c, z_c])
        anim = CrazyflieAnimation(traj)

    # Create class objects
    rate_ctrl = RateController()
    attitude_ctrl = AttitudeController()
    ctrl_mixer = ControlMixer()
    altitiude_ctrl = AltitudeController()
    xy_ctrl = XYController()
    yaw_ctrl = YawController()

    # off-borad controller input values
    u_ob = np.array([
        [0.0], # pitch (theta ->  x) - 0 [deg]
        [0.0], # roll  (phi   -> -y) - 1 [deg]
        [0.0], # thrust              - 2 [PWM]
        [0.0], # yaw rate            - 3 [deg/s]
    ])

    # reference values
    r = np.array([
        [x_c], # x                  - 0 [m]
        [y_c], # y                  - 1 [m]
        [z_c], # z                  - 2 [m]
        [psi_c], # psi              - 3 [deg]
    ]) 

    t = P.t_start

    if save_plot:
        fig, ax = plt.subplots()
        ax.grid()
        x_list = []
        y_list = []
        ref_list = []
        ax.set_xlabel("time [s]")

        if plot_type == 'x':
            ax.set_title("CF X Simulation")
            ax.set_ylabel("x [m]")
        elif plot_type == 'y':
            ax.set_title("CF Y Simulation")
            ax.set_ylabel("y [m]")
        elif plot_type == 'z':
            ax.set_title("CF Z Simulation")
            ax.set_ylabel("z [m]")
        elif plot_type == 'psi':
            ax.set_title("CF Yaw Rate Simulation")
            ax.set_ylabel("psi [rad]")

    while t < P.t_end: # plotter can run the slowest
        t_next_plot = t + P.t_plot
        
        while t < t_next_plot: # offboard controller is slowest at 100 hz
            t_next_ob = t + P.t_ob
            # t_next_phys = t + P.t_phys
            
            # Altitude off-board controller update
            # thrust                          z_c        z
            u_ob[2,0] = altitiude_ctrl.update(r.item(2), cf.state.item(2))

            # XY off-borad controller update
            # theta_c, phi_c                      x_c      , y_c
            u_ob[0,0], u_ob[1,0] = xy_ctrl.update(r.item(0), r.item(1), cf.state)

            # Yaw rate off-board controller update
            # r_c                       psi_c      psi
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
                q_c, p_c = attitude_ctrl.update(u_ob.item(0), u_ob.item(1), cf.state)

                while t < t_next_att: # rate controller is the fastest running at 500 hz
                    t = t + P.t_rate

                    # Conduct rate control
                    del_phi, del_theta, del_psi = rate_ctrl.update(q_c, p_c, u_ob.item(3), cf.state)

                    # Update state of model
                    u_pwm = ctrl_mixer.update(u_ob.item(2), del_phi, del_theta, del_psi)
                    u_pwm = cf.saturate(u_pwm, P.input_limits)
                    u = cf.pwm_to_rpm(u_pwm)
                    y = cf.update(u)
                    
                    if save_plot:
                        # x_list.append((100.0/30.0)*t)
                        x_list.append(t)
                        if plot_type == 'x':
                            y_list.append(cf.state.item(0))
                            ref_list.append(r.item(0))
                        elif plot_type == 'y':
                            y_list.append(cf.state.item(1))
                            ref_list.append(r.item(1))
                        elif plot_type == 'z':
                            y_list.append(cf.state.item(2))
                            ref_list.append(r.item(2))
                        elif plot_type == 'psi':
                            y_list.append(cf.state.item(3))
                            ref_list.append(0.0174533*r.item(3)) # convert to rad
                    
                    # if show_anim:
                    #     plot.update(t, r, cf.state, ctrl)
                    #     anim.update(cf.state)
                    #     plt.pause(0.0000001)

                # if show_anim:
                #     plot.update(t, r, cf.state, ctrl)
                #     anim.update(cf.state)
                #     plt.pause(0.0000001)

            if show_anim:
                plot.update(t, r, cf.state, ctrl)
                anim.update(cf.state)
                plt.pause(0.0000001)
                
        # # Worst animation granularity, very fast
        # if show_anim:
        #     plot.update(t, r, cf.state, ctrl)
        #     anim.update(cf.state)
        #     plt.pause(0.0000001)

    if save_plot:
        ax.plot(x_list, y_list, c='r')
        ax.plot(x_list, ref_list, c='b')
        if plot_type == 'x':
            fig.savefig("../plots/hover_x_1m_sim_3")
        elif plot_type == 'y':
            fig.savefig("../plots/hover_y_1m_sim_3")
        elif plot_type == 'z':
            fig.savefig("../plots/hover_z_1m_sim_3")
        elif plot_type == 'psi':
            fig.savefig("../plots/hover_yaw_1rad_sim_3")

    if show_anim:
        print('Press key to close')
        plt.waitforbuttonpress()
        plt.close()

if __name__ == "__main__":
    # Fly to simultaneous x, y, z, and yaw setpoints
    x_c = 1.0     # [m]
    y_c = 1.0     # [m]
    z_c = 1.0     # [m]
    psi_c = 0.0   # [deg]  
    test_all(x_c, y_c, z_c, psi_c, show_anim=False, save_plot=True, \
        plot_type='z') # works! 04/08/2020