from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.colors import cnames
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys

from crazyflie_dynamics import CrazyflieDynamics
import crazyflie_param as P

class CrazyflieAnimation:
    def __init__(self, traj):
        """
        Create cf animation
        """
        self.flag_init = True
        self.fig, self.ax = plt.subplots()
        self.handle = []
        self.traj = traj
        self.x_list = []
        self.y_list = []
        self.z_list = []

        # Length of leg
        self.d = P.d
    
    def update(self, state):
        """
        Update the cf position and orientation
        """
        x = state.item(0); y = state.item(1); z = state.item(2)
        phi = state.item(5); theta = state.item(4); psi = state.item(3)
        self.drawCrazyflie(x, y, z, phi, theta, psi)

        if self.flag_init == True:
            self.flag_init = False


    def drawCrazyflie(self, x, y, z, phi, theta, psi):
        """
        Plot visualization of crazyflie position and orientation 
        that will get updated in the simulation loop

        Parameters:
        -----------
        x     = x-position  [m]
        y     = y-position  [m]
        z     = z-position  [m]
        phi   = roll angle  [rad]
        theta = pitch angle [rad]
        psi   = yaw angle   [rad]
        """
        # TODO: Find out why the sign convention is reversed for these - is in the rot matrix formulation?
        phi = -phi; theta = -theta; psi = -psi 
        cf_x = [-self.d/np.sqrt(2), self.d/2, self.d/np.sqrt(2),  self.d/np.sqrt(2), -self.d/np.sqrt(2), 0.0]
        cf_y = [ self.d/np.sqrt(2), 0.0,      self.d/np.sqrt(2), -self.d/np.sqrt(2), -self.d/np.sqrt(2), 0.0]
        cf_z = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        pts = np.vstack((cf_x, cf_y, cf_z))

        # Rotation matrix following canada paper convention
        R = np.matrix([
            [np.cos(theta)*np.cos(psi), \
                np.cos(theta)*np.sin(psi), \
                -np.sin(theta)],
            [np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi), \
                np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(theta)*np.cos(psi), \
                np.sin(phi)*np.cos(theta)],
            [np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi), \
                np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi), \
                np.cos(phi)*np.cos(theta)]
        ])
        # rotate the drawing by the rotation matrix
        pts = R*pts
        # translate the drawing
        pts[0] += x; pts[1] += y; pts[2] += z

        # append the historical cartesian positions
        self.x_list.append(x); self.y_list.append(y); self.z_list.append(z)

        # plot the drone
        ax = plt.axes(projection='3d')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-0.5, 1.5)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.scatter(pts[0], pts[1], pts[2], s=1, c='k')

        # plot the drone path
        ax.plot(self.x_list, self.y_list, self.z_list, c='r', label='path', alpha=0.5)

        # plot hover starting point and goal or trajectory depending on controller used
        if self.traj.shape[0] > 3: # traj traking controller
            ax.plot(self.traj[:,0], self.traj[:,3], self.traj[:,6], c='b')
        else: # hover controller
            ax.scatter(0.0, 0.0, 0.0, c='g', marker='x', s=20, label='start')
            ax.scatter(self.traj[0], self.traj[1], self.traj[2], c='b', marker='x', s=20, label='goal')
            ax.legend()

if __name__ == "__main__":
    cf = CrazyflieDynamics()
    crazyflieAnimation = CrazyflieAnimation()

    # Test some static frames
    crazyflieAnimation.drawCrazyflie(x= 0.0, y=0.5, z=0.5, phi=0.0, theta=0.0, psi=1.0)
    plt.show()