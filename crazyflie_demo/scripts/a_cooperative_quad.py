#!/usr/bin/env python
import numpy as np
import math
# import scipy.interpolate as si
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from mpl_toolkits import mplot3d
import pickle

# Import crazyflie model modules
from a_cf_controller_phys import AltitudeControllerPhys, XYControllerPhys, YawControllerPhys, XYControllerTrajPhys
import sys
sys.path.append("../model/")
from data_plotter import DataPlotter
import crazyflie_param as P

# Import ros specifc modules
import rospy
from geometry_msgs.msg import Twist, Vector3, TransformStamped # twist used in cmd_vel
from vicon_bridge.srv import viconGrabPose

class CooperativeQuad:
    def __init__(self, cf_name):
        rospy.init_node('test', anonymous=True)
        self.cf_name = cf_name
        self.msg = Twist()
        self.hz = 30.0
        self.t_phys = 1/self.hz # TODO make P.t_phys import
        self.rate = rospy.Rate(self.hz)

        self.pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        # self.pub = rospy.Publisher(self.cf_name + "/cmd_vel", Twist, queue_size=10)

        # TOPIC
        self.pose = TransformStamped()
        self.pose.transform.rotation.w = 1.0
        # TOPIC

    def callback(self, pose):
        self.pose = pose
    
    def listener(self):
        rospy.Subscriber("/vicon/" + self.cf_name + "/" + self.cf_name, TransformStamped, self.callback)
        pose = self.pose
        rospy.spin()

    def dummyForLoop(self):
        # REQUIRED TO OVERCOME INITIAL PUBLISHER BLOCK IMPLEMENTED BY USC
        self.msg.linear = Vector3(0, 0, 0)
        self.msg.angular = Vector3(0, 0, 0)
        for _ in range(100):
            self.pub.publish(self.msg)
            self.rate.sleep()

    def hoverStiff(self, x_c, y_c, z_c, yaw_c, goal_r, is_break=True):
        """
        Hovers the drone to an accurate global setpoint
        Drone will stay at setpoint until other function is called
        Stiff refers to optimization for global positional accuracy

        Parameters
        ----------
        x_c, y_c, z_c, yaw_c = reference setpoints
        goal_r = bounding radius for when drone is "close enough" to commanded setpoint
        """
        print(self.cf_name + ' started hover controller')

        rospy.Subscriber("/vicon/" + self.cf_name + "/" + self.cf_name, TransformStamped, self.callback)
        pose = self.pose

        # Initialize required hover controllers
        altitude_ctrl_phys = AltitudeControllerPhys()
        xy_ctrl_phys = XYControllerPhys()
        yaw_ctrl_phys = YawControllerPhys()
        # print("after class declarations")
        
        while not rospy.is_shutdown():
            pose_prev = pose
            pose = self.pose
            quat = [pose.transform.rotation.x, pose.transform.rotation.y, pose.transform.rotation.z, pose.transform.rotation.w]
            x = pose.transform.translation.x; y = pose.transform.translation.y; z = pose.transform.translation.z
            if math.isnan(pose.transform.translation.x): # handle nans by setting to last known position
                pose = pose_prev

            # Obtain yaw angle from quaternion
            R = Rotation.from_quat(quat)
            x_global = R.apply([1, 0, 0]) # project to world x-axis
            yaw = np.arctan2(np.cross([1, 0, 0], x_global)[2], np.dot(x_global, [1, 0, 0]))

            self.msg.linear.z = altitude_ctrl_phys.update(z_c, z)
            self.msg.linear.x, self.msg.linear.y = xy_ctrl_phys.update(x_c, x, y_c, y, yaw)
            self.msg.angular.z = yaw_ctrl_phys.update(yaw_c, yaw)

            ### Goal behavior ###
            if is_break:
                if (x > (x_c - goal_r) and x < (x_c + goal_r)) and \
                    (y > (y_c - goal_r) and y < (y_c + goal_r)) and \
                    (z > (z_c - goal_r) and z < (z_c + goal_r)):
                    print(self.cf_name + ' found the hover setpoint!')
                    break # include to move to other function

            self.pub.publish(self.msg)
            self.rate.sleep()

    def trajTrackingStandingWave(self, traj, z_c):
        """
        Runs a trajectory tracking algorithm that follows a standing wave

        Parameters
        ----------
        traj = trajectory that increments at each loop iteration
        """
        print(self.cf_name + ' started tracking standing wave controller')

        rospy.Subscriber("/vicon/" + self.cf_name + "/" + self.cf_name, TransformStamped, self.callback)
        pose = self.pose

        # Initialize required controllers
        altitude_ctrl_phys = AltitudeControllerPhys()
        xy_ctrl_phys = XYControllerPhys()
        xy_traj_ctrl_phys = XYControllerTrajPhys()
        yaw_ctrl_phys = YawControllerPhys()
        
        y_c = 0.0; v_c = 0.0 # keep y values equal to zero for now 
        yaw_c = 0.0

        # Will finish at end of trajectory matrix, 1 entry per loop interation
        for i in range(traj.shape[0] - 1):
            print('completion stage is {} out of {}'.format(i, traj.shape[0]))
            print('traj x is {}'.format(traj[i, 0]))
            pose_prev = pose
            pose = self.pose
            quat = [pose.transform.rotation.x, pose.transform.rotation.y, pose.transform.rotation.z, pose.transform.rotation.w]
            x = pose.transform.translation.x; y = pose.transform.translation.y; z = pose.transform.translation.z
            if math.isnan(pose.transform.translation.x): # handle nans by setting to last known position
                pose = pose_prev
        
            # Obtain yaw angle from quaternion
            R = Rotation.from_quat(quat)
            x_global = R.apply([1, 0, 0]) # project to world x-axis
            yaw = np.arctan2(np.cross([1, 0, 0], x_global)[2], np.dot(x_global, [1, 0, 0]))

            # TODO: make flexible with y values
            r_t      = np.array([traj[i, 0], y_c]) # traj pos values
            r_t_vect = np.array([traj[i+1, 0], y_c]) - r_t # vector from current pos to next pos in traj
            rd_t     = np.array([traj[i, 1], v_c]) # traj vel values
            r        = np.array([x, y]) # actual drone pos

            self.msg.linear.z = altitude_ctrl_phys.update(z_c, z)
            self.msg.linear.x, self.msg.linear.y = xy_traj_ctrl_phys.update(r_t, rd_t, r_t_vect, r, yaw_c)
            self.msg.angular.z = yaw_ctrl_phys.update(yaw_c, yaw)

            print('x commanded val is: ', self.msg.linear.x)
            print('y commanded val is: ', self.msg.linear.y)

            self.pub.publish(self.msg)
            self.rate.sleep()
        
        # Save out data through pickle
        xy_traj_ctrl_phys.exportPlotData()

    def land(self):
        print(self.cf_name + ' land function called')
        self.hoverStiff(0.0, 0.0, 0.2, 0.0, 0.075)

def main():
    try:
        # Initialize drone control class with arg matching vicon object name
        cf1 = CooperativeQuad('crazyflie4')
        cf1.dummyForLoop()

        # Hover at z=0.5, works tested 1/27/2020
        goal_r = 0.1
        cf1.hoverStiff(0.0, 0.0, 0.5, 0.0, goal_r)
        cf1.land()

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()