#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist,Vector3,TransformStamped # twist used in cmd_vel
from crazyflie_driver.msg import Hover # used in cmd_hover commands vel, yaw rate, and hover height
from crazyflie_driver.srv import Takeoff
from std_msgs.msg import Duration
from vicon_bridge.srv import viconGrabPose
import numpy as np
from scipy.spatial.transform import Rotation
import math
import scipy.interpolate as si
import matplotlib.pyplot as plt
from std_msgs.msg import Float64MultiArray

class Tester:
    def __init__(self):
        self.msg = Twist()
        self.hz = 30.0 # if not set to 100, will not broadcast
        self.rate = rospy.Rate(self.hz)
        self.pub = rospy.Publisher('crazyflie/cmd_vel', Twist, queue_size=0)
        # rospy.wait_for_service('/vicon/grab_vicon_pose')
        self.pose_getter = rospy.ServiceProxy('/vicon/grab_vicon_pose', viconGrabPose)
        self.destination = np.array([0,0,0.5])
        self.olddestination = self.destination
        self.circle_radius = 0.1

    def callback(self,data):
        print("callback running, data = ",data.data)
        # self.destination = self.destination + data.data
        # if(self.olddestination.all != self.destination.all):
        #     self.waypointsWithPID(self.circle_radius)


    def destination_listener(self):
        rospy.loginfo("listener running")
        # rospy.init_node('listener', anonymous=True)
        rospy.Subscriber("Destination", Float64MultiArray,self.callback)
        # return 
        rospy.spin()
    
    def getPose(self, vicon_object):
        self.pose = self.pose_getter(vicon_object, vicon_object, 1)
        self.pose1 = self.pose.pose.pose
        return self.pose1
        
    def waypointsWithPID(self, circle_radius):
        # self.destination_listener()
        # REQUIRED TO OVERCOME INITIAL PUBLISHER BLOCK IMPLEMENTED BY USC
        waypoints = self.destination
        self.msg.linear = Vector3(0, 0, 0)
        self.msg.angular = Vector3(0, 0, 0)
        for i in range(100):
            self.pub.publish(self.msg)
            self.rate.sleep()
        
        # Followed this paper, section 3.1, for PID controller
        # https://arxiv.org/pdf/1608.05destination_listenerTODO add caps for all commands
        self.thrust_cap_low = -20000
        self.z_error_before = 0.
        self.z_error_cap = 1.5
        
        # XY controller gains and initialization
        self.x_kp = 10. # Table 3.1.3
        self.x_ki = 2.
        self.y_kp = -10.
        self.y_ki = -2.
        self.x_error_historical = 0.
        self.y_error_historical = 0.
        self.x_before = 0.
        self.y_before = 0.
        self.x_cap = 30.
        self.y_cap = 30.

        # Yaw rate controller gains
        self.yaw_kp = -20. # Table 3.1.3

        # Set initial reference values
        # x_ref = waypoints[0,0]; y_ref = waypoints[0,1]; z_ref = waypoints[0,2]
        origin = self.getPose('crazyflie4')
        self.pose_actual = origin
        no_points = waypoints.shape[0]
        
        # Hold yaw constant throughout
        yaw_ref = 0

        time_step = (1/self.hz)

        while not rospy.is_shutdown():
            # Get current drone pose
            self.pose_before = self.pose_actual
            self.pose_actual = self.getPose('crazyflie4')
            if math.isnan(self.pose_actual.orientation.x): # If nan is thrown, set to last known position
                self.pose_actual = self.pose_before

            # Set reference reference values
            x_ref = waypoints[0]; y_ref = waypoints[1]; z_ref = waypoints[2]

            ### Altitude controller ###

            # Get true z value
            self.z_actual = self.pose_actual.position.z

            # Get error
            self.z_error = z_ref - self.z_actual
            
            # Find integral component
            if self.z_error_historical <= self.z_error_cap:
                self.z_error_historical += (self.z_error * time_step)
            
            # Find derivative component
            self.z_error_der = (self.z_error - self.z_error_before) / time_step
            self.z_error_before = self.z_error

            # Sum PID errors and multiply by gains
            self.z_error_scaled = (self.z_error * self.z_kp) + (self.z_error_historical * self.z_ki) \
                + (self.z_error_der * self.z_kd) # Eq. 3.1.7

            # publish to thrust command
            self.msg.linear.z = self.z_feed_forward + self.z_error_scaled

            ### XY position controller ###

            # get true x and y values
            self.x_actual = self.pose_actual.position.x
            self.y_actual = self.pose_actual.position.y

            # Obtain yaw angle from quaternion
            self.quat_actual = [self.pose_actual.orientation.x, self.pose_actual.orientation.y, \
                self.pose_actual.orientation.z, self.pose_actual.orientation.w]
            R = Rotation.from_quat(self.quat_actual)
            self.global_x = R.apply([1, 0, 0]) # project to world x-axis
            self.yaw_angle = np.arctan2(np.cross([1, 0, 0], self.global_x)[2], \
                np.dot(self.global_x, [1, 0, 0]))
            
            # Find XY error in the drone body frame Eq. 3.1.10
            # Find u (x-velocity) and v (y-velocity) in the body frame
            self.x_error_world = x_ref - self.x_actual
            self.y_error_world = y_ref - self.y_actual

            self.x_e = self.x_error_world * np.cos(self.yaw_angle) + self.y_error_world * np.sin(self.yaw_angle)
            self.u = (self.x_actual - self.x_before) / time_step
            self.x_before = self.x_actual

            self.y_e = -(self.x_error_world * np.sin(self.yaw_angle)) + self.y_error_world * np.cos(self.yaw_angle)
            self.v = (self.y_actual - self.y_before) / time_step
            self.y_before = self.y_actual

            # Eq. 3.1.11 and Eq. 3.1.12
            self.x_diff = self.x_e - self.u
            self.y_diff = self.y_e - self.v

            # Find integral component - store historical error
            self.x_error_historical += (self.x_diff * time_step)
            self.y_error_historical += (self.y_diff * time_step)

            # Sum PI errors and multiply by gains
            self.x_error_scaled = (self.x_diff * self.x_kp) \
                + (self.x_error_historical * self.x_ki)
            self.y_error_scaled = (self.y_diff * self.y_kp) \
                + (self.y_error_historical * self.y_ki)

            # Cap errors to prevent unstable maneuvek
            if self.y_error_scaled >= self.y_cap:
                self.y_error_scaled = self.y_cap
            
            elif self.y_error_scaled <= -self.y_cap:
                self.y_error_scaled = -self.y_cap

            # Plublish commanded actions
            self.msg.linear.x = self.x_error_scaled
            self.msg.linear.y = self.y_error_scaled

            ### Yaw-rate controller Eq. 3.1.13 ###
            self.yaw_error = yaw_ref - self.yaw_angle
            self.yaw_error_scaled = self.yaw_kp * self.yaw_error
            self.msg.angular.z = self.yaw_error_scaled

            ### Useful print statements for debug ###

            # print("The commanded thrust is: {}".format(self.msg.linear.z))
            # print("The z error is {}. Historical error is {}. Derivatice error is {}. Total scaled error is: {}"\
            #     .format(self.z_error, self.z_error_historical, self.z_error_der, self.z_error_scaled)) # HERE
            # print("X command: {}. Y command {}."\
            #     .format(self.x_error_scaled ,self.y_error_scaled))
            # print("The orientation is: {} with type {}".format(self.quat_actual[0], type(self.quat_actual[0])))
            # print('Yaw angle: {}'.format(self.yaw_angle))
            # print('x in body frame: {}'. format(self.x_e))
            # print('y in body frame: {}c
            # print('x ref: {} y ref: {} z ref: {}'.format(x_ref, y_ref, z_ref))

            # (self.z_actual > (z_ref - circle_radius) and self.z_actual < (z_ref + circle_radius)) and \
            # (self.z_actual > z_ref) and \

            # Waypoint incremeneter, last statement ensures drone will stay at last point
            if (self.x_actual > (x_ref - circle_radius) and self.x_actual < (x_ref + circle_radius)) and \
                (self.y_actual > (y_ref - circle_radius) and self.y_actual < (y_ref + circle_radius)) and \
                (self.z_actual > (z_ref - circle_radius) and self.z_actual < (z_ref + circle_radius)):
                # # counter < no_points - 1: # Hover at last point in waypoints array
                # counter += 1
                # print('found next point!!')
                break

            # if counter == no_points: # Land that bitch
            #     print('elif ran!!!!')
            #     break
            
            self.pub.publish(self.msg)
            self.rate.sleep()

def bspline_planning(x, y, sn):
    """
    Path Plannting with B-Spline
    author: Atsushi Sakai (@Atsushi_twi)
    """
    t = range(len(x))
    x_tup = si.splrep(t, x, k=BSN)
    y_tup = si.splrep(t, y, k=BSN)

    x_list = list(x_tup)
    xl = x.tolist()
    x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]

    y_list = list(y_tup)
    yl = y.tolist()
    y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]

    ipl_t = np.linspace(0.0, len(x) - 1, sn)
    rx = si.splev(ipl_t, x_list)
    ry = si.splev(ipl_t, y_list)

    return rx, ry



if __name__ == "__main__":
    print("running")
    rospy.init_node('waypoints listener')
    drone1 = Tester()
    # drone1.waypointsWithPID(drone1.circle_radius)

    try:
        print("trying listener")
        drone1.destination_listener()

    except Exception as e:
        print(e)
        print('Exception was called!!!')