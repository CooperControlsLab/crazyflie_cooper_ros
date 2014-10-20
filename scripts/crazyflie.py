#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist

import time, sys
from threading import Thread

class CrazyflieROS:
    Disconnected = 0
    Connecting = 1
    Connected = 2

    """Wrapper between ROS and Crazyflie SDK"""
    def __init__(self, link_uri):
        self.link_uri = link_uri
        self._cf = Crazyflie()

        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        self._cmdVel = Twist()
        rospy.Subscriber("cmd_vel", Twist, self._cmdVelChanged)

        self._state = CrazyflieROS.Disconnected
        Thread(target=self._update).start()

    def _try_to_connect(self):
        rospy.loginfo("Connecting to %s" % self.link_uri)
        self._state = CrazyflieROS.Connecting
        self._cf.open_link(self.link_uri)

    def _connected(self, link_uri):
        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""

        rospy.loginfo("Connected to %s" % link_uri)
        self._state = CrazyflieROS.Connected

    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie
        at the speficied address)"""
        rospy.logfatal("Connection to %s failed: %s" % (link_uri, msg))
        self._state = CrazyflieROS.Disconnected

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        rospy.logfatal("Connection to %s lost: %s" % (link_uri, msg))
        self._state = CrazyflieROS.Disconnected

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""
        rospy.logfatal("Disconnected from %s" % link_uri)
        self._state = CrazyflieROS.Disconnected

    def _send_setpoint(self):
        roll = self._cmdVel.linear.y
        pitch = self._cmdVel.linear.x
        yawrate = self._cmdVel.angular.z
        thrust = max(10000, int(self._cmdVel.linear.z))
        #print(roll, pitch, yawrate, thrust)
        self._cf.commander.send_setpoint(roll, pitch, yawrate, thrust)

    def _cmdVelChanged(self, data):
        self._cmdVel = data
        self._send_setpoint()

    def _update(self):
        while not rospy.is_shutdown():
            if self._state == CrazyflieROS.Disconnected:
                self._try_to_connect()
            elif self._state == CrazyflieROS.Connected:
                # Crazyflie will shut down if we don't send any command for 500ms
                # Hence, make sure that we don't wait too long
                self._send_setpoint()
                rospy.sleep(0.2)
            else:
                rospy.sleep(0.5)
        self._cf.commander.send_setpoint(0, 0, 0, 0)
        # Make sure that the last packet leaves before the link is closed
        # since the message queue is not flushed before closing
        rospy.sleep(0.1)
        self._cf.close_link()

if __name__ == '__main__':
    rospy.init_node('crazyflie', anonymous=True)
    crazyflieSDK = rospy.get_param("~crazyflieSDK")
    uri = rospy.get_param("~uri")

    sys.path.append(crazyflieSDK)
    import cflib
    from cflib.crazyflie import Crazyflie

    # Initialize the low-level drivers (don't list the debug drivers)
    cflib.crtp.init_drivers(enable_debug_driver=False)

    CrazyflieROS(uri)
    rospy.spin()