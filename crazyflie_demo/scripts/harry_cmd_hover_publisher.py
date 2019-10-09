#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist,Vector3
from crazyflie_driver.msg import Hover

rospy.init_node('velocity publisher')
# pub = rospy.Publisher('crazyflie1/cmd_vel', Twist, queue_size=0)
rate = rospy.Rate(2)
pub = rospy.Publisher('cf1/cmd_hover/', Hover, queue_size=0)

# t = Twist() # instatiate a command 
# t.linear = Vector3(0, 0, 40000)
# t.angular = Vector3(0, 0, 10)

msg = Hover()
msg.vx = 0.0
msg.vy = 0.0
msg.yawrate = 0
msg.zDistance = 0.5
msg.header.seq = 1
msg.header.stamp = rospy.Time.now()
# pub.publish(msg)
# rate.sleep() 

while not rospy.is_shutdown():
    pub.publish(msg)
    rate.sleep()