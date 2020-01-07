#!/usr/bin/env python

import rospy
import numpy as np
import math
from math import pi

from sensor_msgs.msg import LaserScan

import time
import rospkg
import sys
rospack = rospkg.RosPack()
env_path = rospack.get_path('deep_learning')

sys.path.append(env_path+'/env')


import rospy
from std_msgs.msg import String
def rotate(l, n):
    return l[n:] + l[:n]

def filter(output = 360 , real):
    #data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
    rospy.init_node('laser_filter', anonymous=True)

    
    sub = rospy.Subscriber('scan', LaserScan, readLaser, output)
    rospy.spin()

def readLaser(msg, out, real):
    out = 360 / out
    data = np.array(msg.ranges)
    new_data = []
    for i in range(len(data)):
        if i % out == 0:
            new_data.append(data[i])
    if real:
        new_data = rotate(new_data,12)
    msg.ranges = new_data
    pub.publish(msg)

    
    #print (np.array(msg.ranges))
    #print (len(np.array(msg.ranges)))
    rospy.loginfo("Laser filted !!")

if __name__ == '__main__':
    real = rospy.get_param("~real",False)
    pub = rospy.Publisher('scan_f', LaserScan, queue_size=10)
    filter(24,real=real)