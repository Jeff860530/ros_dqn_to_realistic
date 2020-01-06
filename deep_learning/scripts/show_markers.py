#!/usr/bin/python

# https://github.com/DavidB-CMU/rviz_tools_py


import numpy
import random

# ROS includes
import roslib
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, Polygon
from tf import transformations, TransformBroadcaster # rotation_matrix(), concatenate_matrices()

import rviz_tools as rviz_tools

from geometry_msgs.msg import PointStamped


# Initialize the ROS Node


# Define exit handler
def cleanup_node():
    print "Shutting down node"
    markers.deleteAllMarkers()

    rospy.on_shutdown(cleanup_node)




def move_target(x,y):
    point1 = Point(x-0.2,y-0.2,0.01)
    point2 = Point(x+0.2,y+0.2,0.01) 
    return point1,point2


def move(data):
    try:
        #data = rospy.wait_for_message('/clicked_point', PoseStamped, timeout=2)
        rospy.loginfo("Get point")
        markers.deleteAllMarkers()
        p1,p2 = move_target(data.point.x, data.point.y)
        #print("position",data.point.x)
        markers.publishRectangle(p1, p2, 'red')   

        br.sendTransform((data.point.x, data.point.y, 0),
                        transformations.quaternion_from_euler(0, 0, 0),
                        rospy.Time.now(),
                        "maker_tf",
                        "map") 
    except:
        rospy.loginfo("Get nothing")
        pass
    
    rospy.Rate(2).sleep() #1 Hz

if __name__ == '__main__':
    print("Init node")
    rospy.init_node('maker', anonymous=False, log_level=rospy.INFO, disable_signals=False)
    br = TransformBroadcaster()
    print("Init maker")
    markers = rviz_tools.RvizMarkers('/map', 'visualization_marker')
    sub = rospy.Subscriber('/clicked_point', PointStamped, move)
    
    rospy.spin()
    
