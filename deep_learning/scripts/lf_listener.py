#!/usr/bin/env python  
import rospy
import math
import tf
import geometry_msgs.msg

if __name__ == '__main__':
    rospy.init_node('turtle_tf_listener')

    listener = tf.TransformListener()
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            (trans,rot) = listener.lookupTransform('/map', '/maker_tf', rospy.Time(0))
            rospy.loginfo(format(trans[0])+" / "+format(trans[1]))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.loginfo("no message")
            continue

        #rospy.loginfo(trans[1], trans[0])

        rate.sleep()
