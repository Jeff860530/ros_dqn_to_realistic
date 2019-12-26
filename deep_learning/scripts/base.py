#!/usr/bin/python

import serial
import time
import sys, select, termios, tty
import rospy
import tf
import math
import string
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class BaseControl:
    def __init__(self):        

        self.wheelRad = float(0.072/2.0) #m
        self.wheelSep = float(0.17) #m
        self.scale = rospy.get_param('~scale', '1.0')

        try:
            self.serial = serial.Serial('/dev/ttyACM0' , 115200, timeout= 0.5 )
            rospy.loginfo("Connect success ...")

            try:
                print ("Flusing first 20 data readings ...")
                for x in range(0, 20):
                    data = self.serial.read()
                    time.sleep(0.01)
            except:
                print ("Flusing faile ")
                sys.exit(0)

        except serial.serialutil.SerialException:
            rospy.logerr("Can not receive data from the port: "#+ self.device_port + 
            ". Did you specify the correct port ?")
            self.serial.close
            sys.exit(0) 
        rospy.loginfo("Communication success !")

        # rospy.loginfo("Flusing first 50 data readings ...")
        #     for x in range(0, 50):
        #         self.serial.readline().strip()
        #         time.sleep(0.01)

        # ROS handler        
        self.sub = rospy.Subscriber('/car/cmd_vel', Twist, self.cmdCB, queue_size=10)
        #self.pub = rospy.Publisher(self.odom_topic, Odometry, queue_size=10)


        #self.timer_odom = rospy.Timer(rospy.Duration(0.1), self.timerOdomCB)
        self.timer_cmd = rospy.Timer(rospy.Duration(0.1), self.timerCmdCB) # 10Hz
        #self.tf_broadcaster = tf.TransformBroadcaster()


        # variable        
        self.trans_x = 0.0 # cmd
        self.rotat_z = 0.0
        self.WL_send = 0.0
        self.WR_send = 0.0
        self.current_time = rospy.Time.now()
        self.previous_time = rospy.Time.now()
        self.pose_x = 0.0 # SI
        self.pose_y = 0.0
        self.pose_yaw = 0.0
        # self.WL = 0
        # self.WR = 0
     
    def cmdCB(self, data):
        self.trans_x = data.linear.x
        self.rotat_z = data.angular.z
        rospy.loginfo("Vx:{}  W: {}".format(self.trans_x , self.rotat_z))

    

    #################################################################################################

    def timerCmdCB(self, event):

        self.WR_send = int(self.trans_x + self.wheelSep*self.rotat_z*2)
        self.WL_send = int(self.trans_x - self.wheelSep*self.rotat_z*2)
        # rospy.logerr("WR_send: "+ chr(self.WR_send))
        # rospy.logerr("WL_send: "+ chr(self.WL_send))       
        if self.WR_send < 0:
            R_forward = 0
        else:
        	R_forward = 1
        if self.WL_send < 0:
            L_forward = 0
        else:
        	L_forward = 1
    	self.WR_send = abs(self.WR_send)
    	self.WL_send = abs(self.WL_send)
        if self.WR_send > 255:
            self.WR_send = 255
        if self.WL_send > 255:
            self.WL_send = 255
        self.WL_send = str(self.WL_send)
        self.WR_send = str(self.WR_send)
        while len(self.WL_send)<3:
        	self.WL_send = "0"+self.WL_send
    	while len(self.WR_send)<3:
        	self.WR_send = "0"+self.WR_send
        #output = chr(255) + chr(254) + chr(self.WL_send) + chr(L_forward) + chr(self.WR_send) + chr(R_forward)   
        output = "(" + str(L_forward) + str(self.WL_send) + str(R_forward) + str(self.WR_send)+")"     
        #print output     
        #rospy.loginfo(output.encode())
        self.serial.write(output)
        rospy.loginfo(output)

        
if __name__ == "__main__":
    try:    
        # ROS Init    
        rospy.init_node('base_control', anonymous=True)

        # Constract BaseControl Obj
        rospy.loginfo("Bot Base Control ...")
        bc = BaseControl()
        rospy.spin()
    except KeyboardInterrupt:    
        bc.serial.close        
        print("Shutting down")
