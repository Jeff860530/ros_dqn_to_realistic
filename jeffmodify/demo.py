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


def localization():
	pass

def getting_start():
	print('Please put the bot in the real map')
	raw_input('If done,press Enter to continue')
	print('Starting localization')	
	localization()
	print('Finish localization')
	raw_input('bot is mactch the position in rviz,press Enter to continue')


if __name__ == "__main__":
	getting_start()
	pass

	while True :
		raw_input('Please point the target on rviz,press Enter to continue')
		try:    
			a = input('enter number:')
			#target  =  point on rviz 
		except Exception as e:        
			print('e')
			print("target is too close to move,please repoint the target")


