#!/home/rospc/torch_gpu_ros/bin/python

# modefied from https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning/blob/master/turtlebot3_dqn/nodes/moving_obstacle

import rospy
import time
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelState, ModelStates
import random




class MoveBot():
    def __init__(self):
        #rospy.init_node('moving_target')
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=3)

        self.locationTable = [[1,1],[1,-1],[-1,-1],[-1,1]]
        

    def movingTo(self,goal_x,goal_y,goal_z):
        #pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        
        obstacle = ModelState()
        model = rospy.wait_for_message('gazebo/model_states', ModelStates)
        #print(model.name)
        #print(model.twist[2])
        # print(type(obstacle))
        #time.sleep(5)
        for i in range(len(model.name)):
            if model.name[i] == 'bot':  # the model name is defined in .xacro file
                obstacle.model_name = 'bot'
                
                obstacle.pose = model.pose[i]
                obstacle.pose.position.x = float(goal_x)
                obstacle.pose.position.y = float(goal_y)
                obstacle.twist = Twist()
                obstacle.twist.angular.z = float(goal_z)
                self.pub_model.publish(obstacle)
                #time.sleep(5)

    def movingAt(self,random_bot=False, random_bot_rotate = False):
        #print("bot",random_bot,random_bot_rotate)
        #pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        
        obstacle = ModelState()
        model = rospy.wait_for_message('gazebo/model_states', ModelStates)
 
        for i in range(len(model.name)):
            if model.name[i] == 'dog':  # the model name is defined in .xacro file
                obstacle.model_name = 'dog'
                
                obstacle.pose = model.pose[i]
                #print(self.goalTable[p][0])
                if random_bot:
                    obstacle.pose.position.x = float(random.randint(-10,10)/10.0)
                    obstacle.pose.position.y = float(random.randint(-10,10)/10.0)

                if random_bot_rotate:     
                    obstacle.twist = Twist()
                    obstacle.twist.angular.z = float(random.randint(0,314)/100.0)
                
                if random_bot or random_bot_rotate:
                    self.pub_model.publish(obstacle)
                
                if random_bot and not random_bot_rotate:
                    rospy.loginfo("Random bot location")
                elif random_bot or random_bot_rotate:
                    rospy.loginfo("Random bot location & rotation")
                #time.sleep(5)

        #return float(self.goalTable[p][0]),float(self.goalTable[p][1])
                