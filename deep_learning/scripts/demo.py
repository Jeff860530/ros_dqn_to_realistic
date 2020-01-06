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


import rospkg
rospack = rospkg.RosPack()
env_path = rospack.get_path('deep_learning')

sys.path.append(env_path+'/env')
from environment_dog1 import Env

import machine_offical


def localization():
    #raw_input('watting for localization,If done,press Enter to continue')
	#pass
def start():
	print('Please put the bot in the real map')
	raw_input('If done,press Enter to continue')
	print('Starting localization')	
	localization()
	print('Finish localization')
	raw_input('bot is mactch the position in rviz,press Enter to continue')

def finish():
	raw_input('Finishing Demo,press Enter to finish')


def send_action_to_arduino(get_action):
	'''
    #add in main#
    rospy.init_node('AB_teleop')
    pub = rospy.Publisher('/car/cmd_vel', Twist, queue_size=5)
    '''
    action = get_action
    target_linear_vel = 0
    target_angular_vel = 0
    
    if action == 0 :
        target_angular_vel = 
    elif action == 1 :
        target_angular_vel = 
    elif action == 2 :
        target_angular_vel = 
    elif action == 3 :
        target_angular_vel = 
    elif action == 4 :
        target_angular_vel = 
    elif action == 5 :
        target_angular_vel = 
    twist = Twist()
    twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = target_angular_vel
    pub.publish(twist)

def main():
    rospy.init_node('real_dqn')

    result = Float32MultiArray()
    get_action = Float32MultiArray()
    
    tbCallBack = TensorBoard(log_dir=os.getenv("HOME")+"/tboard")
    summary_writer = tf.summary.create_file_writer(os.getenv("HOME")+"/tboard/real")

    state_size = 26
    action_size = 5

    env = Env(action_size)


    print("Creat ReinforceAgent")
    agent = machine_offical.ReinforceAgent(state_size, action_size)


    scores, episodes = [], []
    global_step = 0
    start_time = time.time()

    done_step = 0
    for e in range(agent.load_episode + 1, EPISODES):
        done = False

        ### manual reset
        raw_input('Manualing reset bot.If done,press Enter to continue')
        ###
        time.sleep(3)
        ### manual choose target
        raw_input('Manualing choose target.If done,press Enter to continue')
        ####mark to get target##########################################


        ###rplidar scan_f to get state
        state = env.reset(real = True)

        score = 0
        time_out_step = 500


        for t in range(agent.episode_step):
            action = agent.getAction(state)

            next_state, reward, done, goal = env.step(action)

            agent.appendMemory(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                if global_step <= agent.target_update:
                    agent.trainModel()
                else:
                    agent.trainModel(True)
            
            score += reward
            state = next_state
            get_action.data = [action, score, reward]
            pub_get_action.publish(get_action)
			send_action_to_arduino(get_action)###################################get_action means action to gazbo
            if t >= time_out_step:
                rospy.loginfo("Time out!!")
                done = True

            if done:
                done_step += 1
                ##
                with summary_writer.as_default():
                    tf.summary.scalar('Total_reward', score,step=done_step)
                    tf.summary.scalar('Average_max_Q_value', np.max(agent.q_value),step=done_step)
                
                
                # summary_writer.summary.scalar("Total reward", score)
                # summary_writer.summary.scalar("Average mac Q-value", np.max(agent.q_value))
                ##
                result.data = [score, np.max(agent.q_value)]
                pub_result.publish(result)
                agent.updateTargetModel()
                scores.append(score)
                episodes.append(e)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
                              e, score, len(agent.memory), agent.epsilon, h, m, s)
                param_keys = ['epsilon']
                param_values = [agent.epsilon]
                param_dictionary = dict(zip(param_keys, param_values))
                if not goal:
                    break
                else:
                    time_out_step = t + 400

            global_step += 1
            if global_step % agent.target_update == 0:
                rospy.loginfo("UPDATE TARGET NETWORK")

        if e % 5 == 0 and agent.save_model:
            agent.model.save(agent.dirPath + format(e, '04d') + '_model_tmp.h5')
            with open(agent.dirPath + format(e, '04d') + '_model_tmp.json', 'w') as outfile:
                json.dump(param_dictionary, outfile)

        if e % 5 == 0 and len(agent.memory) > 1200 and agent.save_memory:
            mem2save = np.array(agent.memory)[-1000:]
            with h5py.File(agent.dirPath + format(e, '04d') + '_mem.h5', 'w') as hf:
                hf.create_dataset("state", (1000,26), np.float64)
                hf.create_dataset("action", (1000,), np.int8)
                hf.create_dataset("reward", (1000,), np.float64)
                hf.create_dataset("next_state", (1000,26), np.float64)
                hf.create_dataset("done", (1000,1), np.bool)
                hf["state"][...] = np.array([i[0] for i in mem2save])
                hf["action"][...] = np.array([i[1] for i in mem2save])
                hf["reward"][...] = np.array([i[2] for i in mem2save])
                hf["next_state"][...] = np.array([i[3] for i in mem2save])
                hf["done"][...] = np.array([i[4] for i in mem2save])
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
    summary_writer.close()
if __name__ == "__main__":
	start()
	'''
	while True :
		raw_input('Please point the target on rviz,press Enter to continue')
		try:    
			a = input('enter number:')
			#target  =  point on rviz 
		except Exception as e:        
			print('e')
			print("target is too close to move,please repoint the target")
		
	'''
	#DQN()
	finish()


