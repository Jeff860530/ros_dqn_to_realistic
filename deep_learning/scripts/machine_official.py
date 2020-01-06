#!/home/tedbest/tfp27/bin/python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import os
import json
import numpy as np
import scipy.io as sio
import random
import time
import sys

from collections import deque
from std_msgs.msg import Float32MultiArray

import rospkg
rospack = rospkg.RosPack()
env_path = rospack.get_path('deep_learning')

sys.path.append(env_path+'/env')
from environment_dog1 import Env
import glob

import h5py
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation
import os
import keras
from keras.models import Model,load_model
from keras import layers
from keras import models
import tensorflow as tf
from keras.callbacks import TensorBoard

###

EPISODES = 200

class ReinforceAgent():
    def __init__(self, state_size, action_size):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = env_path+'/model/'
        self.result = Float32MultiArray()
        self.load_model = False
        self.save_model = True

        self.load_memory = False
        self.save_memory = False

        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 2000
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 64
        self.memory = deque(maxlen=20000)
        self.model = self.buildModel()
        self.target_model = self.buildModel()
        self.updateTargetModel()
        #print('prepare_to_load_weight')
        if self.load_model:
            model_file = glob.glob(self.dirPath+"*.h5")
            model_file = [int(i[-17:-13]) for i in model_file]
            print(model_file)
            maxep = max(model_file)
            self.load_episode = maxep
            self.model.load_weights((self.dirPath+ format(maxep, '04d') +"_model_tmp.h5"), by_name = True)
            print("successful load mode",self.dirPath+ format(maxep, '04d') +"_model_tmp.h5")
            with open(self.dirPath+ format(maxep, '04d') +'_model_tmp.json') as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')
        
        if self.load_memory:
            mem_file = glob.glob(self.dirPath+"*.csv")
            mem_file = [int(i[-11:-7]) for i in mem_file]
            print(self.dirPath+ format(mem_file[0], '04d') +"_mem.h5")
            h5f = h5py.File(self.dirPath+ format(mem_file[0], '04d') +"_mem.h5",'r')
            # mem_load = np.load(self.dirPath+ format(mem_file[0], '04d') +"_mem.h5")
            #(state, action, reward, next_state, done)
            state_load = h5f['state'][:]
            action_load = h5f['action'][:]
            reward_load = h5f['reward'][:]
            next_state_load = h5f['next_state'][:]
            done_load = h5f['done'][:]
            h5f.close()
            for m in range(len(action_load)):
                self.appendMemory(state_load[m],
                                    action_load[m],
                                    reward_load[m],
                                    next_state_load[m],
                                    done_load[m])


        #self.model.summary()
        # print('self.model.summary()')
        
        ###################################
        
        for i,layer in enumerate(self.model.layers):
            if i==5 or i==7 :
                layer.trainable=True
            else:
                #layer.trainable=False
                layer.trainable=True
             
        for i,layer in enumerate(self.model.layers):
            if layer.trainable==True:
                print(i,layer.name,'Ture')
            else:
                print(i,layer.name,'False')
        ###################################
        
    def buildModel(self):
        model = Sequential()
        dropout = 0.2
        model.add(Dense(64, input_shape=(26,),activation='relu',name="dense_1", kernel_initializer='lecun_uniform'))
        model.add(Dense(64, activation='relu',name="dense_2", kernel_initializer='lecun_uniform'))
        model.add(Dropout(dropout,name="dropout_1"))
        model.add(Dense(5,name="dense_3",kernel_initializer='lecun_uniform'))

        ##############
        x=Dense(5,activation='relu',name="new_dense_three",kernel_initializer='lecun_uniform')(model.layers[3].input) 
        added = keras.layers.add([model.output, x],name="layers_combine")
        preds=Dense(5,activation='linear',name="activation")(added)
        model=Model(inputs=model.input,outputs=preds)
        ##############
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06))
        #model.summary()
        
        return model


    def getQvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * np.amax(next_target)

    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())

    def getAction(self, state):
        if np.random.rand() <= self.epsilon:
            self.q_value = np.zeros(self.action_size)
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state.reshape(1, len(state)))
            self.q_value = q_value
            return np.argmax(q_value[0])

    def appendMemory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def trainModel(self, target=False):
        mini_batch = random.sample(self.memory, self.batch_size)
        X_batch = np.empty((0, self.state_size), dtype=np.float64)
        Y_batch = np.empty((0, self.action_size), dtype=np.float64)

        for i in range(self.batch_size):
            states = mini_batch[i][0]
            actions = mini_batch[i][1]
            rewards = mini_batch[i][2]
            next_states = mini_batch[i][3]
            dones = mini_batch[i][4]

            q_value = self.model.predict(states.reshape(1, len(states)))
            self.q_value = q_value

            if target:
                next_target = self.target_model.predict(next_states.reshape(1, len(next_states)))

            else:
                next_target = self.model.predict(next_states.reshape(1, len(next_states)))

            next_q_value = self.getQvalue(rewards, next_target, dones)

            X_batch = np.append(X_batch, np.array([states.copy()]), axis=0)
            Y_sample = q_value.copy()

            Y_sample[0][actions] = next_q_value
            Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)

            if dones:
                X_batch = np.append(X_batch, np.array([next_states.copy()]), axis=0)
                Y_batch = np.append(Y_batch, np.array([[rewards] * self.action_size]), axis=0)

        self.model.fit(X_batch, Y_batch, batch_size=self.batch_size, epochs=1, verbose=0,  callbacks=[tbCallBack])

if __name__ == '__main__':
    rospy.init_node('dqn_machine')
    
    tbCallBack = TensorBoard(log_dir=os.getenv("HOME")+"/tboard")
    summary_writer = tf.summary.create_file_writer(os.getenv("HOME")+"/tboard/q3")

    state_size = 26
    action_size = 5

    env = Env(action_size)
    print("Creat ReinforceAgent")
    agent = ReinforceAgent(state_size, action_size)
    scores, episodes = [], []
    global_step = 0
    start_time = time.time()

    done_step = 0
    for e in range(agent.load_episode + 1, EPISODES):
        done = False
        state = env.reset()
        score = 0
        time_out_step = 500

        
        if e > 100:
            env.ramdom_target = True

        if e > 150:
            env.ramdom_bot = True

        if e > 175:
            env.ramdom_bot_rotate = True
        
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
            #get_action.data = [action, score, reward]
            #pub_get_action.publish(get_action)

            if t >= time_out_step:
                rospy.loginfo("Time out!!")
                done = True

            if done:
                done_step += 1

                with summary_writer.as_default():
                    tf.summary.scalar('Total_reward', score,step=done_step)
                    tf.summary.scalar('Average_max_Q_value', np.max(agent.q_value),step=done_step)
                
                
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
            #np.savetxt(agent.dirPath + format(e, '04d') + '_mem.csv', mem2save, delimiter=",",fmt = '%s')
            with h5py.File(agent.dirPath + format(e, '04d') + '_mem.h5', 'w') as hf:
                #(state, action, reward, next_state, done)
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
            # h5f = h5py.File(agent.dirPath + format(e, '04d') + '_mem.h5', 'w')
            # h5f.create_dataset('mem', data=mem2save)
            # h5f.close()

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
    summary_writer.close()

