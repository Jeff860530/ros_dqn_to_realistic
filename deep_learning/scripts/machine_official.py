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

import h5py
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation
import os
import keras
from keras.models import Model,load_model
from keras import layers
from keras import models
###

EPISODES = 200

class ReinforceAgent():
    def __init__(self, state_size, action_size):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = env_path+'/model/'
        self.result = Float32MultiArray()
        self.load_model = False
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
        self.memory = deque(maxlen=1000000)
        self.model = self.buildModel()
        self.target_model = self.buildModel()
        self.updateTargetModel()
        print('prepare_to_load_weight')
        if self.load_model:
            self.model.load_weights((self.dirPath +"model_tmp.h5"), by_name = True)
	    print("successful load modef",(self.dirPath+str(self.load_episode)+"model_tmp.h5"))
            with open(self.dirPath+str(self.load_episode)+'model_tmp.json') as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')

        self.model.summary()
        # print('self.model.summary()')
        
        ###################################

        for i,layer in enumerate(self.model.layers):
            if i==5 or i==7 :
                layer.trainable=True
            else:
                layer.trainable=False
        for i,layer in enumerate(self.model.layers):
            if layer.trainable==True:
                print(i,layer.name,'Ture')
            else:
                print(i,layer.name,'False')
        ###################################
        
    def buildModel(self):
        model = Sequential()
        dropout = 0.2
        '''
        model.add(Dense(64, input_shape=(self.state_size,), activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dense(64, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dropout(dropout))
        model.add(Dense(self.action_size, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06))
        model.summary()
        '''
        model.add(Dense(64, input_shape=(26,),activation='relu',name="dense_1", kernel_initializer='lecun_uniform'))
        model.add(Dense(64, activation='relu',name="dense_2", kernel_initializer='lecun_uniform'))
        model.add(Dropout(dropout,name="dropout_1"))
        model.add(Dense(5,name="dense_3",kernel_initializer='lecun_uniform'))
        '''##############
        x=Dense(64,input_shape=(26,),activation='relu',name="new_dense_one",kernel_initializer='lecun_uniform')(model.layers[0].input) 
        x=Dense(64,activation='relu',name="new_dense_two",kernel_initializer='lecun_uniform')(x) 
        x=Dropout(0.2,name="new_dropout")(x) 
        x=Dense(5,activation='relu',name="new_dense_three",kernel_initializer='lecun_uniform')(x) 
        added = keras.layers.add([model.output, x],name="layers_combine")
        preds=Dense(5,activation='linear',name="activation")(added)
        model=Model(inputs=model.input,outputs=preds)
        '''##############
        x=Dense(5,activation='relu',name="new_dense_three",kernel_initializer='lecun_uniform')(model.layers[3].input) 
        added = keras.layers.add([model.output, x],name="layers_combine")
        preds=Dense(5,activation='linear',name="activation")(added)
        model=Model(inputs=model.input,outputs=preds)
        ##############
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06))
        model.summary()
        
        return model

    # print('haha1')
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

        self.model.fit(X_batch, Y_batch, batch_size=self.batch_size, epochs=1, verbose=0)

if __name__ == '__main__':
    rospy.init_node('turtlebot3_dqn_stage_5')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()

    state_size = 26
    action_size = 5

    env = Env(action_size)
    # print('super mario!!!!!!!!!!!!!!')
    agent = ReinforceAgent(state_size, action_size)
    scores, episodes = [], []
    global_step = 0
    start_time = time.time()
    # print('!!!!!!!!!!!!!!!!!!muton!!!!!!!!!!!!!!!!!!!!')
    for e in range(agent.load_episode + 1, EPISODES):
        done = False
        state = env.reset()
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

            if e % 5 == 0:
                agent.model.save(agent.dirPath + str(e) + '_model_tmp.h5')
                with open(agent.dirPath + str(e) + '_model_tmp.json', 'w') as outfile:
                    json.dump(param_dictionary, outfile)

            if t >= time_out_step:
                rospy.loginfo("Time out!!")
                done = True

            if done:
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

        if e == 100:
            env.ramdom_target = True

        if e == 150:
            env.ramdom_bot = True

        if e == 175:
            env.ramdom_bot_rotate = True


            global_step += 1
            if global_step % agent.target_update == 0:
                rospy.loginfo("UPDATE TARGET NETWORK")

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

