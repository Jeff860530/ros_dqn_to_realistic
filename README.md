# ros_dqn_to_realistic

Purpose:We want to make a small car using reinforcement-learning and move from simulation to the real world.We first make the bot to pretrain model and then test pretrained model by bot2 with different weights and torques.

step1:

training robot with idel car (name:bot) [in simulator]
for 200 epochs

step2:

training robot with different car's performance (less wheel torque, more car mass) (name:bot2) [in simulator]
for 60 epochs

step3:
training robot in real world


### Training the ideal robot in simulator
```shell= 
roslaunch bot gazebo_field.launch

rosrun deep_learning laser_filter.py

rosrun deep_learning machine_official.py
```
### Transfer learning using pretrain model in differenter characteristic robot in simulator
```shell= 
roslaunch bot2 gazebo_field.launch

rosrun deep_learning laser_filter.py

rosrun deep_learning machine_official_t.py
```
