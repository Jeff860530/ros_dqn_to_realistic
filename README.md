# ros_dqn_to_realistic

### Purpose:
####We want to make a small car using reinforcement-learning and move from simulation to the real world.We first make the bot to pretrain model and then test pretrained model by bot2 with different weights and torques.

#### bot & bot2 are players with different weight and torque in gazebo 

### bot have pretrained model with good performance in gazebo
```shell= 
roslaunch bot gazebo_field.launch

rosrun deep_learning machine_official.py
```
### bot2 transfer the model pretrained by bot
roslaunch bot2 gazebo_field.launch

rosrun deep_learning machine_official2.py
