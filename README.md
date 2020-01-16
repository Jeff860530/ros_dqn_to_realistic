# ros_dqn_to_realistic

Purpose:We want to make a small car using reinforcement-learning and move from simulation to the real world.We first make the bot to pretrain model and then test pretrained model by bot2 with different weights and torques.

step1:

training robot with idel car (name:bot) [in simulator]
for 200 epochs

<img src="https://github.com/tony92151/ros_dqn_to_realistic/blob/master/image/step1.gif"/>

step2:

training robot with different car's performance (less wheel torque, more car mass) (name:bot2) [in simulator]
for 60 epochs

step3:
training robot in real world

<img src="https://github.com/tony92151/ros_dqn_to_realistic/blob/master/image/step3.gif"/>

# How to use

## Clone repositories

1.Clone this repository

> cd catkin_ws/src

> git clone https://github.com/tony92151/ros_dqn_to_realistic.git

2.Clone repository for Rplidar-A1

> git clone https://github.com/robopeak/rplidar_ros.git

3.Clone repository for rf2o  (not official sdk)

> git clonehttps://github.com/artivis/rf2o_laser_odometry.git

4.instsll dependent

> sudo apt-get install remmina synaptic gimp git ros-kinetic-navigation ros-kinetic-amcl ros-kinetic-slam-gmapping ros-kinetic-mrpt-slam ros-kinetic-mrpt-icp-slam-2d ros-kinetic-robot-localization ros-kinetic-ar-track-alvar -y 

6.Compile all

> cd ..

> catkin_make


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

### Transfer learning in realistic

```shell= 
roslaunch deep_learning ml_real.launch

roslaunch deep_learning laser.launch real_scan:=true

roslaunch deep_learning rf2o.launch

roslaunch deep_learning amcl.launch

rosrun deep_learning show_markers.py

rosrun deep_learning base.py

rosrun deep_learning real_train.py
```
