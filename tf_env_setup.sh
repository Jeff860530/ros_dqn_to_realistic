virtualenv --system-site-packages -p python2.7 ~/tfp27

source ~/tfp27/bin/activate

pip install -U rosinstall msgpack empy defusedxml netifaces

pip install tensorflow==2.0.0

pip install tensorflow-tensorboard

pip install keras==2.1.5
