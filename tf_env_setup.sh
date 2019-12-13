virtualenv --system-site-packages -p python2.7 ~/tfp27

source ~/tfp27/bin/activate

pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp27-none-linux_x86_64.whl

pip install keras==2.1.5
