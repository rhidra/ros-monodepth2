ROS Monodepth2
==============

Small ROS package of the [Monodepth2](https://github.com/nianticlabs/monodepth2) depth estimation model.

## Installation

```shell script
pip2 install torch torchvision --user
cd catkin_ws/src/
git clone git@github.com:rhidra/ros-monodepth2.git
cd ros-monodepth2
python2 test_simple.py --image_path assets/test_image.jpg --model_name mono_640x192
```

You will also need CV Bridge.