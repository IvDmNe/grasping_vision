# Description
This repo contains ROS nodes for processing RGB-D data from Intel realsense d435.
It has several steps:
1. segment rgb image and get mask of one object using Mask R-CNN for segmentation and KNN for classification with iterative learning
2. After applying mask on rgb and depth image get point cloud of an object
3. Find Oriented bounding box of point cloud (with outlier removal)
4. Find coefficients of a plane on point cloud of complete scene

## Dependencies
ROS melodic

For python:
Install conda and activate env (for example:``` conda create -n cv python=3.8.5```
```
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c conda-forge opencv detectron2 tqdm pandas scipy ros-rospy

pip3 install pytimedinput
```

Additionally, install opencv_bridge for ROS (i.e. from https://cyaninfinite.com/ros-cv-bridge-with-python-3/)

For C++:
-PCL
-Eigen


## Run 
```
1. roslaunch launch/launch_them_all.launch
2. rosrun scripts/command_node.py (in another terminal)
```
