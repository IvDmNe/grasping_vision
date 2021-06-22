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
pip install opencv-python tqdm pandas scipy rospy

python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
pip3 install pytimedinput
```

```pip install sklearn``` for visualization

Additionally, install opencv_bridge for ROS (i.e. from https://cyaninfinite.com/ros-cv-bridge-with-python-3/)

For C++:
* PCL 
* Eigen


## Run 
```
1. roslaunch launch/launch_them_all.launch
2. rosrun scripts/command_node.py (in another terminal)
```

## Usage

In the command_node user can enter one of the following commands:
  * inference (default)
  * train {name of object}
  * give {name of object}
  
1. In inference mode the segmentation node segments image and classify each object. One random bounding box is outputed
2. In train mode the node stores all images of an object for 30 seconds and then feed deep features of them into KNN-classifier
3. In give mode the image is segmented and the coordinates of bouding box of a desired object are sent to the topic ```/obb_array```
