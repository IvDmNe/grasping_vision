# Description
This repo contains ROS nodes for processing RGB-D data from Intel realsense d435.
It has several steps:
1. segment rgb image and get mask of one object using Mask R-CNN for segmentation and KNN for classification with iterative learning
2. After applying mask on rgb and depth image get point cloud of an object
3. Find Oriented bounding box of point cloud (with outlier removal)
4. Find coefficients of a plane on point cloud of complete scene

## Dependencies
RO melodic

For Python scripts run conda env create -f scripts/segm.yaml

For C++:
-PCL
-Eigen


## Run 
1. roslaunch launch/launch_them_all.launch
2. rosrun scripts/command_node.py (in another terminal)
