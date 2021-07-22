# Description
This repo contains ROS nodes for object recognition from RGB-D data using Intel realsense d435.
It has several steps:
1. segment rgb image and get mask of one object using Mask R-CNN for segmentation and KNN for classification with iterative learning
2. After applying mask on rgb and depth image get point cloud of an object
3. Find Oriented bounding box of point cloud (with outlier removal)
4. Find coefficients of a plane on point cloud of complete scene

## Docker setup

Create workspace folder, src folder in it and clone this repo into it:
```
mkdir -p ~/ros_ws/src
cd ~/ros_ws/src
git clone https://github.com/IvDmNe/grasping_vision.git
cd grasping_vision
```
Change line 7 in ```run_docker.sh``` according to your workspace location:
```    -v your_path_to_workspace:/ws \```

Build docker image

```sh build_docker.sh```
Install nvidia-container-toolkit to use GPU in docker: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

# Run
Run realsense with ROS:
```roslaunch launch rs_aligned_depth.launch```

(Opionally) Open rviz: ```rviz -d rviz_config.rviz```


Start docker

```sh run_docker.sh``` 

Prepare ros project and build it:
```
cd ws
catkin_make
source devel/setup.bash --extend
```
Launch node for segmentation and bouding box calculating:
```
cd src/grasping_vision
roslaunch launch/launch_them_all.launch
```


Open another terminal and run command_node.py:

```
sudo docker ps (to get a name of running container)
sudo docker exec -it -w /ws/src/grasping_vision/scripts name_of_container bash
python3 command_node.py
```

## Usage

In the command_node user can enter one of the following commands:
  * inference (default)
  * train {name of object}
  * give {name of object}
  
1. In inference mode the segmentation node segments image and classify each object.
2. In train mode the node stores all images of an object for 30 seconds and then feed deep features of them into KNN-classifier. 
3. In give mode the image is segmented and the coordinates of bouding box of a desired object are sent once to a topic ```/obb_array```. Once the "give" command was sent, the arraay is sent continously in topic until new "give" command is sent. 

The topic ```/obb_array``` has float32 one-dimensional array representing a pose of the bounding box in the followong format: [major_vector, middle_vector, mass_center, dimensions] (in total 12 elements). Major vector represents X-axis, middle vector - Y-axis, dimensions - size of bounding box in a coordinate system, which axes are the major vector, middle vector and a vector maden by a cross product of two first ones. 
