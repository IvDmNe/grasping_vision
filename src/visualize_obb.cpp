#include <Eigen/Dense>
#include <boost/foreach.hpp>
#include <chrono>

// ROS
#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include "sensor_msgs/PointCloud2.h"
#include <message_filters/time_synchronizer.h>
#include <message_filters/subscriber.h>




// PCL
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/kdtree/impl/kdtree_flann.hpp>

typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;

pcl::visualization::PCLVisualizer::Ptr simpleVis(void) {
	// --------------------------------------------
	// -----Open 3D viewer and add point cloud-----
	// --------------------------------------------
	pcl::visualization::PCLVisualizer::Ptr viewer(
			new pcl::visualization::PCLVisualizer("3D Viewer"));

	viewer->setBackgroundColor(255, 255, 255);
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	viewer->setCameraPosition(0.0, 0.0, 0.0, 0.0, -1.0, 0.0);
	viewer->removeCoordinateSystem();
	return (viewer);
}

pcl::visualization::PCLVisualizer::Ptr viewer = simpleVis();

class vis_bbox {
	ros::Subscriber pc_sub;
	ros::Subscriber bbox_sub;
	PointCloud stored_pc;
	std_msgs::Float32MultiArray bbox_array;
	bool receive_pc_msg = false;
	bool receive_bbox_msg = false;


 public:
	vis_bbox(ros::NodeHandle* nh) {
		pc_sub = nh->subscribe<PointCloud>("/camera/depth/color/points", 1, &vis_bbox::pc_callback, this);
		bbox_sub = nh->subscribe<std_msgs::Float32MultiArray>("/obb_array", 1, &vis_bbox::bbox_callback, this);


		// message_filters::Subscriber<PointCloud> pc_sub(*nh, "/camera/depth/color/points", 1);
		// message_filters::Subscriber<std_msgs::Float32MultiArray> bbox_sub(*nh, "/obb_array", 1);

		// message_filters::TimeSynchronizer<PointCloud, std_msgs::Float32MultiArray> sync(pc_sub, bbox_sub, 10);

		printf("plane segmentation node: Init completed!");
	}

	void pc_callback(const PointCloud::ConstPtr& msg) {
		pcl::copyPointCloud(*msg, stored_pc);
		receive_pc_msg = true;
	}

	void bbox_callback(const std_msgs::Float32MultiArray::ConstPtr& array ) {
		bbox_array = *array;
		receive_bbox_msg = true;

	}

	void loop() {
		if (!receive_pc_msg && !receive_bbox_msg) {
			std::cout << "No data received" << std::endl;
			return;
		}


		std::cout << bbox_array << std::endl;
	}
};

int main(int argc, char** argv) {
	ros::init(argc, argv, "vis_bbox");
	ros::NodeHandle nh;
	vis_bbox node(&nh);

	ros::Rate loop_rate(50);  // 50 Hz
	while (nh.ok()) {
		node.loop();
		ros::spinOnce();
		loop_rate.sleep();
	}
}
