#include <Eigen/Dense>
#include <boost/foreach.hpp>
#include <chrono>

// ROS
#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>

#include "sensor_msgs/PointCloud2.h"

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

// pcl::visualization::PCLVisualizer::Ptr simpleVis(void) {
// 	// --------------------------------------------
// 	// -----Open 3D viewer and add point cloud-----
// 	// --------------------------------------------
// 	pcl::visualization::PCLVisualizer::Ptr viewer(
// 			new pcl::visualization::PCLVisualizer("3D Viewer"));

// 	viewer->setBackgroundColor(255, 255, 255);
// 	viewer->addCoordinateSystem(1.0);
// 	viewer->initCameraParameters();

// 	viewer->setCameraPosition(0.0, 0.0, 0.0, 0.0, -1.0, 0.0);
// 	viewer->removeCoordinateSystem();
// 	return (viewer);
// }

// pcl::visualization::PCLVisualizer::Ptr viewer = simpleVis();

class segmentPC {
	ros::Publisher pub;
	ros::Subscriber sub;
	PointCloud stored_pc;
	bool receive_msg = false;


 public:
	segmentPC(ros::NodeHandle* nh) {
		sub = nh->subscribe<PointCloud>("/camera/depth/color/points", 1,
										&segmentPC::sub_callback, this);

		pub = nh->advertise<std_msgs::Float32MultiArray>("/plane_coeffs", 1);

		printf("plane segmentation node: Init completed!");
	}

	void sub_callback(const PointCloud::ConstPtr& msg) {
		pcl::copyPointCloud(*msg, stored_pc);
		receive_msg = true;
	}

	void loop() {
		if (!receive_msg) {
			// std::cout << "No data received" << std::endl;
			return;
		}
		double downsample_voxel_size = 0.002;
		double distance_thresh = 1.0;
		double plane_inliers_thresh = 0.02;
		auto start = std::chrono::high_resolution_clock::now();

		// viewer->spinOnce(50);
		// viewer->removeAllPointClouds();
		// viewer->removeAllShapes();

		PointCloud::Ptr cur_cloud(new PointCloud);
		pcl::copyPointCloud(stored_pc, *cur_cloud);

		// downsample point cloud
		pcl::VoxelGrid<pcl::PointXYZRGB> sor;
		sor.setInputCloud(cur_cloud);
		double leafsize = downsample_voxel_size;
		sor.setLeafSize(leafsize, leafsize, leafsize);
		sor.filter(*cur_cloud);

		// filter by depth
		pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;

		kdtree.setInputCloud(cur_cloud);

		pcl::PointXYZRGB searchPoint;
		searchPoint.x = 0.0d;
		searchPoint.y = 0.0d;
		searchPoint.z = 0.0d;

		std::vector<int> near_indices_int;
		std::vector<float> dists;
		kdtree.radiusSearch(searchPoint, distance_thresh, near_indices_int, dists);

		pcl::PointIndices::Ptr near_indices(new pcl::PointIndices);
		near_indices->indices = near_indices_int;

		pcl::ExtractIndices<pcl::PointXYZRGB> extract_far;
		PointCloud::Ptr cloud_near(new PointCloud);
		extract_far.setInputCloud(cur_cloud);
		extract_far.setIndices(near_indices);
		extract_far.setNegative(false);
		extract_far.filter(*cloud_near);

		if (cloud_near->points.size() == 0) {
			std::cout << "No near points found" << std::endl;
			return;
		}

		// get plane coefficients
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
		// Create the segmentation object
		pcl::SACSegmentation<pcl::PointXYZRGB> seg;
		// Optional
		seg.setOptimizeCoefficients(true);
		// Mandatory
		seg.setModelType(pcl::SACMODEL_PLANE);
		seg.setMethodType(pcl::SAC_RANSAC);

		seg.setDistanceThreshold(plane_inliers_thresh);

		seg.setInputCloud(cloud_near);
		seg.segment(*inliers, *coefficients);

		// std::cerr << "Model coefficients: " << coefficients->values[0] << " "
		// 					<< coefficients->values[1] << " " << coefficients->values[2]
		// 					<< " " << coefficients->values[3] << std::endl;

		
		if (inliers->indices.size() == 0) {
			PCL_ERROR("Could not estimate a planar model for the given data");
			return;
		}

		std_msgs::Float32MultiArray out_array;
		for (auto el: coefficients->values)
		{
			out_array.data.push_back(el);
		}
		pub.publish(out_array);
	}
};

int main(int argc, char** argv) {
	ros::init(argc, argv, "plane_estimation");
	ros::NodeHandle nh;
	segmentPC node(&nh);

	ros::Rate loop_rate(50);  // 50 Hz
	while (nh.ok()) {
		node.loop();
		ros::spinOnce();
		loop_rate.sleep();
	}
}
