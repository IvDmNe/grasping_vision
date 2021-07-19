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
	viewer->addCoordinateSystem(10.0);
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

		// std::cout << "a" << std::endl;

		viewer->spinOnce(50);
		viewer->removeAllPointClouds();
		viewer->removeAllShapes();

		if (!receive_pc_msg || !receive_bbox_msg) {
			// std::cout << "No data received" << std::endl;
			return;
		}

		PointCloud::Ptr cur_cloud (new PointCloud);
		pcl::copyPointCloud(stored_pc, *cur_cloud);

		std_msgs::Float32MultiArray cur_array;
		cur_array = bbox_array;

		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cur_cloud);
        viewer->addPointCloud<pcl::PointXYZRGB> (cur_cloud, rgb, "sample cloud");  

		Eigen::Vector3f mass_center(cur_array.data[6], cur_array.data[7], cur_array.data[8]);
		Eigen::Vector3f dims(cur_array.data[9], cur_array.data[10], cur_array.data[11]);
		Eigen::Vector3f major_vector(cur_array.data[0], cur_array.data[1], cur_array.data[2]);
		Eigen::Vector3f middle_vector(cur_array.data[3], cur_array.data[4], cur_array.data[5]);

		Eigen::Vector3f minor_vector = major_vector.cross(middle_vector);

		Eigen::Matrix3f rot_matrix;
		rot_matrix.col(0) = major_vector;
		rot_matrix.col(1) = middle_vector;
		rot_matrix.col(2) = minor_vector;
	
		Eigen::Quaternionf quat(rot_matrix);
	


		pcl::PointXYZ o;
		o.x = mass_center[0];
		o.y = mass_center[1];
		o.z = mass_center[2];
		viewer->addSphere (o, 0.03, "sphere", 0);



		// quat.FromTwoVectors(major_vector, middle_vector);

		// std::cout << quat << std::endl;

		int counter = 1;
		pcl::PointXYZRGB center(mass_center(0), mass_center(1), mass_center(2));
		pcl::PointXYZRGB x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
		pcl::PointXYZRGB y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
		pcl::PointXYZRGB z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));
		
		viewer->addCube(mass_center, quat, dims(0), dims(1), dims(2), "OBB_" + std::to_string(counter));
		viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "OBB_" + std::to_string(counter));
		
		viewer->addLine(center, x_axis, 1.0f, 0.0f, 0.0f, "major eigen vector " + std::to_string(counter));//main ingredient
		viewer->addLine(center, y_axis, 0.0f, 1.0f, 0.0f, "middle eigen vector " + std::to_string(counter));
		viewer->addLine(center, z_axis, 0.0f, 0.0f, 1.0f, "minor eigen vector " + std::to_string(counter));


		// std::cout << cur_cloud << std::endl;
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
