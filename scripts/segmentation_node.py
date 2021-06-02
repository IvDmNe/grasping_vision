#!/home/ivan/anaconda3/bin/python

from models.segmentation_net import *
from models.knn import *

import rosnode
import message_filters
import cv2 as cv
import time
import os
import sys
import os.path as osp
import numpy as np
import rospy
import threading

from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge
# from matplotlib import pyplot as plt

# import torch
from torch.nn import functional as F

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.utils.colormap import random_color

from scipy.spatial.distance import euclidean

setup_logger()


lock = threading.Lock()
freq = 100


class ImageListener:

    def __init__(self):

        self.im = None
        self.depth = None
        rgb_frame_id = None
        rgb_frame_stamp = None
        self.working_mode = 'inference'
        self.depth_encoding = None
        self.prev_training_mask_info = None

        self.cv_bridge = CvBridge()

        # initialize a node
        rospy.init_node("sod", log_level=rospy.INFO)
        self.rgb_pub = rospy.Publisher('/rgb_masked', Image, queue_size=10)
        self.depth_pub = rospy.Publisher('/depth_masked', Image, queue_size=10)
        self.segmented_view_pub = rospy.Publisher(
            '/segmented_view', Image, queue_size=10)

        self.base_frame = 'measured/base_link'
        self.camera_frame = 'measured/camera_color_optical_frame'
        self.target_frame = self.base_frame

        rgb_sub = message_filters.Subscriber('/camera/color/image_raw',
                                             Image, queue_size=10)

        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw',
                                               Image, queue_size=10)

        rospy.Subscriber('/command_from_human',
                         String, self.callback_mode)

        self.seg_net = seg()
        self.classifier = knn_torch(datafile='knn_data.pth')

        ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], 1, 0.1)
        ts.registerCallback(self.callback_rgbd)

        rospy.loginfo('init complete')

    def callback_rgbd(self, rgb, depth):
        self.depth_encoding = depth.encoding
        if depth.encoding == '32FC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(
                depth).copy().astype(np.float32)
            depth_cv /= 1000.0
            # print('16UC1')
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')

        with lock:
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp

    def callback_mode(self, mode):
        self.working_mode = mode.data

    def do_segmentation(self, image):

        proposal_boxes, seg_mask_features, instances = self.seg_net.forward(
            image)

        masked_image = image.copy()

        for box in proposal_boxes:

            pts = box.detach().cpu().long()
            cv.rectangle(masked_image, (pts[0], pts[1]),
                         (pts[2], pts[3]), (255, 0, 0), 2)

        # draw object masks
        for box, mask in zip(instances.pred_boxes, instances.pred_masks):
            x1, y1, x2, y2 = box.round().long()
            sz = (x2 - x1, y2 - y1)

            mask_rs = cv.resize(mask.squeeze().detach().cpu().numpy(), sz)

            cur_mask = np.zeros((image.shape[:-1]))
            cur_mask[y1:y2, x1:x2] = mask_rs
            masked_image[cur_mask > 0.5] = (0, 255, 255)

        if len(seg_mask_features) == 0 or len(instances.pred_boxes.tensor) == 0:
            rospy.logerr_throttle(2, 'no objects found')
            return

        # decrease dimension of features by global pooling
        # print(seg_mask_features.shape)
        features = F.avg_pool2d(
            seg_mask_features, kernel_size=seg_mask_features.size()[2:])

        mask = get_one_mask(
            instances.pred_boxes.tensor.round().long().detach().cpu().numpy(), instances.pred_masks, image).astype(np.uint8)

        return features, proposal_boxes.tensor, masked_image, mask

    def send_images_to_topics(self, image_masked, depth_masked, image_segmented):
        rgb_msg = self.cv_bridge.cv2_to_imgmsg(image_masked)
        rgb_msg.header.stamp = rospy.Time.now()
        # rgb_msg.header.frame_id = rgb_frame_id
        rgb_msg.encoding = 'bgr8'
        self.rgb_pub.publish(rgb_msg)

        depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_masked)
        depth_msg.header.stamp = rospy.Time.now()
        # depth_msg.header.frame_id = rgb_frame_id
        depth_msg.encoding = '32FC1'
        self.depth_pub.publish(depth_msg)

        mask_msg = self.cv_bridge.cv2_to_imgmsg(image_segmented)
        mask_msg.header.stamp = rospy.Time.now()
        # mask_msg.header.frame_id = rgb_frame_id
        mask_msg.encoding = 'bgr8'
        self.segmented_view_pub.publish(mask_msg)

    def save_data(self, features, cl, im_shape, boxes):

        center_idx = get_nearest_to_center_box(
            im_shape, boxes.cpu().numpy())
        # classes = [self.working_mode.split(
        #     ' ')[1] for _ in range(features.shape[0]]

        # check if bounding box is very different from previous
        if self.check_sim(boxes[center_idx].cpu().numpy()):
            self.classifier.add_points(
                features[center_idx].squeeze().unsqueeze(0), [cl])

    def run_proc(self):

        start = time.time()

        with lock:
            if self.im is None:
                rospy.logerr_throttle(5, "No image received")
                return
            image = self.im.copy()
            depth = self.depth.copy()
            # rgb_frame_id = self.rgb_frame_id
            # rgb_frame_stamp = self.rgb_frame_stamp

        # segment rgb image
        image = cv.resize(image, (640, 480))
        depth = cv.resize(depth, (640, 480))
        # cv.imshow('im', image)
        # cv.waitKey()
        ret_seg = self.do_segmentation(image)
        if ret_seg:
            features, boxes, image_segmented, mask = ret_seg
        else:
            return

        # filter depth by 1 meter
        depth[depth > 1.0] = 1.0

        # choose action according to working mode
        if self.working_mode.split(' ')[0] == 'train':
            # choose only the nearest bbox to the center
            cl = self.working_mode.split(
                ' ')[1]
            self.save_data(features, cl, image.shape, boxes)
            # return
        elif self.working_mode != 'inference':
            rospy.logerr_throttle(
                1, f'invalid working mode: {self.working_mode}')
            return
        else:
            classes = self.classifier.classify(features.squeeze())

            if isinstance(classes, str):
                classes = [classes]
            if classes:
                # draw labels
                for cl, box in zip(classes, boxes):

                    pt = (box[:2].round().long()) - 1
                    cv.putText(image_segmented, cl, tuple(pt),
                               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # NOT IMPLEMENTED
        # classify each mask and draw labels
        # currently one random mask is chosen

        # apply masking
        image_masked = cv.bitwise_and(image, image, mask=mask)
        depth_masked = cv.bitwise_and(depth, depth, mask=mask)

        self.send_images_to_topics(image_masked, depth_masked, image_segmented)

        end = time.time()
        fps = 1 / (end - start)
        rospy.loginfo(f'FPS: {fps:.2f}')

    def check_sim(self, box):
        box_center = ((box[3] + box[1]) // 2, (box[2] + box[0]) // 2)

        if not self.prev_training_mask_info:
            self.prev_training_mask_info = box_center
            return True
        # print(euclidean(box_center, self.prev_training_mask_info))
        if euclidean(box_center, self.prev_training_mask_info) > 80:
            rospy.logwarn(
                f"skipping image, too far from previous: {euclidean(box_center, self.prev_training_mask_info):.2f} (threshold: {80})")
            # print(euclidean(box_center, self.prev_training_mask_info))
            return False
        else:
            self.prev_training_mask_info = box_center
            return True


def get_nearest_to_center_box(im_shape, boxes):
    center = np.array(im_shape[:-1]) // 2
    min_dist = 1000000  # just a big number
    min_idx = -1
    for idx, box in enumerate(boxes):
        box_center = ((box[3] + box[1]) // 2, (box[2] + box[0]) // 2)
        dist = euclidean(box_center, center)
        if dist < min_dist:
            min_dist = dist
            min_idx = idx

    return min_idx


def get_one_mask(boxes, mask, image):
    cent_ix = get_nearest_to_center_box(image.shape, boxes)
    x1, y1, x2, y2 = boxes[cent_ix]
    sz = (x2 - x1, y2 - y1)
    mask_rs = cv.resize(mask[cent_ix].squeeze().detach().cpu().numpy(), sz)

    cur_mask = np.zeros((image.shape[: -1]))
    cur_mask[y1: y2, x1: x2] = mask_rs
    return cv.threshold(cur_mask, 0.5, 1.0, cv.THRESH_BINARY)[1]


if __name__ == '__main__':

    listener = ImageListener()
    rate = rospy.Rate(freq)
    try:
        while not rospy.is_shutdown():
            listener.run_proc()
            rate.sleep()
    finally:
        cv.destroyAllWindows()
