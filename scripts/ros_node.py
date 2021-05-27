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

from cv_bridge import CvBridge, CvBridgeError
from matplotlib import pyplot as plt

import torch
from torch.nn import functional as F

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.utils.colormap import random_color


setup_logger()


lock = threading.Lock()
freq = 100


# def filter_outputs(outputs):
#     all_area = outputs['instances'].pred_masks[0].shape[0] * \
#         outputs['instances'].pred_masks[0].shape[1]

#     idxs_to_left = []
#     for idx, mask in enumerate(outputs['instances'].pred_masks):
#         if float(torch.sum(mask) / all_area) < 0.4:
#             idxs_to_left.append(idx)

#     res_out = detectron2.structures.Instances((720, 1080))
#     res_out.set('pred_masks', outputs['instances'].pred_masks[idxs_to_left])
#     res_out = {'instances': res_out}
#     return res_out


class ImageListener:

    def __init__(self):

        self.im = None
        self.depth = None
        rgb_frame_id = None
        rgb_frame_stamp = None
        self.working_mode = 'inference'
        self.depth_encoding = None

        self.cv_bridge = CvBridge()

        # initialize a node
        rospy.init_node("sod")
        self.rgb_pub = rospy.Publisher('/rgb_masked', Image, queue_size=10)
        self.depth_pub = rospy.Publisher('/depth_masked', Image, queue_size=10)

        self.base_frame = 'measured/base_link'
        self.camera_frame = 'measured/camera_color_optical_frame'
        self.target_frame = self.base_frame

        rgb_sub = message_filters.Subscriber('/camera/color/image_raw',
                                             Image, queue_size=10)

        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw',
                                               Image, queue_size=10)

        rospy.Subscriber('/command_cl/mode',
                         String, self.callback_mode)

        cv.namedWindow('main', cv.WINDOW_GUI_EXPANDED)

        self.seg_net = seg()
        self.classifier = knn_torch()

        ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], 1, 0.1)
        ts.registerCallback(self.callback_rgbd)

        print('init complete')

    def callback_rgb(self, rgb):

        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'rgb8')
        with lock:
            self.im = im.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp

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
        self.working_mode = mode

    def run_proc(self):

        start = time.time()

        with lock:
            if self.im is None:
                return
            image = self.im.copy()
            depth = self.depth.copy()
            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp

        # print(image.shape)
        # image = cv.resize(image, (640, 480))
        # depth = cv.resize(depth, (640, 480))

        # cv.imshow('depth', depth)
        depth[depth > 2.0] = 2.0
        # depth *= 127.5
        # depth = depth.astype(np.uint8)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        proposal_boxes, seg_mask_features, instances = self.seg_net.forward(
            image)

        masked_image = image.copy()
        for box in proposal_boxes:

            pts = box.detach().cpu().long()
            cv.rectangle(image, (pts[0], pts[1]),
                         (pts[2], pts[3]), (255, 0, 0))

        # draw object masks
        for box, mask in zip(instances.pred_boxes, instances.pred_masks):
            x1, y1, x2, y2 = box.round().long()
            sz = (x2 - x1, y2 - y1)

            mask_rs = cv.resize(mask.squeeze().detach().cpu().numpy(), sz)

            cur_mask = np.zeros((image.shape[:-1]))
            cur_mask[y1:y2, x1:x2] = mask_rs
            # image[cur_mask > 0.5] = random_color()
            image[cur_mask > 0.5] = (0, 255, 255)

        cv.imshow('main', image)
        key = cv.waitKey(1)

        if key == ord('q') & 0xFF:
            cv.destroyAllWindows()
            exit()

        if len(seg_mask_features) == 0 or len(instances.pred_boxes.tensor) == 0:
            print('no found objects')
            return

        features = F.max_pool2d(
            seg_mask_features, kernel_size=seg_mask_features.size()[2:])

        # if mode is inference
        if self.working_mode != 'train':
            classes = self.classifier.classify(features)
        else:
            # choose only the nearest bbox to the center
            classes = [self.working_mode.split(
                ' ')[1] for _ in features.shape[0]]
            self.classifier.add_points(features, classes)

        mask = get_one_mask(
            instances.pred_boxes, instances.pred_masks, image).astype(np.uint8)

        # print(mask)
        # mask = mask.astype(np.uint8)

        # plt.imshow(mask)
        # plt.show()
        # label = seg_mask_features.detach().cpu().numpy()

        # draw one chosen mask
        # cntrs, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        # print(len(cntrs))
        # if len(cntrs) > 0:

        # cv.drawContours(image, cntrs, -1, (255, 0, 0))
        # cv.drawContours(depth, cntrs, -1, (255, 0, 0))

        # image[mask] = (0, 0, 0)

        # mask = cv.bitwise_not(mask.astype(bool))

        image_masked = cv.bitwise_and(masked_image, masked_image, mask=mask)
        depth_masked = cv.bitwise_and(depth, depth, mask=mask)

        # depth[mask] = 0

        # plt.imshow(mask)
        # plt.show()
        # cv.imshow('mask', mask)
        # cv.imshow('1', image_masked)
        # cv.imshow('2', depth_masked)
        # cv.waitKey(1)

        # image_masked = image
        # depth_masked = depth

        rgb_msg = self.cv_bridge.cv2_to_imgmsg(image_masked)
        rgb_msg.header.stamp = rgb_frame_stamp
        rgb_msg.header.frame_id = rgb_frame_id
        rgb_msg.encoding = 'bgr8'
        self.rgb_pub.publish(rgb_msg)

        depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_masked)
        depth_msg.header.stamp = rgb_frame_stamp
        depth_msg.header.frame_id = rgb_frame_id
        depth_msg.encoding = '32FC1'
        # depth_msg.encoding = self.depth_encoding
        self.depth_pub.publish(depth_msg)

        end = time.time()
        fps = 1 / (end - start)
        print(f'FPS: {fps:.2f}')


def get_one_mask(box, mask, image):
    # for box, mask in zip(instances.pred_boxes, instances.pred_masks):

    x1, y1, x2, y2 = box.tensor[0].round().long()
    sz = (x2 - x1, y2 - y1)
    # print(mask.shape)
    mask_rs = cv.resize(mask[0].squeeze().detach().cpu().numpy(), sz)

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
