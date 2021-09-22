#!/usr/bin/python3


# this project packages
from torchvision.transforms.functional import perspective
from models.segmentation_net import seg
from models.feature_extractor import dino_wrapper, image_embedder
from models.mlp import MLP
from models.knn import knn_torch
from utilities.utils import find_nearest, find_nearest_to_center_cntr, get_centers, \
    get_one_mask, get_nearest_to_center_box, get_padded_image, removeOutliers

# ROS
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import message_filters
from cv_bridge import CvBridge

# PyTorch
import torch
from torch.nn import functional as F

# Detectron 2
# from detectron2.utils.logger import setup_logger
from detectron2.utils.colormap import colormap

# misc
import threading
import cv2 as cv
import time
import numpy as np
import re
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.spatial.distance import euclidean

torch.set_grad_enabled(False)

# setup_logger()


lock = threading.Lock()
freq = 100
conf_thresh = 0.8
min_dist_thresh = 0.2


class ImageListener:

    def __init__(self):

        self.im = None
        rgb_frame_id = None
        rgb_frame_stamp = None

        self.working_mode = 'inference'
        self.depth_encoding = None
        self.prev_training_mask_info = None
        self.x_data_to_save = None
        self.prev_mode = 'inference'

        self.cv_bridge = CvBridge()

        # initialize a node
        rospy.init_node("segmenting", log_level=rospy.INFO)
        self.segmented_view_pub = rospy.Publisher(
            '/segmented_view', Image, queue_size=10)

        # self.base_frame = 'measured/base_link'
        # self.camera_frame = 'measured/camera_color_optical_frame'
        # self.target_frame = self.base_frame

        rgb_sub = rospy.Subscriber('/camera/color/image_raw',
                                   Image, self.callback_rgb)

        self.seg_net = seg()

        emb_size = 64

        self.colors = colormap()

        rospy.loginfo('Segmentaion node: Init complete')

    def callback_rgb(self, rgb):

        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')

        self.im = im.copy()
        self.rgb_frame_id = rgb.header.frame_id
        self.rgb_frame_stamp = rgb.header.stamp

    def do_segmentation(self, image):

        instances = self.seg_net.forward(
            image)

        if len(instances.pred_boxes.tensor) == 0:
            rospy.logerr_throttle(2, 'no objects found')
            return

        final_masks = []

        for box, mask in zip(instances.pred_boxes, instances.pred_masks):
            # get masked images
            x1, y1, x2, y2 = box.round().long()
            sz = (int(x2 - x1), int(y2 - y1))

            mask_rs = cv.resize(
                mask.squeeze().detach().cpu().numpy(), sz)

            cur_mask = np.zeros((image.shape[:-1]), dtype=np.uint8)
            cur_mask[y1:y2, x1:x2] = (mask_rs + 0.5).astype(int)

            image_masked = cv.bitwise_and(image, image, mask=cur_mask)

            final_mask = image_masked[y1:y2, x1:x2]

            final_mask_sq = get_padded_image(final_mask)

            final_masks.append(final_mask_sq)

        return final_masks, instances.pred_boxes.tensor, instances.pred_masks

    def send_image_to_topics(self, image_segmented=None):

        if image_segmented is not None:
            mask_msg = self.cv_bridge.cv2_to_imgmsg(image_segmented)
            mask_msg.header.stamp = rospy.Time.now()
            mask_msg.encoding = 'bgr8'
            self.segmented_view_pub.publish(mask_msg)

    def run_proc(self):

        start = time.time()

        with lock:
            if self.im is None:
                rospy.logerr_throttle(5, "No image received")
                return
            image = self.im.copy()

        # segment rgb image
        image = cv.resize(image, (640, 480))

        image_segmented = image.copy()
        ret_seg = self.do_segmentation(image)
        if ret_seg:
            _, boxes, pred_masks = ret_seg

            # draw contours of found objects
            for box, m in zip(boxes, pred_masks):
                c = self.colors[0].astype(np.uint8).tolist()

                # draw bounding box
                pts = box.detach().cpu().long()
                cv.rectangle(image_segmented, (int(pts[0]), int(pts[1])),
                             (int(pts[2]), int(pts[3])), c, 2)

                # draw object masks
                x1, y1, x2, y2 = box.round().long()
                sz = (int(x2 - x1), int(y2 - y1))

                mask_rs = cv.resize(
                    m.squeeze().detach().cpu().numpy(), sz)

                cur_mask = np.zeros((image.shape[:-1]), dtype=np.uint8)
                cur_mask[y1:y2, x1:x2] = (mask_rs + 0.5).astype(int)
                cntrs, _ = cv.findContours(
                    cur_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                cv.drawContours(image_segmented, cntrs, -
                                1, c, 2)

        self.send_image_to_topics(
            image_segmented)

        end = time.time()
        fps = 1 / (end - start)
        rospy.logwarn_throttle(1, f'FPS: {fps:.2f}')


if __name__ == '__main__':

    listener = ImageListener()
    rate = rospy.Rate(freq)
    try:
        while not rospy.is_shutdown():
            listener.run_proc()
            rate.sleep()
    except:
        pass
    finally:
        cv.destroyAllWindows()
