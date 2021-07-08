#!/usr/bin/python3


# this project packages
from models.segmentation_net import *
from models.feature_extractor import image_embedder
from models.mlp import MLP
from models.knn import *
from utils.utils import *

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
from detectron2.utils.logger import setup_logger
from detectron2.utils.colormap import colormap

# misc
import threading
import cv2 as cv
import time
import numpy as np

torch.set_grad_enabled

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
        self.x_data_to_save = None
        self.prev_mode = 'inference'

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

        self.embedder = image_embedder('mobilenetv3_small_128_models.pth')
        self.classifier = knn_torch(
            datafile='knn_data_metric_learning.pth')

        ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], 1, 0.1)
        ts.registerCallback(self.callback_rgbd)

        self.colors = colormap()

        self.start_time = time.time()

        rospy.loginfo('Segmentaion node: Init complete')

    def callback_rgbd(self, rgb, depth):
        self.depth_encoding = depth.encoding
        if depth.encoding == '32FC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(
                depth).copy().astype(np.float32)
            depth_cv /= 1000.0
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
        self.prev_mode = self.working_mode
        self.working_mode = mode.data
        rospy.logwarn(f"changing working mode to {self.working_mode}")

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
            
            ks = 7
            kernel = np.ones((ks, ks), np.uint8)
            cur_mask = cv.erode(cur_mask, kernel)
            image_masked = cv.bitwise_and(image, image, mask=cur_mask)

            final_mask = image_masked[y1:y2, x1:x2]

            final_mask_sq = get_padded_image(final_mask)

            
            final_masks.append(final_mask_sq)

        cent_idx = get_nearest_to_center_box(image.shape, instances.pred_boxes.tensor.cpu().numpy())
        

        return final_masks[cent_idx], instances.pred_boxes.tensor, instances.pred_masks, cent_idx


    def run_proc(self):

        start = time.time()

        image_masked = None
        depth_masked = None

        with lock:
            if self.im is None:
                rospy.logerr_throttle(5, "No image received")
                return
            image = self.im.copy()
            depth = self.depth.copy()

        # segment rgb image
        image = cv.resize(image, (640, 480))
        depth = cv.resize(depth, (640, 480))

 

        image_segmented = image.copy()
        ret_seg = self.do_segmentation(image)
        if ret_seg:
            images_masked, boxes, pred_masks, cent_idx = ret_seg
        else:
            return

        # filter depth by 1 meter
        depth[depth > 1.0] = 1.0

        # with torch.no_grad():
        #     features = self.embedder(images_masked)


        # choose only the nearest bbox to the center
        
        # center_idx = get_nearest_to_center_box(image.shape, boxes)

        # box = boxes[center_idx]
        # m = pred_masks[center_idx]

        for ix, (box, m) in enumerate(zip(boxes, pred_masks)):

            c = (255, 0, 0) if ix == cent_idx else (0, 0, 255)

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
        # mask = cur_mask

        center = np.array(image.shape[:-1]) // 2

        cv.circle(image_segmented, center[::-1], 2, (0, 255, 0))

        # image_masked = cv.bitwise_and(image, image, mask=mask)
        cv.imshow('mask', images_masked)
        cv.imshow('segmented', image_segmented)
        cv.waitKey(1)

        cl = 'cleaner'
        if not os.path.exists('saved_masks'):
            os.mkdir('saved_masks')
        if not os.path.exists(f'saved_masks/{cl}'):
            os.mkdir(f'saved_masks/{cl}')

        dur = time.time() - self.start_time
        cv.imwrite(f'saved_masks/{cl}/{cl}_{dur:.3f}.png', images_masked)

        print(f'saved_masks/{cl}/{cl}_{dur:.3f}.png')


        if dur > 30.0:
            exit()

        end = time.time()
        fps = 1 / (end - start)
        rospy.logwarn(f'FPS: {fps:.2f}')

    
if __name__ == '__main__':

    listener = ImageListener()
    rate = rospy.Rate(freq)
    try:
        while not rospy.is_shutdown():
            listener.run_proc()
            rate.sleep()
    finally:
        cv.destroyAllWindows()
