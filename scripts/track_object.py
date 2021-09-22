#!/usr/bin/python3


# this project packages
from models.segmentation_net import *
from models.feature_extractor import image_embedder, dino_wrapper
from models.mlp import MLP
from models.knn import *
from utilities.utils import *

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

from utilities.sort import *

# from deep_sort.deep_sort import DeepSort
# from deep_sort.sort.iou_matching import iou
torch.set_grad_enabled

setup_logger()


lock = threading.Lock()
freq = 100

base_iou_thresh = 0.8
iou_decay = 0.05


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

        depth_sub = message_filters.Subscriber('/camera/color/image_raw',
                                               Image, queue_size=10)

        rospy.Subscriber('/command_from_human',
                         String, self.callback_mode)

        self.seg_net = seg()

        # self.embedder = image_embedder('mobilenetv3_small_128_models.pth')
        self.embedder = dino_wrapper()

        # self.classifier = knn_torch(
        #     datafile='knn_data_metric_learning.pth')

        ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], 1, 0.1)
        ts.registerCallback(self.callback_rgbd)

        self.colors = colormap()

        self.start_time = time.time()

        self.current_id = None

        self.last_box = []

        self.counter = 0
        self.total = 0

        self.iou_thresh = base_iou_thresh

        rospy.loginfo('Segmentaion node: Init complete')

    def callback_rgbd(self, rgb, depth):
        self.depth_encoding = depth.encoding

        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')

        with lock:
            self.im = im.copy()
            self.depth = im.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp

    def callback_mode(self, mode):
        self.prev_mode = self.working_mode
        self.working_mode = mode.data
        rospy.logwarn(f"changing working mode to {self.working_mode}")

    def do_segmentation(self, image):

        # cv.imshow('im', image)
        # cv.waitKey()
        # exit()
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

            # ks = 7
            # kernel = np.ones((ks, ks), np.uint8)
            # cur_mask = cv.erode(cur_mask, kernel)
            image_masked = cv.bitwise_and(image, image, mask=cur_mask)

            final_mask = image_masked[y1:y2, x1:x2]

            final_mask_sq = get_padded_image(final_mask)

            final_masks.append(final_mask_sq)

        cent_idx = get_nearest_to_center_box(
            image.shape, instances.pred_boxes.tensor.detach().cpu().numpy())

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

        dets = boxes.cpu().round().long().numpy()

        x = (dets[:, 0] + dets[:, 2]) / 2
        y = (dets[:, 1] + dets[:, 3]) / 2
        w = (dets[:, 2] - dets[:, 0])
        h = (dets[:, 3] - dets[:, 1])

        # dets_xywh = np.stack((x, y, w, h)).T

        # confs = np.ones((dets.shape[0], 1))
        # tracker_data = self.tracker.update(np.concatenate((dets, confs), axis=1))
        # tracker_data = self.tracker.update(dets_xywh, confs, image)

        tracker_data = []

        # print(tracker_data)

        # print(ids.shape, dets.shape)

        # inds_after_tracking = np.where()

        # print(dets)
        # print('--------------------')
        # print(tracker_data)

        if len(tracker_data) != 0:
            # print(tracker_data[0].shape)

            for row in tracker_data:
                # for box, i in zip(tracker_data[:,:-1], tracker_data[:, -1]):
                # print(box, i)

                box = row[:-1]
                i = row[-1]

                # draw bounding box
                pts = box.astype(int)

                c = (0, 255, 0)

                pt = (box[[0, 3]].astype(int)) - 2

                # print(pts)

                pt = (int(pt[0]), int(pt[1]))

                cv.putText(image_segmented, f'{i}', pt,
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)

                # cv.rectangle(image_segmented, (int(pts[0]), int(pts[1])),
                #                 (int(pts[2]), int(pts[3])), c, 2)

        self.total += 1

        if len(self.last_box) != 0:
            cv.rectangle(image_segmented, (int(self.last_box[0]), int(self.last_box[1])),
                         (int(self.last_box[2]), int(self.last_box[3])), (0, 255, 255), 2)

        perc = self.counter / self.total * 100
        print(f'{self.counter} / {self.total}, {perc:.2f} %')

        for ix, (box, m) in enumerate(zip(boxes, pred_masks)):

            box = box.cpu().detach().long().numpy()
            iou_n = 0
            if cent_idx == ix:
                if len(self.last_box) != 0:
                    iou_n = iou(self.last_box, np.expand_dims(box, axis=0))

                    if iou_n > self.iou_thresh:
                        self.last_box = box
                else:
                    self.last_box = box

            c = (255, 0, 0) if (
                ix == cent_idx) and iou_n > self.iou_thresh else (0, 0, 255)

            if (ix == cent_idx) and iou_n > self.iou_thresh:
                self.counter += 1
                self.iou_thresh = base_iou_thresh

            if (ix == cent_idx) and iou_n < self.iou_thresh:
                self.iou_thresh -= iou_decay

            # draw bounding box
            pts = box

            cv.rectangle(image_segmented, (int(pts[0]), int(pts[1])),
                         (int(pts[2]), int(pts[3])), c, 2)

            # draw object masks
            x1, y1, x2, y2 = box
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

        # center = np.array(image.shape[:-1]) // 2

        # cv.circle(image_segmented, center[::-1], 2, (0, 255, 0))

        # image_masked = cv.bitwise_and(image, image, mask=mask)

        # cv.imshow('mask', images_masked)

        cv.imshow('segmented', image_segmented)
        cv.waitKey(1)
        # cl = 'box'
        # root_folder = 'noisy_masks'
        # if not os.path.exists(root_folder):
        #     os.mkdir(root_folder)
        # if not os.path.exists(f'{root_folder}/{cl}'):
        #     os.mkdir(f'{root_folder}/{cl}')

        # dur = time.time() - self.start_time
        # cv.imwrite(f'{root_folder}/{cl}/{cl}_{dur:.3f}.png', images_masked)

        # print(f'{root_folder}/{cl}/{cl}_{dur:.3f}.png')

        # if dur > 30.0:
        #     exit()

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
