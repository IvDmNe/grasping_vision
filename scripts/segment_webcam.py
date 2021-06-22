#!/home/ivan/anaconda3/bin/python

# this project packages
from models.segmentation_net import *
from models.knn import *

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
from scipy.spatial.distance import euclidean

setup_logger()


# lock = threading.Lock()
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

        self.cam = cv.VideoCapture(0)

        # self.cv_bridge = CvBridge()

        # # initialize a node
        # rospy.init_node("sod", log_level=rospy.INFO)
        # self.rgb_pub = rospy.Publisher('/rgb_masked', Image, queue_size=10)
        # self.depth_pub = rospy.Publisher('/depth_masked', Image, queue_size=10)
        # self.segmented_view_pub = rospy.Publisher(
        #     '/segmented_view', Image, queue_size=10)

        self.base_frame = 'measured/base_link'
        self.camera_frame = 'measured/camera_color_optical_frame'
        self.target_frame = self.base_frame

        # rgb_sub = message_filters.Subscriber('/camera/color/image_raw',
        #                                      Image, queue_size=10)

        # depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw',
        #                                        Image, queue_size=10)

        # rospy.Subscriber('/command_from_human',
        #                  String, self.callback_mode)

        self.seg_net = seg()
        self.classifier = knn_torch(
            datafile='/home/ivan/ros_ws/src/pc_proc/scripts/knn_data.pth')

        # ts = message_filters.ApproximateTimeSynchronizer(
        #     [rgb_sub, depth_sub], 1, 0.1)
        # ts.registerCallback(self.callback_rgbd)

        self.colors = colormap()

        # rospy.loginfo('Segmentaion node: Init complete')

    def do_segmentation(self, image):

        proposal_boxes, seg_mask_features, instances = self.seg_net.forward(
            image)

        # for box in proposal_boxes:

        # pts = box.detach().cpu().long()
        # cv.rectangle(masked_image, (pts[0], pts[1]),
        #              (pts[2], pts[3]), (255, 0, 0), 2)

        # draw object masks
        # for box, mask in zip(instances.pred_boxes, instances.pred_masks):
        #     x1, y1, x2, y2 = box.round().long()
        #     sz = (x2 - x1, y2 - y1)

        #     mask_rs = cv.resize(mask.squeeze().detach().cpu().numpy(), sz)

        #     cur_mask = np.zeros((image.shape[:-1]))
        #     cur_mask[y1:y2, x1:x2] = mask_rs
        #     masked_image[cur_mask > 0.5] = (0, 255, 255)

        if len(seg_mask_features) == 0 or len(instances.pred_boxes.tensor) == 0:
            print('no objects found')
            return

        # decrease dimension of features by global pooling
        features = F.avg_pool2d(
            seg_mask_features, kernel_size=seg_mask_features.size()[2:])

        # mask = get_one_mask(
        #     instances.pred_boxes.tensor.round().long().detach().cpu().numpy(), instances.pred_masks, image).astype(np.uint8)

        # return features, proposal_boxes.tensor, masked_image, mask, instances.pred_masks
        # return features, proposal_boxes.tensor, instances.pred_masks
        return features, instances.pred_boxes.tensor, instances.pred_masks

    # def send_images_to_topics(self, image_masked=None, depth_masked=None, image_segmented=None):
    #     if image_masked is not None:
    #         rgb_msg = self.cv_bridge.cv2_to_imgmsg(image_masked)
    #         rgb_msg.header.stamp = rospy.Time.now()
    #         # rgb_msg.header.frame_id = rgb_frame_id
    #         rgb_msg.encoding = 'bgr8'
    #         self.rgb_pub.publish(rgb_msg)
    #     if depth_masked is not None:
    #         depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_masked)
    #         depth_msg.header.stamp = rospy.Time.now()
    #         # depth_msg.header.frame_id = rgb_frame_id
    #         depth_msg.encoding = '32FC1'
    #         self.depth_pub.publish(depth_msg)
    #     if image_segmented is not None:
    #         mask_msg = self.cv_bridge.cv2_to_imgmsg(image_segmented)
    #         mask_msg.header.stamp = rospy.Time.now()
    #         # mask_msg.header.frame_id = rgb_frame_id
    #         mask_msg.encoding = 'bgr8'
    #         self.segmented_view_pub.publish(mask_msg)

    def save_data(self, features, cl, im_shape, boxes):

        center_idx = get_nearest_to_center_box(
            im_shape, boxes.cpu().numpy())

        # check if bounding box is very different from previous
        if self.check_sim(boxes[center_idx].cpu().numpy()):
            if self.x_data_to_save == None:
                self.x_data_to_save = features[center_idx].squeeze(
                ).unsqueeze(0)
            else:
                self.x_data_to_save = torch.cat(
                    [self.x_data_to_save, features[center_idx].squeeze().unsqueeze(0)])

    def run_proc(self):

        start = time.time()

        image_masked = None
        depth_masked = None

        # with lock:
        # if self.im is None:
        #     rospy.logerr_throttle(5, "No image received")
        #     return
        # image = self.im.copy()
        # depth = self.depth.copy()
        _, image = self.cam.read()
        # print(image.shape)

        # segment rgb image
        image = cv.resize(image, (640, 480))
        # depth = cv.resize(depth, (640, 480))

        image_segmented = image.copy()
        ret_seg = self.do_segmentation(image)
        if ret_seg:
            features, boxes, pred_masks = ret_seg
        else:
            return

        # filter depth by 1 meter
        # depth[depth > 1.0] = 1.0

        # choose action according to working mode
        if self.working_mode.split(' ')[0] == 'train':
            # choose only the nearest bbox to the center
            cl = self.working_mode.split(
                ' ')[1]
            self.save_data(features, cl, image.shape, boxes)

        elif self.working_mode == 'inference':
            if self.prev_mode.split(' ')[0] == 'train' and self.working_mode == 'inference':
                self.feed_features_to_classifier()

            classes = self.classifier.classify(features.squeeze())

            if isinstance(classes, str):
                classes = [classes]
            if classes:
                # draw labels and masks
                for cl, box, m in zip(classes, boxes, pred_masks):
                    idx = self.classifier.classes.index(cl)
                    c = self.colors[idx].astype(np.uint8).tolist()

                    # draw bounding box
                    pts = box.detach().cpu().long()
                    cv.rectangle(image_segmented, (pts[0], pts[1]),
                                 (pts[2], pts[3]), c, 2)

                    # draw label
                    pt = (box[:2].round().long()) - 2
                    cv.putText(image_segmented, cl, tuple(pt),
                               cv.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)

                    # draw object masks
                    x1, y1, x2, y2 = box.round().long()
                    sz = (x2 - x1, y2 - y1)

                    mask_rs = cv.resize(
                        m.squeeze().detach().cpu().numpy(), sz)

                    cur_mask = np.zeros((image.shape[:-1]), dtype=np.uint8)
                    cur_mask[y1:y2, x1:x2] = (mask_rs + 0.5).astype(int)
                    cntrs, _ = cv.findContours(
                        cur_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                    cv.drawContours(image_segmented, cntrs, -
                                    1, c, 2)

                mask = get_one_mask(
                    boxes.cpu().int().numpy(), pred_masks, image).astype(np.uint8)

                cv.imshow('im', image_segmented)
                cv.waitKey(1)

        elif self.working_mode.split(' ')[0] == 'give':
            demand_class = self.working_mode.split(' ')[1]
            # rospy.logwarn(f'Command: {self.working_mode}')
            self.working_mode = 'inference'
            classes = self.classifier.classify(features.squeeze())

            if isinstance(classes, str):
                classes = [classes]

            if classes:
                if not (demand_class in classes):
                    # rospy.logwarn(
                    #     f'object: {demand_class} not found among {classes}')
                    return

                idx = classes.index(demand_class)
                mask = get_one_mask(
                    boxes.cpu().int().numpy(), pred_masks, image, idx).astype(np.uint8)
                # apply masking
                image_masked = cv.bitwise_and(image, image, mask=mask)
                # depth_masked = cv.bitwise_and(depth, depth, mask=mask)

        # else:
        #     # rospy.logerr_throttle(
        #     #     1, f'invalid working mode: {self.working_mode}')
            # return

        # self.send_images_to_topics(image_masked, depth_masked, image_segmented)

        end = time.time()
        fps = 1 / (end - start)
        print(f'FPS: {fps:.2f}')

    def check_sim(self, box):
        box_center = ((box[3] + box[1]) // 2, (box[2] + box[0]) // 2)

        if not self.prev_training_mask_info:
            self.prev_training_mask_info = box_center
            return True
        # print(euclidean(box_center, self.prev_training_mask_info))
        if euclidean(box_center, self.prev_training_mask_info) > 80:
            # rospy.logwarn(
            #     f"skipping image, too far from previous: {euclidean(box_center, self.prev_training_mask_info):.2f} (threshold: {80})")
            # print(euclidean(box_center, self.prev_training_mask_info))
            return False
        else:
            self.prev_training_mask_info = box_center
            return True

    def feed_features_to_classifier(self):
        # feed saved features to classifier when working mode is changed to "inference"

        # rospy.logwarn('saving features')

        # inl_inds = removeOutliers(self.x_data_to_save.cpu(), 1.5)
        # self.x_data_to_save = self.x_data_to_save[inl_inds]

        self.classifier.add_points(self.x_data_to_save, [self.prev_mode.split(' ')[
            1]] * self.x_data_to_save.shape[0])

        self.x_data_to_save = None
        self.prev_mode = 'inference'


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


def get_one_mask(boxes, mask, image, n_mask=None):
    if n_mask is None:
        cent_ix = get_nearest_to_center_box(image.shape, boxes)
    else:
        cent_ix = n_mask
    x1, y1, x2, y2 = boxes[cent_ix]
    sz = (x2 - x1, y2 - y1)
    mask_rs = cv.resize(mask[cent_ix].squeeze().detach().cpu().numpy(), sz)

    cur_mask = np.zeros((image.shape[: -1]))
    cur_mask[y1: y2, x1: x2] = mask_rs
    return cv.threshold(cur_mask, 0.5, 1.0, cv.THRESH_BINARY)[1]


def removeOutliers(x, outlierConstant):
    # a = np.array(x)
    # print(a.shape)
    cur_x = x.clone()
    # inliers = np.array()
    for col in range(x.shape[1]):
        a = cur_x[:, col]

        upper_quartile = np.percentile(a, 75)
        lower_quartile = np.percentile(a, 25)
        IQR = (upper_quartile - lower_quartile) * outlierConstant
        quartileSet = (lower_quartile - IQR, upper_quartile + IQR)

        cur_x = cur_x[np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))]
        print(cur_x.shape)

    # print(cur_x)
    return cur_x
    # return np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))


if __name__ == '__main__':

    listener = ImageListener()
    # rate = rospy.Rate(freq)
    try:
        while True:
            listener.run_proc()
            # rate.sleep()
    finally:
        cv.destroyAllWindows()
