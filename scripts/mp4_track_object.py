#!/usr/bin/python3


# this project packages
from numpy.lib import math
from models.segmentation_net import *
from models.feature_extractor import image_embedder, dino_wrapper
from models.mlp import MLP
from models.knn import *
from utilities.utils import *


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
import os

from utilities.sort import *

from deep_sort.deep_sort import DeepSort
from deep_sort.sort.iou_matching import iou
torch.set_grad_enabled

setup_logger()

base_iou_thresh = 0.8
iou_decay = 0.00


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

        self.seg_net = seg()

        emb_size = 64

        emb_file = '/home/iiwa/ros_ws/src/grasping_vision/scripts/models/embedder_best1.pth'
        trunk_file = '/home/iiwa/ros_ws/src/grasping_vision/scripts/models/trunk_best1.pth'

        # self.embedder = image_embedder(
        #     trunk_file=trunk_file, emb_file=emb_file, emb_size=emb_size)
        self.embedder = dino_wrapper()

        # self.classifier = knn_torch(
        #     datafile='knn_data_metric_learning.pth')

        self.colors = colormap()

        self.start_time = time.time()

        # self.tracker = Sort()
        # self.tracker = DeepSort(
        #     'ckpt.t7', max_dist=0.5, max_iou_distance=0.9, n_init=5, nn_budget=100, max_age=200)
        # self.tracker.extractor = dino_wrapper()

        self.current_id = None

        self.last_box = []

        self.counter = 0
        self.total = 0

        self.iou_thresh = base_iou_thresh
        self.last_embs = []

        print('Segmentaion node: Init complete')

    def do_segmentation(self, image):

        instances = self.seg_net.forward(
            image)

        if len(instances.pred_boxes.tensor) == 0:
            # rospy.logerr_throttle(2, 'no objects found')
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

        image = self.im.copy()

        image_segmented = image.copy()

        ret_seg = self.do_segmentation(image)

        if ret_seg:
            images_masked, boxes, pred_masks, cent_idx = ret_seg
        else:
            return image

        self.total += 1

        if len(self.last_box) != 0:
            cv.rectangle(image_segmented, (int(self.last_box[0]), int(self.last_box[1])),
                         (int(self.last_box[2]), int(self.last_box[3])), (0, 255, 255), 2)

        perc = self.counter / self.total * 100
        print(f'{self.counter} / {self.total}, {perc:.2f} %')

        for ix, (box, m) in enumerate(zip(boxes, pred_masks)):

            c = (0, 0, 255)

            box = box.cpu().detach().long().numpy()

            # get embeddings of each object
            x1, y1, x2, y2 = box
            sz = (int(x2 - x1), int(y2 - y1))

            mask_rs = cv.resize(m.squeeze().detach().cpu().numpy(), sz)

            cur_mask = np.zeros((image.shape[:-1]), dtype=np.uint8)
            cur_mask[y1:y2, x1:x2] = (mask_rs + 0.5).astype(int)

            image_masked = cv.bitwise_and(image, image, mask=cur_mask)

            final_mask = image_masked[y1:y2, x1:x2]

            final_mask_sq = get_padded_image(final_mask)

            embs = self.embedder(
                [final_mask_sq]).detach().cpu()
            iou_n = 0

            # check if box is near to borders
            center = (240, 320)
            x = (x1 + x2) // 2
            y = (y1 + y2) // 2

            cv.circle(image_segmented, center[::-1], 200, (255, 255, 0), 1)
            if math.sqrt((center[1] - y)**2 + (center[0] - x)**2) > 200:
                c = (255, 255, 255)
                continue

            if cent_idx == ix:
                if len(self.last_box) != 0:
                    iou_n = iou(self.last_box, np.expand_dims(box, axis=0))

                    if iou_n > self.iou_thresh:
                        self.last_box = box
                        c = (255, 0, 0)

                else:
                    self.last_box = box

            # c = (255, 0, 0) if (
            #     ix == cent_idx) and iou_n > self.iou_thresh else (0, 0, 255)
            if len(self.last_embs) == 0:
                self.last_embs = embs
            dist = cdist(embs.cpu(), self.last_embs.cpu(),
                         metric='cosine').squeeze()
            print(dist)

            dist_thresh = 0.4
            if dist < dist_thresh:
                self.last_embs = embs
                cv.imshow('lat mask', final_mask_sq)
                # self.last_box = box

            # if ix != cent_idx:

            # compactness_last = (self.last_box[2] - self.last_box[0]) * (self.last_box[3] -
            #                                                             self.last_box[1]) / (self.last_box[1] + self.last_box[0])
            # compactness_curr = (box[2] - box[0]) * (box[3] -
            #                                         box[1]) / (box[1] + box[0])

            # area_sim = (self.last_box[2] - self.last_box[0]) * (self.last_box[3] -
            #                                                     self.last_box[1]) / ((box[2] - box[0]) * (box[3] - box[1]))

            # compactness_sim = compactness_last / compactness_curr

            # print(compactness_sim)
            # print(compactness_last, compactness_curr)

            # if area_sim

            if dist < dist_thresh:
                c = (0, 255, 0)
            # if abs(compactness_sim - 1) < 0.4:
            #     c = (0, 255, 0)
            #     cent_idx = ix

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

        end = time.time()
        fps = 1 / (end - start)
        print(f'FPS: {fps:.2f}')
        return image_segmented


if __name__ == '__main__':
    freq = 100
    node = ImageListener()

    obj = 'phone'

    input_filename = f'/home/iiwa/Nenakhov/bag_dataset/{obj}_dataset_camera_color_image_raw.mp4'

    if not os.path.isfile(input_filename):
        print('invalid filename: ', input_filename)
    output_filename = f'{obj}.mp4'
    reader = cv.VideoCapture(input_filename)

    frame_width = int(reader.get(3))
    frame_height = int(reader.get(4))
    frame_size = (frame_width, frame_height)
    writer = cv.VideoWriter(output_filename, cv.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 20, frame_size)

    if not reader.isOpened():
        print('video not readed!')

    while reader.isOpened():
        ret, frame = reader.read()
        if not ret:
            print('no frame received, exit')
            break

        node.im = frame
        out_image = node.run_proc()

        cv.imshow('seg', out_image)
        key = cv.waitKey(1)

        if ord('q') & 0xFF == key:
            cv.destroyAllWindows()
            break
        elif key == ord('w'):
            cv.waitKey()
