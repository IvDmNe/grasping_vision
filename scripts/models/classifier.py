import cv2 as cv
import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
import torch
from matplotlib import pyplot as plt
from torch import nn
from utils.vis import visualize_segmentation, get_rects, get_rotated_rois
from utils.augment import convert_to_tensor, SquarePad
from models.feature_extractor import feature_extractor
import os
from models.knn import knn_torch
from utils.utils import *
import torchvision
import time
from tqdm import tqdm


class classifier:
    def __init__(self, backbone=None, knnClassifier=None, save_dir=None):

        if not backbone:
            backbone = torchvision.models.squeezenet1_1(pretrained=True)
            backbone.eval()
        self.extractor = feature_extractor(backbone)

        if torch.cuda.is_available():
            self.extractor.cuda()

        self.mode = 'inference'
        self.was_trained = False
        self.train_cl = None

        if knnClassifier:
            self.knn = knnClassifier
        else:
            self.knn = knn_torch()

        self.prev_rects = None

        self.prev_ids = None

        self.total_id = 0
        if not save_dir:
            self.save_dir = 'save_images'
        os.makedirs(self.save_dir, exist_ok=True)

    def save_deep_features(self):

        files = []

        for d in os.listdir(self.save_dir):
            for f in os.listdir(self.save_dir + '/' + d):
                files.append(self.save_dir + '/' + d + '/' + f)

        # for root, dirs, fs in os.walk(self.save_dir + '/' + self.train_cl):
        #     if fs:
        #         for f in fs:
        #             files.append(root + '/' + f)
        rgb_files = []
        depth_files = []
        for f in files:
            if 'rgb' in f:
                rgb_files.append(f)
            elif 'depth' in f:
                depth_files.append(f)

        all_deep_features = []
        cl_names = []
        print(files)

        if len(files) == 0:
            print('no files found')
            return
        for rgb_f, depth_f in tqdm(zip(rgb_files, depth_files), total=len(rgb_files)):

            rgb = cv.imread(rgb_f)
            class_name = rgb_f.split('/')[-2]
            cl_names.append(class_name)
            depth_im = cv.imread(depth_f)
            depth = cv.cvtColor(depth_im, cv.COLOR_BGR2GRAY)
            rgb_rois = convert_to_tensor([rgb], shape=(224, 224))

            depth_rois = convert_to_tensor([depth], shape=(224, 224))

            if torch.cuda.is_available():
                rgb_rois = rgb_rois.cuda()
                depth_rois = depth_rois.cuda()

            deep_rgb_features = self.extractor(rgb_rois)
            deep_depth_features = self.extractor(depth_rois)

            # deep_features = torch.cat([deep_rgb_features, deep_depth_features])
            deep_features = deep_rgb_features
            all_deep_features.append(deep_features)

        all_deep_features = torch.stack(all_deep_features)
        self.knn.add_points(all_deep_features, cl_names)

    def save_rois(self, rgb, depth, mask, class_name):

        rgb_rois, depth_rois, cntrs = get_rotated_rois(rgb, depth, mask)
        if not rgb_rois:
            print('no objects')
            return

        center_cntr = find_nearest_to_center_cntr(cntrs, rgb.shape)

        if center_cntr is None:
            print('no objects found at center')
            return
        center_index = cntrs.index(center_cntr)
        center_rgb_roi = rgb_rois[center_index]
        center_depth_roi = depth_rois[center_index]

        # center_rgb_roi = SquarePad(center_rgb_roi)
        # center_depth_roi = SquarePad(center_depth_roi)

        timestamp = time.time()
        print('')
        os.makedirs(f'{self.save_dir}/{class_name}', exist_ok=True)
        rgb_file = f'{self.save_dir}/{class_name}/rgb_{timestamp}.png'
        depth_file = f'{self.save_dir}/{class_name}/depth_{timestamp}.png'

        cv.imwrite(rgb_file, center_rgb_roi)
        cv.imwrite(depth_file, center_depth_roi)
        os.makedirs(f'{self.save_dir}/{class_name}/raw_images', exist_ok=True)
        rgb_raw_file = f'{self.save_dir}/{class_name}/raw_images/rgb_{timestamp}.png'
        depth_raw_file = f'{self.save_dir}/{class_name}/raw_images/depth_{timestamp}.png'
        mask_file = f'{self.save_dir}/{class_name}/raw_images/mask_{timestamp}.png'

        cv.imwrite(rgb_raw_file, rgb)
        cv.imwrite(depth_raw_file, depth)
        cv.imwrite(mask_file, mask)

        print(rgb_file)
        print(depth_file)
        return

    def process_rgbd(self, rgb_im, depth_im, mask):

        if self.mode == 'train':
            self.save_rois(rgb_im, depth_im, mask, self.train_cl)

        elif self.mode == 'inference':

            if self.was_trained:
                print('start saving deep features')
                self.save_deep_features()
                self.was_trained = False
                print('deep features saved')

            # return if there was no training before
            if self.knn.x_data is None:
                # print('no trained data')
                return

            # feed to feature extractor each roi
            rgb_rois, depth_rois, cntrs = get_rotated_rois(
                rgb_im, depth_im, mask)

            if not rgb_rois:
                return
            # print('f')
            rgb_rois = convert_to_tensor(rgb_rois, shape=(224, 224))
            depth_rois = convert_to_tensor(depth_rois, shape=(224, 224))

            if torch.cuda.is_available():
                rgb_rois = rgb_rois.cuda()
                depth_rois = depth_rois.cuda()

            # print(print(next(self.extractor.parameters()).device)
            deep_rgb_features = self.extractor(rgb_rois)
            # deep_depth_features = self.extractor(depth_rois)

            if len(deep_rgb_features.shape) == 1:
                deep_rgb_features = deep_rgb_features.unsqueeze(0)
                # deep_depth_features = deep_depth_features.unsqueeze(0)

            # deep_features = torch.cat([deep_rgb_features, deep_depth_features], dim=1)
            deep_features = deep_rgb_features

            # feed deep features to knn
            classes = self.knn.classify(deep_features)

            drawing = rgb_im.copy()

            # cv.drawContours(drawing, cntrs, -1, (255, 0, 255), 2)
            NUM_COLORS = len(set(self.knn.y_data))
            cm = plt.get_cmap('gist_rainbow')
            colors = [cm(1. * i/NUM_COLORS) for i in range(NUM_COLORS)]
            centers_of_objs = []
            for (cntr, cl) in zip(cntrs, classes):

                i = list(set(self.knn.y_data)).index(cl)
                color = (colors[i][0] * 255, colors[i]
                         [1] * 255, colors[i][2] * 255)

                M = cv.moments(cntr)
                if M['m00'] == 0:
                    print('division by zero')
                    continue
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                centers_of_objs.append((cX, cY))
                cv.drawContours(drawing, cntrs, -1, color, 2)
                cv.putText(drawing, cl, (cX - 10, cY - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv.circle(drawing, (cX, cY), 2, color)

            return drawing, classes, centers_of_objs

        else:
            print('unknown mode', self.mode, '!')
