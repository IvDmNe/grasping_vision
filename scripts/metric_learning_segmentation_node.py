#!/usr/bin/python3


# this project packages
from torchvision.transforms.functional import perspective
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
import re
import albumentations as A
from albumentations.pytorch import ToTensorV2

torch.set_grad_enabled(False)

setup_logger()


lock = threading.Lock()
freq = 100


class ImageListener:

    def __init__(self):

        self.im = None
        self.depth = None
        rgb_frame_id = None
        rgb_frame_stamp = None

        self.image_masked = None
        self.depth_masked = None

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

        emb_size = 64

        # folder = f'/home/iiwa/Nenakhov/metric_learning/example_saved_models/mobilenetv3_small_{emb_size}_100cl'
        # fs = os.listdir(folder)

        # r = re.compile("embedder_best")
        # emb_file = folder + '/' + list(filter(r.match, fs))[0]

        # r = re.compile("trunk_best")
        # trunk_file = folder + '/' + list(filter(r.match, fs))[0]

        emb_file = 'models/embedder_best1.pth'
        trunk_file = 'models/trunk_best1.pth'


        self.embedder = image_embedder(trunk_file=trunk_file, emb_file=emb_file, emb_size=emb_size)
        self.classifier = knn_torch(
            datafile='datafiles/14_07_data_aug5.pth', knn_size=20)
            # datafile='knn_data_metric_learning.pth')

        ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], 1, 0.1)
        ts.registerCallback(self.callback_rgbd)

        self.colors = colormap()

        self.transforms = A.Compose(
           [A.LongestMaxSize(max_size=224),
            A.PadIfNeeded(min_height=224, min_width=224),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.3, rotate_limit=180, p=0.5, border_mode=cv.BORDER_CONSTANT),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.Perspective(scale=(0.05, 0.2)),
            A.RandomBrightnessContrast(p=0.5),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # ToTensorV2(),
            ])


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

        # print(instances.pred_masks.shape)

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

    def send_images_to_topics(self, image_masked=None, depth_masked=None, image_segmented=None):

        if image_masked is not None:
            rgb_msg = self.cv_bridge.cv2_to_imgmsg(image_masked)
            rgb_msg.header.stamp = rospy.Time.now()
            # rgb_msg.header.frame_id = rgb_frame_id
            rgb_msg.encoding = 'bgr8'
            self.rgb_pub.publish(rgb_msg)
        if depth_masked is not None:
            depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_masked)
            depth_msg.header.stamp = rospy.Time.now()
            # depth_msg.header.frame_id = rgb_frame_id
            depth_msg.encoding = '32FC1'
            self.depth_pub.publish(depth_msg)
        if image_segmented is not None:
            mask_msg = self.cv_bridge.cv2_to_imgmsg(image_segmented)
            mask_msg.header.stamp = rospy.Time.now()
            # mask_msg.header.frame_id = rgb_frame_id
            mask_msg.encoding = 'bgr8'
            self.segmented_view_pub.publish(mask_msg)

    def save_data(self, images_masked, im_shape, boxes):

        center_idx = get_nearest_to_center_box(
            im_shape, boxes.cpu().numpy())

        # augment masked images before passing to embedder
        imgs = [self.transforms(image=images_masked[center_idx])['image'] for _ in range(5)]
            
        
        features = self.embedder(imgs)


        # check if bounding box is very different from previous



        if self.check_sim(boxes[center_idx].cpu().numpy()):
            if self.x_data_to_save is None:
                self.x_data_to_save = features.squeeze()
            else:
                self.x_data_to_save = torch.cat(
                    [self.x_data_to_save, features.squeeze()])
        return center_idx

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

        # image = cv.resize(image, (640 // 2, 480 // 2))
        # depth = cv.resize(depth, (640 // 2, 480 // 2))

        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)


        image_segmented = image.copy()
        ret_seg = self.do_segmentation(image)
        if ret_seg:
            images_masked, boxes, pred_masks = ret_seg
        # else:
        #     return

            # filter depth by 1 meter
            depth[depth > 1.0] = 1.0

            if self.classifier.y_data is None:
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

            # choose action according to working mode
            if self.working_mode.split(' ')[0] == 'train':
                # choose only the nearest bbox to the center
                cl = self.working_mode.split(
                    ' ')[1]


                


                center_idx = self.save_data(images_masked, image.shape, boxes)

                box = boxes[center_idx]
                m = pred_masks[center_idx]

                c = (255, 0, 0)

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
                mask = cur_mask


            elif self.working_mode == 'inference':
                if self.prev_mode.split(' ')[0] == 'train' and self.working_mode == 'inference':
                    self.feed_features_to_classifier()

                features = self.embedder(images_masked)
                ret = self.classifier.classify(features)
                
                if ret:
                    classes, confs, min_dists = ret

                    if isinstance(classes, str):
                        classes = [classes]
                    if classes:
                        # draw labels and masks
                        for cl, conf, min_dist, box, m in zip(classes, confs, min_dists, boxes, pred_masks):
                            idx = self.classifier.classes.index(cl)
                            c = self.colors[idx].astype(np.uint8).tolist()

                            # if confidence is less than the threshold, PAINT IT BLACK
                            if min_dist > 0.2 or conf < 0.8:
                                # continue 
                                c = (0, 0, 0)

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
                            # if confidence is less than the threshold, don't draw label and confidence
                            if min_dist > 0.2 or conf < 0.8:
                                continue               
                            # draw bounding box
                            pts = box.detach().cpu().long()
                            cv.rectangle(image_segmented, (int(pts[0]), int(pts[1])),
                                        (int(pts[2]), int(pts[3])), c, 2)

                            # draw label
                            pt = (box[:2].round().long()) - 2
                            pt = (int(pt[0]), int(pt[1]))
                            cv.putText(image_segmented, f'{cl} {conf:.2f} {min_dist:.2f}', pt,
                                    cv.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)

                        mask = get_one_mask(
                            boxes.cpu().int().numpy(), pred_masks, image).astype(np.uint8)
                        # print(mask)

            elif self.working_mode.split(' ')[0] == 'give':
                features = self.embedder(images_masked)
                demand_class = self.working_mode.split(' ')[1]
                rospy.logwarn(f'Command: {self.working_mode}')
                self.working_mode = 'inference'
                classes, confs, dists = self.classifier.classify(features.squeeze())


                if isinstance(classes, str):
                    classes = [classes]
                if classes:

                    # threshold predictions
                    class_idxs = [idx for idx, (cl, conf, dist) in enumerate(zip(classes, confs, dists)) if (conf > 0.8 and dist < 0.2)]
                    high_propb_classes = [classes[i] for i in class_idxs]
                    # rospy.logwarn(classes)




                    if not (demand_class in high_propb_classes):
                        rospy.logwarn(
                            f'object: {demand_class} not found among {high_propb_classes}')
                        return

                    idx = class_idxs[high_propb_classes.index(demand_class)]
                    mask = get_one_mask(
                        boxes.cpu().int().numpy(), pred_masks, image, idx).astype(np.uint8)
                    # apply masking

                    self.image_masked = cv.bitwise_and(image, image, mask=mask)
                    self.depth_masked = cv.bitwise_and(depth, depth, mask=mask)
            elif self.working_mode.split(' ')[0] == 'remove':
                # remove known class

                self.classifier.remove_class(self.working_mode.split(' ')[1])
                

                rospy.logwarn(f'Class "{self.working_mode.split(" ")[1]}" removed')
                self.working_mode = 'inference'
                

            else:
                rospy.logerr_throttle(
                    1, f'invalid working mode: {self.working_mode}')
                # return
        
        self.send_images_to_topics(self.image_masked, self.depth_masked, image_segmented)


        end = time.time()
        fps = 1 / (end - start)
        rospy.logwarn_throttle(1, f'FPS: {fps:.2f}')

    def check_sim(self, box):
        box_center = ((box[3] + box[1]) // 2, (box[2] + box[0]) // 2)

        if not self.prev_training_mask_info:
            self.prev_training_mask_info = box_center
            return True
        # print(euclidean(box_center, self.prev_training_mask_info))
        if euclidean(box_center, self.prev_training_mask_info) > 50:
            rospy.logwarn(
                f"skipping image, too far from previous: {euclidean(box_center, self.prev_training_mask_info):.2f} (threshold: {80})")
            # print(euclidean(box_center, self.prev_training_mask_info))
            return False
        else:
            self.prev_training_mask_info = box_center
            return True

    def feed_features_to_classifier(self):
        # feed saved features to classifier when working mode is changed to "inference"

        rospy.logwarn('saving features')
        if self.x_data_to_save is not None:
            print(self.x_data_to_save.shape)

            self.classifier.add_points(self.x_data_to_save, [self.prev_mode.split(' ')[
                1]] * self.x_data_to_save.shape[0])
        else:
            rospy.logwarn_throttle(5,'No features were saved')

        self.x_data_to_save = None
        self.prev_mode = 'inference'


if __name__ == '__main__':

    listener = ImageListener()
    rate = rospy.Rate(freq)
    try:
        while not rospy.is_shutdown():

            # with profiler.profile(with_stack=True, profile_memory=True) as prof:
            listener.run_proc()

            # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=10))
            # exit()
            rate.sleep()
    finally:
        cv.destroyAllWindows()
