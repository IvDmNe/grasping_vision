import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.structures import ImageList

from detectron2.structures import Instances
import torch
# import cv2 as cv
import numpy as np
# from detectron2.layers.nms import batched_nms
from torchvision.ops import nms
import rospy
import os
from pathlib import Path

class seg:
    def __init__(self):
        self.cfg = get_cfg()

        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.00  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        # self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        #     "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        if not Path('model_final_f10217.pkl').isfile():
            os.system('wget https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl')

        self.cfg.MODEL.WEIGHTS = 'model_final_f10217.pkl'
        self.predictor = DefaultPredictor(self.cfg)


    def forward(self, image):
        with torch.no_grad():

            t_image = ImageList.from_tensors(
                [torch.Tensor(image).permute(2, 0, 1)])

            features = self.predictor.model.backbone(
                t_image.tensor.cuda())

            proposals, _ = self.predictor.model.proposal_generator(
                t_image, features)

            mask_features = [features[f]
                             for f in self.predictor.model.roi_heads.in_features]

            min_idx = -1
            ids = []
            for idx, (box, score) in enumerate(zip(proposals[0].proposal_boxes[:10], torch.sigmoid(proposals[0].objectness_logits[:10]))):
                if score < 0.92:
                    break
                pts = box.detach().cpu().long()

                # if area is too big or if confidence is less than .92
                if ((pts[2] - pts[0]) * (pts[3] - pts[1]) / (image.shape[0]*image.shape[1]) > 0.3):
                    continue

                ids.append(idx)

            inds_after_nms = nms(
                proposals[0].proposal_boxes[ids].tensor.cpu(), proposals[0].objectness_logits[ids].cpu(), 0.3)

            # mask_features = self.predictor.model.roi_heads.mask_pooler(
            #     mask_features, [proposals[0].proposal_boxes[ids][inds_after_nms]])

            # rospy.logerr(mask_features.shape)
            # rospy.logerr(len(proposals[0].proposal_boxes[ids][inds_after_nms]))

            new_prop = proposals[0][ids][inds_after_nms]

            # rospy.logerr(features)
            # rospy.logerr(new_prop.proposal_boxes.tensor.shape)
            instances, _ = self.predictor.model.roi_heads(
                t_image, features, [new_prop])
            # rospy.logerr(instances[0].pred_classes)

            if len(instances[0]) > len(new_prop):
                instances[0] = instances[0][:len(new_prop)]

            insts_inds_after_nms = nms(
                instances[0].pred_boxes.tensor, instances[0].scores, 0.8)

            mask_features = self.predictor.model.roi_heads.mask_pooler(
                mask_features, [instances[0][insts_inds_after_nms].pred_boxes])

            return mask_features, instances[0][insts_inds_after_nms]
