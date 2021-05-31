import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.structures import ImageList

from detectron2.structures import Instances
import torch
# import cv2 as cv
import numpy as np


class seg:
    def __init__(self):
        self.cfg = get_cfg()
        # self.cfg.defrost()
        # self.cfg['MODEL']['ROI_HEADS']['SCORE_THRESH_TEST'] = 0.2
        # self.cfg['MODEL']['ROI_MASK_HEAD']['SCORE_THRESH_TEST'] = 0.2
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)

    def preprocess(self, original_image):

        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            # if self.input_format == "RGB":
            #     # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.predictor.aug.get_transform(
                original_image).apply_image(original_image)
            print(original_image.shape)
            print(image.shape)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            print(image.shape)

            inputs = {"image": image.cuda(), "height": height, "width": width}
            return inputs

    def forward(self, image):
        with torch.no_grad():

            # scale = 1/1.333
            # preprocessed_image = self.preprocess(image)
            # images = ImageList.from_tensors(
            #     [torch.Tensor(image).permute(2, 0, 1)])  # preprocessed input tensor

            # instances = self.predictor.model.backbone(
            #     images.tensor.cuda())

            t_image = ImageList.from_tensors(
                [torch.Tensor(image).permute(2, 0, 1)])

            # print(image.shape)

            features = self.predictor.model.backbone(
                t_image.tensor.cuda())

            proposals, _ = self.predictor.model.proposal_generator(
                t_image, features)

            mask_features = [features[f]
                             for f in self.predictor.model.roi_heads.in_features]

            min_idx = -1
            ids = []
            for idx, (box, score) in enumerate(zip(proposals[0].proposal_boxes[:10], torch.sigmoid(proposals[0].objectness_logits[:10]))):
                # print(score)

                if score < 0.975:
                    break
                pts = box.detach().cpu().long()

                # if area is too big or if confidence is less than .975
                if ((pts[2] - pts[0]) * (pts[3] - pts[1]) / (image.shape[0]*image.shape[1]) > 0.3):
                    continue

                ids.append(idx)

            # boxes_after_nms = nms(
            #     proposals[0].proposal_boxes[ids].tensor.cpu().numpy(), 0.6)
            # print(proposals[0].proposal_boxes[ids], boxes_after_nms)

            mask_features = self.predictor.model.roi_heads.mask_pooler(
                mask_features, [proposals[0].proposal_boxes[ids]])

            new_prop = Instances(image.shape[:-1])
            new_prop.proposal_boxes = proposals[0].proposal_boxes[ids]
            new_prop.objectness_logits = proposals[0].objectness_logits[ids]
            instances, _ = self.predictor.model.roi_heads(
                t_image, features, [new_prop])

            return proposals[0].proposal_boxes[ids], mask_features, instances[0]


# Malisiewicz et al.
def nms(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")
