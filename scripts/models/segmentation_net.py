import cv2 as cv
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.structures import ImageList
from detectron2.modeling import build_model
from detectron2.structures import Instances
from detectron2.checkpoint import DetectionCheckpointer

import torch
# import cv2 as cv
import numpy as np
# from detectron2.layers.nms import batched_nms
from torchvision.ops import nms
import rospy
import os
from pathlib import Path
import pwd
import onnx
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit

from matplotlib import pyplot as plt

import yaml

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()
class seg:
    def __init__(self):
        self.cfg = get_cfg()

        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        # self.cfg.merge_from_file(model_zoo.get_config_file(
        #     "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

        self.cfg.merge_from_file('model_config.yaml')


        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.00  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        # self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        #     "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

        # username = pwd.getpwuid( os.getuid() )[ 0 ]
        
        new_dir = '/ws/src/grasping_vision/scripts'
        rospy.logwarn(f'current dir: {os.getcwd()}. Changing to {new_dir}')
        # os.chdir(new_dir)
        rospy.logwarn(f'No model weights found in {os.getcwd()}, downloading...')
        # if not Path('model_final_f10217.pkl').is_file():
        #     os.system('wget https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl')

        self.cfg.MODEL.WEIGHTS = 'model_final_f10217.pkl'

        with open('model_config.yaml', 'w') as fp:
            yaml.dump(self.cfg, fp)

        self.model = build_model(self.cfg)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)

        # convert model to tensorrt

        # ONNX_FILE_PATH = 'Mask-RCNN_backbone.onnx'

        # torch.onnx.export(self.model.backbone, torch.ones([1, 3, 480, 640]).cuda(), ONNX_FILE_PATH, input_names=['input'],
        #           output_names=['output'], export_params=True)

        # onnx_model = onnx.load(ONNX_FILE_PATH)
        # onnx.checker.check_model(onnx_model)

        # initialize TensorRT engine and parse ONNX model
        # self.engine, self.context = build_engine(ONNX_FILE_PATH)
        
        # # get sizes of input and output and allocate memory required for input data and for output data
        # for binding in self.engine:
        #     if self.engine.binding_is_input(binding):  # we expect only one input
        #         input_shape = self.engine.get_binding_shape(binding)
        #         input_size = trt.volume(input_shape) * self.engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
        #         self.device_input = cuda.mem_alloc(input_size)
        #     else:  # and one output
        #         output_shape = self.engine.get_binding_shape(binding)
        #         # create page-locked memory buffers (i.e. won't be swapped to disk)
        #         self.host_output = cuda.pagelocked_empty(trt.volume(output_shape) * self.engine.max_batch_size, dtype=np.float32)
        #         self.device_output = cuda.mem_alloc(self.host_output.nbytes)

        
        # # Create a stream in which to copy inputs/outputs and run inference.
        # self.stream = cuda.Stream()





        # sample_im = torch.ones([1, 3, 480, 640])
        # self.t_backbone = torch2trt(self.model.backbone, [sample_im.cuda()])

        # self.convert_to_trt()

        
        




    def forward(self, image):
        with torch.no_grad():

            t_image = ImageList.from_tensors(
                [torch.Tensor(image).permute(2, 0, 1)])

            # preprocess input data
            # host_input = np.array(preprocess_image("turkish_coffee.jpg").numpy(), dtype=np.float32, order='C')
            # cuda.memcpy_htod_async(self.device_input, t_image, self.stream)

            # # run inference
            # self.context.execute_async(bindings=[int(self.device_input), int(self.device_output)], stream_handle=self.stream.handle)
            # cuda.memcpy_dtoh_async(self.host_output, self.device_output, self.stream)
            # self.stream.synchronize()

            # # postprocess results
            # features = torch.Tensor(self.host_output).reshape(self.engine.max_batch_size, self.output_shape[0])
            # postprocess(output_data)



            features = self.model.backbone(
                t_image.tensor.cuda())
            # print(features.shape)

            proposals, _ = self.model.proposal_generator(
                t_image, features)

            # mask_features = [features[f]
            #                  for f in self.model.roi_heads.in_features]

            min_idx = -1
            ids = []
            for idx, (box, score) in enumerate(zip(proposals[0].proposal_boxes[:15], torch.sigmoid(proposals[0].objectness_logits[:15]))):
                if score < 0.7:
                    print(score)
                    break

                pts = box.detach().cpu().long()

                # if area is too big or if confidence is less than .92
                if ((pts[2] - pts[0]) * (pts[3] - pts[1]) / (image.shape[0]*image.shape[1]) > 0.3):
                    continue

                ids.append(idx)

            inds_after_nms = nms(
                proposals[0].proposal_boxes[ids].tensor.cpu(), proposals[0].objectness_logits[ids].cpu(), 0.2)


            new_prop = proposals[0][ids][inds_after_nms]



            # print(len(new_prop.proposal_boxes))

            instances, _ = self.model.roi_heads(
                t_image, features, [new_prop])

            # print(len(instances[0]))

            # for box in new_prop.proposal_boxes:
                    
            #     # draw bounding box
            #     pts = box.detach().cpu().long()
            #     cv.rectangle(image, (int(pts[0]), int(pts[1])),
            #                     (int(pts[2]), int(pts[3])), (0, 255, 0), 2)
            
            # plt.imshow(image)
            # plt.show()

            # if len(instances[0]) > len(new_prop):
            #     instances[0] = instances[0][:len(new_prop)]

            insts_inds_after_nms = nms(
                instances[0].pred_boxes.tensor, instances[0].scores, 0.4)

            # print(insts_inds_after_nms.shape)

            # mask_features = self.model.roi_heads.mask_pooler(
            #     mask_features, [instances[0][insts_inds_after_nms].pred_boxes])

            return instances[0][insts_inds_after_nms]


# def build_engine(onnx_file_path):
#             # initialize TensorRT engine and parse ONNX model
#             builder = trt.Builder(TRT_LOGGER)
#             network = builder.create_network()
#             parser = trt.OnnxParser(network, TRT_LOGGER)
            
#             # parse ONNX
#             with open(onnx_file_path, 'rb') as model:
#                 print('Beginning ONNX file parsing')
#                 parser.parse(model.read())
#             print('Completed parsing of ONNX file')

#             # allow TensorRT to use up to 1GB of GPU memory for tactic selection
#             builder.max_workspace_size = 1 << 30
#             # we have only one image in batch
#             builder.max_batch_size = 1
#             # use FP16 mode if possible
#             if builder.platform_has_fast_fp16:
#                 builder.fp16_mode = True
                
#             # generate TensorRT engine optimized for the target platform
#             print('Building an engine...')
#             engine = builder.build_cuda_engine(network)
#             context = engine.create_execution_context()
#             print("Completed creating Engine")

#             return engine, context