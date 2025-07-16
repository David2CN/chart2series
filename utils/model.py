import torch
import torchvision.transforms as T
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from PIL import Image
from utils.engine import non_max_suppression


class Detector:
    def __init__(self, weights: str, box_score_thresh: float=0.5,
                 num_classes: int=8, transforms: T.Compose=None):
        self.transforms = transforms
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1,
                                                   box_score_thresh=box_score_thresh)
        
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model.load_state_dict(torch.load(weights))

    def process_input(self, inputs: Image.Image):
        if self.transforms:
            inputs = self.transforms(inputs)
            if len(inputs.shape) < 4:
                inputs = inputs.unsqueeze(0)
        return inputs
    
    def __call__(self, inputs: Image.Image):
        inputs = self.process_input(inputs)
        self.model.eval()
        out = self.model(inputs)
        nms_out = [non_max_suppression(i) for i in out]
        return nms_out
