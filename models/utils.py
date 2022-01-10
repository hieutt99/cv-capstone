import torch.nn.functional as F
from torchvision.models.detection import *

def interpolate(image, size=224):
    return F.interpolate(
                        image, 
                        size=size, 
                        mode='bilinear', 
                        align_corners=True
                        )

def crop_images(images, bboxes, interpolate=True):
    pass 

MODEL_DICT = {
    'fasterrcnn_resnet50_fpn':fasterrcnn_resnet50_fpn,
    'fasterrcnn_mobilenet_v3_large_fpn':fasterrcnn_mobilenet_v3_large_fpn,
}

def create_model(model_name, num_classes=3):
    if model_name in MODEL_DICT.keys():
        return MODEL_DICT[model_name](pretrained_backbone=True, num_classes=num_classes)
    return None