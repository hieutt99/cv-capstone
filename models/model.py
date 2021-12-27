import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import models
from .utils import *


# class MaskClassificationModel(nn.Module):
#     def __init__(self, ):
#         super(MaskClassificationModel, self).__init__()

#         self.yolov5 = None 
#         self.yolov5.eval()
        
#         resnet_kwargs = {
#             "num_classes":1,
#         }
#         self.resnet = resnet18(**resnet_kwargs)

#     def filter_yolo_outputs(self, outputs):
#         pass

#     def interpolate(self, X):
#         pass

#     def forward(self, X):
#         yolo_preds = self.yolov5(X)

def create_resnet(n_classes=1):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, n_classes)
    return model 

def create_efficientnet(n_classes=1, model='efficientnet_b0'):
    if model == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, n_classes)
        return model 

def create_densenet(n_classes=1, model='densenet121'):
    if model == 'densenet121':
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, n_classes)

def create_vgg(n_classes=1, model='vgg16'):
    if model == 'vgg16':
        model = models.vgg16(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, n_classes)
        return model 

class YoloV5CropOutput():
    def __init__(self,):
        
        pass 
    def forward(self, ):
        pass 

class MultiImageClassficationModel(nn.Module):
    def __init__(self, ):
        super(MultiImageClassficationModel, self).__init__()


    def forward(self, ):
        pass