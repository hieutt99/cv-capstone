import os, sys

from cv2 import circle
from utils.dataset import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
from utils.utils import *
import logging


import pandas as pd
import json

# with open(os.path.join('labels', 'facenet_labels_nomask.json')) as fp:
#     facenet_labels = json.load(fp)

# # fname, boxes, probs
# print(len(facenet_labels))


with open(os.path.join('labels', 'yolo_labels_full.json')) as fp:
    yolo_labels = json.load(fp)

# fname, bboxes, confidence, class, name, 
print(len(yolo_labels))

# ==================================================

# output = 'facenet_output'
output = 'yolo_output'

save_folder = os.path.join(os.getcwd(), '_temp_data', output)
image_folder = os.path.join(os.getcwd(), 'data', 'train', 'images')

from torchvision.utils import save_image, draw_bounding_boxes
import torch
from tqdm.auto import tqdm
from PIL import Image
from torchvision import transforms 
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
])

def extract_boxes_facenet(facenet_labels):
    for label in tqdm(facenet_labels):
        fname = label['fname']
        boxes = torch.tensor(label['boxes']).int()
        probs = [str(item) for item in label['probs']]
        
        image = Image.open(os.path.join(image_folder, fname))

        # image = transform(image)*255
        # image = image.type(torch.uint8)

        image = torch.tensor(np.array(image)).permute(2, 0, 1 )
        # print(image.size())


        image = draw_bounding_boxes(image, boxes=boxes, labels=probs, colors='red', width=2)

        save_image(image/255, os.path.join(save_folder, fname))

def extract_boxes_yolo(yolo_labels):
    for label in tqdm(yolo_labels):
        fname = label['fname']
        # _boxes = torch.tensor(label['bboxes']).int()
        _boxes = label['bboxes']
        _names = label['name']
        _confidence = label['confidence']

        image = Image.open(os.path.join(image_folder, fname))
        image = torch.tensor(np.array(image)).permute(2, 0, 1)
        # print(image.size())

        c, h, w = image.size()
        # if max(h, w)>200:


        names = []
        boxes = []
        for index in range(len(_names)):
            if _confidence[index] > 0.35:
                boxes.append(_boxes[index])
                names.append(_names[index])

        boxes = torch.tensor(boxes).int()
        

        image = draw_bounding_boxes(image, boxes=boxes, labels=names, colors='red', width=2)

        save_image(image/255, os.path.join(save_folder, fname))

extract_boxes_yolo(yolo_labels)
sys.exit()

if __name__ == '__main__':
    logger = create_logger(__name__)


#     from torchvision.utils import save_image, make_grid

#     for index, batch in enumerate(train_loader):

#         print(batch['image'].size())
#         if index == 0:
#             grid = make_grid(batch['image'])
#             save_image(grid, 'test.png')

