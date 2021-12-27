import torch.nn.functional as F


def interpolate(image, size=224):
    return F.interpolate(
                        image, 
                        size=size, 
                        mode='bilinear', 
                        align_corners=True
                        )

def crop_images(images, bboxes, interpolate=True):
    pass 
