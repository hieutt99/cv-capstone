import os, sys
from utils.dataset import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
from utils.utils import *

# ===================================================================================
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.yaml', type=str)
parser.add_argument('--do_train', nargs='?', type=str2bool, const=True, default=False)
parser.add_argument('--do_predict', nargs='?', type=str2bool, const=True, default=False)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--train_folder', default='data/train', type=str)
parser.add_argument('--test_folder', default='data/public_test', type=str)
parser.add_argument('--train_policy', default='train', type=str)
parser.add_argument('--test_policy', default='test', type=str)
parser.add_argument('--labels', nargs='+', default=None)

args = parser.parse_args()
# ===================================================================================
running_args = init_running_args(args)

# ===================================================================================
# if running_args.do_train:
#     train_loader = build_dataloader(running_args, )

from torchvision.utils import make_grid, save_image
import torch



# trainer.eval(train_loader)

# ===================================================================================


if __name__ == '__main__':

    print("ok")

