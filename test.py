import os, sys

from cv2 import circle
from utils.dataset import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
from utils.utils import *
import logging

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
parser.add_argument('--train_meta', default='train_meta.csv', type=str)
parser.add_argument('--test_folder', default='data/public_test', type=str)
parser.add_argument('--test_meta', default='public_test_meta.csv', type=str)
parser.add_argument('--train_policy', default='train', type=str)
parser.add_argument('--test_policy', default='test', type=str)
parser.add_argument('--labels', nargs='+', default=None)

args = parser.parse_args()
# ===================================================================================
running_args = init_running_args(args)

# ===================================================================================
# if running_args.do_train:
#     train_loader = build_dataloader(running_args, )

# train_labels, val_labels = train_test_split(data_labels, test_size=0.1, shuffle=True, random_state=32)

# public_test_data_path = os.path.join(os.getcwd(), 'data', 'public_test')
# public_test_labels = pd.read_csv(os.path.join(public_test_data_path, 'public_test_meta.csv'), header=0,)

_meta = getattr(running_args, 'train_meta')
meta = load_meta(_meta, running_args)

meta.dropna(axis='rows', inplace=True)

l = len(meta)

temp = meta.loc[meta['mask'] == 1]

print(len(temp))
print(l)
print(len(temp)/l)

t = meta.loc[meta['mask'] == 0]

print(len(t))
t.to_csv('train_nomask.csv')
print(t.head(10))
sys.exit()
meta = meta[:100]

if running_args.do_train:
    train_loader = build_dataloader(running_args, meta)


from models.test_model import TestModel
model = TestModel()

from torch.nn import BCELoss
criterion = BCELoss()
from utils.trainer import Trainer

trainer = Trainer(model, criterion, running_args.train_args)
trainer.train(train_loader)
# ===================================================================================


if __name__ == '__main__':
    logger = create_logger(__name__)


#     from torchvision.utils import save_image, make_grid

#     for index, batch in enumerate(train_loader):

#         print(batch['image'].size())
#         if index == 0:
#             grid = make_grid(batch['image'])
#             save_image(grid, 'test.png')

