import os
import sys
import logging
import time
import argparse
import numpy as np
from collections import OrderedDict

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
import scipy.io as sio
import torch
import torch.nn as nn

# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
opt = option.dict_to_nonedict(opt)
# Create model
model = create_model(opt)

print(model.netG_Y)
print(model.netG_UV)

if isinstance(model.netG_Y, nn.DataParallel):
    model.netG_Y = model.netG_Y.module

if isinstance(model.netG_UV, nn.DataParallel):
    model.netG_UV = model.netG_UV.module

pretrain_model_G_Y=opt['path']['pretrain_model_G_Y']
load_path_G_UV = opt['path']['pretrain_model_G_UV']
print(pretrain_model_G_Y, load_path_G_UV)
pretrain_model_G_Y=os.path.basename(pretrain_model_G_Y)
load_path_G_UV=os.path.basename(load_path_G_UV)
torch.save(model.netG_Y, pretrain_model_G_Y[:-4]+"_s.pth")
torch.save(model.netG_UV, load_path_G_UV[:-4]+"_s.pth")

print("done")