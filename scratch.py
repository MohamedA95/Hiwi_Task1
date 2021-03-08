import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from model.quant_vgg import QuantVGG
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device

model=QuantVGG()
model.load_state_dict(torch.load("/home/habi/TUK/HIWI/model_best.pth"))
sequantial_container = next(model.children())
for i in sequantial_container:
    print(dir(i))
    print(type(i))
    exit()
print(model)