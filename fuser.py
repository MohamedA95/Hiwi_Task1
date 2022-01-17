import argparse
import collections
import copy

import brevitas
import torch
import torch.distributed as dist
import torch.nn as nn
from brevitas.nn.utils import merge_bn
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torchinfo import summary
from tqdm import tqdm

import data_loader.data_loaders as module_data
import model as module_arch
import model.loss as module_loss
import model.metric as module_metric
from parse_config import ConfigParser
from trainer import Trainer
from utils import read_json

'''
 fuser.py
 This file takes a trained quantized model spicified via -r option and fuses nn.BatchNorm2d layer into preceding QuantConv2d layers.
 After the fusion is done, it traines the model for a number of epochs spicified by --ep option. 
 Finally the fused model is saved with a full folder hirerchy next to the old experiment folder, with '_fused' added to it's name.
'''


def main(config):
    # build model architecture
    model = config.init_obj('arch', module_arch)
    fused_model = config.init_obj('arch', module_arch, batchnorm=False)
    print('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        state_dict, "module.")
    model.load_state_dict(state_dict)
    model.eval()
    fused_model.eval()
    # Start Fusing
    features_iter = model.features.children()
    fusedindex = 0
    with tqdm(total=len(model.features), desc='Fusing the model') as pbar:
        i = next(features_iter, None)
        while i != None:
            if isinstance(i, brevitas.nn.quant_conv.QuantConv2d) or isinstance(i, torch.nn.modules.conv.Conv2d):
                bn = next(features_iter)
                assert isinstance(
                    bn, nn.BatchNorm2d), "The layer after QuantConv2d is not nn.BatchNorm2d"
                i = fuse_conv_bn_eval(i, bn)
                # merge_bn(i, bn)
                fused_model.features[fusedindex] = copy.deepcopy(i)
                fusedindex += 1
                pbar.update()
            elif isinstance(i, torch.nn.modules.pooling.MaxPool2d):
                fused_model.features[fusedindex] = copy.deepcopy(i)
                fusedindex += 1
            elif isinstance(i, brevitas.nn.quant_activation.QuantReLU) or isinstance(i, torch.nn.modules.activation.ReLU):
                fused_model.features[fusedindex] = copy.deepcopy(i)
                fusedindex += 1
            else:
                print(
                    "Faced an Unknown layer: {0} \n Exiting...".format(type(i)))
                exit()
            i = next(features_iter, None)
            pbar.update()
    fused_model.classifier = copy.deepcopy(model.classifier)
    # Start training
    print("Finished fusing, Strarting training...")
    json_config = read_json(config.resume.parent / 'config.json')
    json_config['name'] += '_fused'
    json_config['arch']['args']['batchnorm'] = False
    json_config['trainer']['save_dir'] = '/'.join(
        json_config['trainer']['save_dir'].split('/')[:-1])
    json_config['trainer']['epochs'] = config['trainer']['epochs']
    json_config['data_loader']['args']['batch_size'] = config['data_loader']['args']['batch_size']
    new_config = ConfigParser(json_config)
    torch.backends.cudnn.benchmark = True
    # get function handles of loss and metrics
    criterion = getattr(module_loss, new_config['loss'])
    metrics = [getattr(module_metric, met) for met in new_config['metrics']]
    dist.init_process_group(
        backend='nccl', init_method='tcp://127.0.0.1:34567', world_size=1, rank=0)
    # setup data_loader instances
    new_config.config['data_loader']['args']['batch_size'] //= new_config['n_gpu']
    new_config.config['data_loader']['args']['num_workers'] //= new_config['n_gpu']
    data_loader_obj = new_config.init_obj('data_loader', module_data)
    data_loader = data_loader_obj.get_train_loader()
    valid_data_loader = data_loader_obj.get_valid_loader()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.device(0)
    fused_model.to(device)
    fused_model = torch.nn.parallel.DistributedDataParallel(
        fused_model, find_unused_parameters=True)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    summary(fused_model, input_size=[
            new_config['data_loader']['args']['batch_size']]+new_config['input_size'])
    print('Trainable parameters: {}'.format(
        sum([p.numel() for p in trainable_params])))
    # build optimizer, learning rate scheduler.
    optimizer = new_config.init_obj(
        'optimizer', torch.optim, model.parameters())
    lr_scheduler = new_config.init_obj(
        'lr_scheduler', torch.optim.lr_scheduler, optimizer)
    trainer = Trainer(fused_model, criterion, metrics, optimizer,
                      config=new_config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      train_sampler=data_loader_obj.train_sampler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target help')
    options = [
        CustomArgs(['--ep', '--epochs'], type=int,
                   target='trainer;epochs', help=""),
        CustomArgs(['--bs', '--batch_size'], type=int,
                   target='data_loader;args;batch_size', help="")
    ]
    config = ConfigParser.from_args(args, options, test=True)
    main(config)
