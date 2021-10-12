import argparse
import collections
import torch
import numpy as np
import random
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model as module_arch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torchinfo import summary
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device, get_CIFAR_data


def main(config):
    
    logger = config.get_logger('train')
    if config['n_gpu'] == -1:
        config['n_gpu']= torch.cuda.device_count()
    
    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(config['seed'])
        random.seed(config['seed'])
        logger.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    if config['data_loader']['type'] == 'CIFAR_data_loader':
        dataset=get_CIFAR_data(config['data_loader']['args']['data_dir'],
                               training=config['data_loader']['args']['training'],
                               download=config['data_loader']['args']['download'],
                               flavor=config['data_loader']['args']['flavor'])
    print(len(dataset))
    print(dataset)
    exit()
    mp.spawn(main_worker, nprocs=config['n_gpu'], args=(config))

def main_worker(rank,config)
    logger.info("Using GPU: {} for training" )
    init_process(rank,config['n_gpu'])
    logger.info(rank,"/",config['n_gpu']," process initialized")
    exit()
    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    logger.info(config['name'])
    summary(model,input_size=(config['data_loader']['args']['batch_size'], 3, 224, 224))
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()

def init_process(rank, size, backend='ncll'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target help')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr', help=""),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size', help="")
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
