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
from utils import prepare_device,get_logger


def main(config):
    
    logger = get_logger(name=__name__,log_dir=config.log_dir,verbosity=config['trainer']['verbosity'])
    if config['n_gpu'] == -1:
        config.config['n_gpu']= torch.cuda.device_count()
    
    torch.backends.cudnn.benchmark = True
    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        np.random.seed(config['seed'])
        random.seed(config['seed'])
        logger.warning('You seeded the training. '
                      'This turns on the CUDNN deterministic setting, '
                      'which can slow down your training '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    mp.spawn(main_worker, nprocs=config['n_gpu'], args=(config,))

def main_worker(gpu,config):
    logger = get_logger('Worker{}'.format(gpu),log_dir=config.log_dir,verbosity=config['trainer']['verbosity'])
    logger.info('Using GPU: {} for training'.format(gpu))
    dist.init_process_group(backend=config['dist_backend'],init_method=config['dist_url'], world_size=config['n_gpu'], rank=gpu) # Rank here is the process rank amoung all processes on all nodes, needs modification in case of multi node
    # setup data_loader instances
    config.config['data_loader']['args']['batch_size']//=config['n_gpu']  #Needs modification to support multinode
    config.config['data_loader']['args']['num_workers']//=config['n_gpu'] #Needs modification to support multinode
    data_loader_obj = config.init_obj('data_loader', module_data)
    data_loader = data_loader_obj.get_train_loader()
    valid_data_loader = data_loader_obj.get_valid_loader()
    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    # prepare for (multi-device) GPU training
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu],find_unused_parameters=True)
    if gpu==0:
        logger.info(config['name'])
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        summary(model,input_size=(config['data_loader']['args']['batch_size'], 3, 224, 224))
        logger.info('Trainable parameters: {}'.format(sum([p.numel() for p in trainable_params])))
    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizer = config.init_obj('optimizer', torch.optim, model.parameters())
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=gpu,
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

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target help')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr', help=""),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size', help="")
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
