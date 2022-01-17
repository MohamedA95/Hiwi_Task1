import argparse
import collections
import copy

import brevitas
import torch
import torch.nn as nn
from brevitas.nn.utils import merge_bn
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torchinfo import summary
from tqdm import tqdm
import torch.distributed as dist
import data_loader.data_loaders as module_data
import model as module_arch
import model.loss as module_loss
import model.metric as module_metric
from parse_config import ConfigParser
from trainer import Trainer

'''
fusion_tester
The goal of this file is to compare a model's accuracy on test dataset before and after fusion.
The main function would define two models the origianl model with batchnorm=True and a fused_model with batchnorm=False
Then it will iterate through model.features, fusing BatchNorm2d as it goes. After fusing it will do a copy.deepcopy()
to copy the fused layer from model to fusedmodel. Other layer are copied directly. model.classifer is also copied directly as it is.
Fusion can be done by two functions brevitas.nn.utils.merge_bn (Gives lower accuracy bias is calculated differently from PyTorch's implementation),
torch.nn.utils.fusion.fuse_conv_bn_eval (Gives better accuracy)
'''


def main(config):
    # build model architecture
    model = config.init_obj('arch', module_arch)
    fused_model = config.init_obj('arch', module_arch, batchnorm=False)
    print.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
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
                i=fuse_conv_bn_eval(i,bn)
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
    print("Finished fusing, Strart training")
    torch.backends.cudnn.benchmark = True
    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    dist.init_process_group(backend='nccl',init_method='tcp://127.0.0.1:34567', world_size=1, rank=0) # Rank here is the process rank amoung all processes on all nodes, needs modification in case of multi node
    # setup data_loader instances
    config.config['data_loader']['args']['batch_size']//=config['n_gpu']  #Needs modification to support multinode
    config.config['data_loader']['args']['num_workers']//=config['n_gpu'] #Needs modification to support multinode
    data_loader_obj = config.init_obj('data_loader', module_data)
    data_loader = data_loader_obj.get_train_loader()
    valid_data_loader = data_loader_obj.get_valid_loader()
    torch.cuda.device(0)
    fused_model.cuda(0)
    fused_model = torch.nn.parallel.DistributedDataParallel(fused_model, device_ids=[0],find_unused_parameters=True)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    summary(model,input_size=[config['data_loader']['args']['batch_size']]+config['input_size'])
    print('Trainable parameters: {}'.format(sum([p.numel() for p in trainable_params])))
    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizer = config.init_obj('optimizer', torch.optim, model.parameters())
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    config.resume=
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=0,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      train_sampler=data_loader_obj.train_sampler)

    trainer.train()

    # prepare model for testing
    print("Finished Training, Start testing")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # setup data_loader instances
    test_data_loader = config.init_obj('test_data_loader', module_data).get_test_loader()
    print("Testing unfused model")
    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_data_loader)):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(test_data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__: total_metrics[i].item(
    ) / n_samples for i, met in enumerate(metric_fns)})
    print(log)

    print("Testing unfused model")
    print(summary(fused_model, input_size=[
                config['data_loader']['args']['batch_size']] + config['input_size'], verbose=0))
    fused_model = fused_model.to(device)
    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_data_loader)):
            data = data.to(device)
            target = target.to(device)
            output = fused_model(data)
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(test_data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__: total_metrics[i].item(
    ) / n_samples for i, met in enumerate(metric_fns)})
    print(log)

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
        CustomArgs(['--ep', '--epochs'], type=float, target='optimizer;args;lr', help=""),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size', help="")
    ]
    config = ConfigParser.from_args(args, test=True)
    main(config)
