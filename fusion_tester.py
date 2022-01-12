import argparse
import copy

import brevitas
import torch
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
from utils import get_logger

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
    logger = get_logger(name='Fuser:', log_dir=config.log_dir,
                        test=True, verbosity=config['trainer']['verbosity'])
    # setup data_loader instances
    data_loader_obj = config.init_obj('test_data_loader', module_data)
    test_data_loader = data_loader_obj.get_test_loader()
    # build model architecture
    model = config.init_obj('arch', module_arch)
    fused_model = config.init_obj('arch', module_arch, batchnorm=False)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    if not torch.cuda.is_available():
        checkpoint = torch.load(
            config.resume, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        state_dict, "module.")
    model.load_state_dict(state_dict)

    model.eval()
    fused_model.eval()
    features_iter = model.features.children()
    fusedindex = 0
    with tqdm(total=len(model.features), desc='Fusing the model') as pbar:
        i = next(features_iter, None)
        while i != None:
            if isinstance(i, brevitas.nn.quant_conv.QuantConv2d) or isinstance(i, torch.nn.modules.conv.Conv2d):
                bn = next(features_iter)
                assert isinstance(
                    bn, nn.BatchNorm2d), "The layer after QuantConv2d is not nn.BatchNorm2d"
                # i=fuse_conv_bn_eval(i,bn)
                merge_bn(i, bn)
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

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info("Testing unfused model")
    logger.info(summary(model, input_size=[
                config['data_loader']['args']['batch_size']] + config['input_size'], verbose=0))
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
    logger.info(log)

    logger.info("Testing unfused model")
    logger.info(summary(fused_model, input_size=[
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
    logger.info(log)

    # Currently saving the quatized model gives an error
    # _pickle.PicklingError: Can't pickle <class 'brevitas.inject.Int8ActPerTensorFixedPoint'>: attribute lookup Int8ActPerTensorFixedPoint on brevitas.inject failed
    # new_path=str(config.resume)[:-4]+'_fused.pth'
    # torch.save(fused_model.state_dict,new_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser.from_args(args, test=True)
    main(config)
