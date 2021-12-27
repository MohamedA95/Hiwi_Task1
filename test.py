import argparse
import collections
import torch
from torchinfo import summary
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model as module_arch
from parse_config import ConfigParser
from utils import parameters_extractor,str2bool,get_logger


def main(config):
    logger = get_logger(name=__name__,log_dir=config.log_dir,test=True,verbosity=config['trainer']['verbosity'])
    # setup data_loader instances
    data_loader_obj = config.init_obj('test_data_loader', module_data)
    test_data_loader = data_loader_obj.get_test_loader()
    # build model architecture
    model = config.init_obj('arch', module_arch)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    if not torch.cuda.is_available():
        checkpoint = torch.load(config.resume,map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    # torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict,"module.")
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    summary(model,input_size=[config['data_loader']['args']['batch_size']]+ config['input_size'])
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_data_loader)):
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(test_data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    if 'extract' in config._config:
        if str2bool(config['extract']):
            logger.info("Extracting Parameters\n")
            logger.info(parameters_extractor(model,config['extractor'],config.log_dir))
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target help')
    options = [CustomArgs(['-x', '--extract'], type=str, target=('extract'), help='extract parameters of the model (default: False)')]
    config = ConfigParser.from_args(args,options=options,test=True)
    main(config)
