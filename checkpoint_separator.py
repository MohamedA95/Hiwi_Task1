import argparse
import torch
import model as module_arch
from parse_config import ConfigParser

"""
 Takes a check point file created by the project & separats the weights "state dict" 
  to separate file. The check point pth file cant be use without the project structure.
"""


def main(config):
    logger = config.get_logger('Checkpint Separator')
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    res_path = config.resume.parent.joinpath("state_dict")
    torch.save(state_dict, res_path)
    print("Result:\n", res_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-m', '--model', default=None,
                      type=str, help='path to model')
    config = ConfigParser.from_args(args)
    main(config)
