import argparse
from utils.extractor import parameters_extractor
import torch
import model as module_arch
from pathlib import Path
from parse_config import ConfigParser
from utils import read_json

"""
 Extracts layers properites, weight & bias and writes the result to .h file  with the model name
    under the same path 
"""


def main(model_path,config):
    # build model architecture
    model = getattr(module_arch,config['arch']['type'])(**dict(config['arch']['args']))
    print('Loading checkpoint: {}'.format(model_path))
    checkpoint = torch.load(model_path,map_location='cpu')
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    data=torch.randn(1,3, 224, 224)
    device = torch.device('cpu') #'cuda' if torch.cuda.is_available() else 
    model = model.to(device)
    model.eval()
    data = data.to(device)
    model(data)
    res_path = parameters_extractor(model,config['extractor'],result_path=model_path.parent)
    print("Result:\n", res_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Parameters_extractor')
    args.add_argument('-m', '--model', default=None, type=str, help='path to model')
    args = args.parse_args()
    model_path= Path(args.model)
    config_path = model_path.parent.joinpath('config.json')
    config = read_json(config_path)
    main(model_path,config)
