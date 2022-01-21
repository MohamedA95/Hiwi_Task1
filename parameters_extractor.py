import argparse
from utils.extractor import parameters_extractor
import torch
import model as module_arch
from pathlib import Path
from utils import read_json,str2bool,consume_prefix_in_state_dict_if_present

"""
 Extracts layers properites, weight & bias and writes the result to .h file  with the model name
    under the same path 
"""


def main(args):
    # build model architecture
    model_path= Path(args.model)
    config_path = model_path.parent.joinpath('config.json')
    config = read_json(config_path)
    model = getattr(module_arch,config['arch']['type'])(**dict(config['arch']['args']))
    print('Loading checkpoint: {}'.format(model_path))
    checkpoint = torch.load(model_path,map_location='cpu')
    state_dict = checkpoint['state_dict']
    consume_prefix_in_state_dict_if_present(state_dict,"module.")
    model.load_state_dict(state_dict)
    data=torch.randn([1]+config['input_size'])
    device = torch.device('cpu') #'cuda' if torch.cuda.is_available() else 
    model.eval()
    model = model.to(device)
    data = data.to(device)
    model(data)
    res_path = parameters_extractor(model,config['extractor'],result_path=model_path.parent,fuse=args.fuse)
    print("Result:\n", res_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Parameters_extractor')
    args.add_argument('-m', '--model', default=None, type=str, help='Path to model')
    args.add_argument('-f', '--fuse', default=False,type=str2bool,const=True,nargs='?', help='If set will fuse BatchNorm2d to Conv2d')
    args = args.parse_args()
    main(args)
