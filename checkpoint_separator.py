import argparse
from pathlib import Path

import torch


"""
 Takes a check point file created by the project & separats the weights "state dict" 
  to separate file. The check point pth file cant be use without the project structure.
"""


def main(model_path):
    print('Loading checkpoint: {}'.format(model_path))
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['state_dict']
    res_path = Path(model_path).parent.joinpath("state_dict")
    torch.save(state_dict, res_path)
    print("Result:\n", res_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Checkpint Separator')
    args.add_argument('-m', '--model', default=None,
                      type=str, help='path to model')
    args = args.parse_args()
    main(args.model)
