import torch
import pathlib
import brevitas
from model.quant_vgg import QuantVGG


def quant_conv2d_parser(layer, file, counter):
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_K"), layer.kernel_size))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_STRIDE"), layer.stride))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_IN_CH"), layer.in_channels))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_OUT_CH"), layer.out_channels))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_PADDING"), layer.padding))
    file.write("{:<48}{}\n\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_DILATION"), layer.dilation))


def maxpool2d_parser(maxpool2d, file, counter):
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", counter, "_K"), str(maxpool2d.kernel_size)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", counter, "_STRIDE"), str(maxpool2d.stride)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", counter, "_PADDING"), str(maxpool2d.padding)))
    file.write("{:<48}{}\n\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", counter, "_DILATION"), str(maxpool2d.dilation)))

def quantReLU_parser(layer, file, counter):
    pass

def main():
    pth_path = "/home/habi/TUK/HIWI/checkpoint-epoch5.pth"
    model=QuantVGG()
    try:
        model.load_state_dict(torch.load(pth_path))
    except RuntimeError:
        print("RuntimeError")
    sequantial_container = next(model.children())
    conv2d_counter = 0
    maxpool2d_counter = 0
    with open('config.h', 'w') as file_object:
        file_object.write("#ifndef CONFIG_H_\n#define CONFIG_H_\n\n")
        for i in sequantial_container:
            if i.__class__ == brevitas.nn.quant_conv.QuantConv2d:
                quant_conv2d_parser(i, file_object, conv2d_counter)
                conv2d_counter += 1
            elif i.__class__ == torch.nn.modules.pooling.MaxPool2d:
                maxpool2d_parser(i, file_object, maxpool2d_counter)
                maxpool2d_counter += 1
            elif i.__class__ == brevitas.nn.quant_activation.QuantReLU:
                pass

    print("Result:")
    print(str(pathlib.Path(__file__).parent.absolute())+"/config.h")


if __name__ == '__main__':
    main()