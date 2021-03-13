import torch
import pathlib
import brevitas
from model.quant_vgg import QuantVGG


def quant_conv2d_parser(layer, file, counter):
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_K"), layer.kernel_size))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_IN_CH"), layer.in_channels))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_STRIDE"), layer.stride))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_PADDING"), layer.padding))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_OUT_CH"), layer.out_channels))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_OFM_DIM"), layer.output_channel_dim))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_OFM_DIM"), "((CONV1D_1_IFM_DIM - CONV1D_1_K + 2 * CONV1D_1_PADDING) / CONV1D_1_STRIDE + 1) "))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_PE"), "PE"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_BIAS_BITS"), "layer.quant_bias_bit_width"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_BIAS_INT_BITS"), "layer.quant_input_bit_width"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_WEIGHT_BITS"), "layer.quant_weight_bit_width"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_WEIGHT_INT_BITS"), "layer.int_weight"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_WEIGHT_INT_BITS"), "layer.quant_weight"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_WEIGHT_INT_BITS"), "layer.int_bias"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_IA_BITS"), "RELU_{}_OA_BITS".format(counter-1)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_IA_INT_BITS"), "RELU_{}_OA_INT_BITS".format(counter-1)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_IFM_DIM"), "RELU_{}_OFM_DIM".format(counter-1)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_MUL_BITS"), "9999999999999999"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_MUL_INT_BITS"), "9999999999999999"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_ACC_BITS"), "max_acc_bit_width()"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_ACC_INT_BITS"), "9999999999999999"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_OA_BITS"), "9999999999999999"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_OA_INT_BITS"), "9999999999999999"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_BMEM"), "(CONV1D_1_OFM_CH / CONV1D_1_PE)"))
    file.write("{:<48}{}\n\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_WMEM"), "((1 * CONV1D_1_K * CONV1D_1_IFM_CH * CONV1D_1_OFM_CH) / (CONV1D_1_PE * CONV1D_1_SIMD)) "))
    
def maxpool2d_parser(maxpool2d, file, counter):
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", counter, "_K"), str(maxpool2d.kernel_size)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", counter, "_STRIDE"), str(maxpool2d.stride)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", counter, "_PADDING"), str(maxpool2d.padding)))
    file.write("{:<48}{}\n\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", counter, "_DILATION"), str(maxpool2d.dilation)))

def quantReLU_parser(layer, file, counter):
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", counter, "_IFM_CH"), "CONV1D_0_OFM_CH"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", counter, "_IFM_DIM"), "CONV1D_0_OFM_DIM"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", counter, "_OFM_CH"), "RELU_0_IFM_CH"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", counter, "_OFM_DIM"), "RELU_0_IFM_DIM"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", counter, "_SIMD"), "9999999999999999999999999999999"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", counter, "_IA_BITS"), "CONV1D_0_OA_BITS"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", counter, "_IA_INT_BITS"), "CONV1D_0_OA_INT_BITS"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", counter, "_OA_BITS"), "9999999999999999999999999999999"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", counter, "_OA_INT_BITS"), "9999999999999999999999999999999"))

def main():
    pth_path = "/home/habi/TUK/HIWI/checkpoint-epoch5.pth"
    model=QuantVGG(VGG_type='A', batch_norm=True, bit_width=8, num_classes=1000)
    try:
        model.load_state_dict(torch.load(pth_path)['state_dict'])
    except RuntimeError:
         print("RuntimeError")
    sequantial_container = next(model.children())
    conv2d_counter = 0
    maxpool2d_counter = 0
    with open('config.h', 'w') as file_object:
        file_object.write("#ifndef CONFIG_H_\n#define CONFIG_H_\n\n")
        for i in sequantial_container:
            print(type(i))
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