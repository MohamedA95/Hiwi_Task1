from shutil import Error
import torch
import pathlib
import brevitas
from model.quant_vgg import QuantVGG


def quant_conv2d_parser(layer, file, counter):
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_K"), layer.kernel_size))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_IFM_CH"), layer.in_channels))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_STRIDE"), layer.stride))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_PADDING"), layer.padding))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_OFM_CH"), layer.out_channels))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_OFM_DIM"), "((CONV2D_1_IFM_DIM - CONV2D_1_K + 2 * CONV2D_1_PADDING) / CONV2D_1_STRIDE + 1) "))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_PE"), "1"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_SIMD"), "1"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_BIAS_BITS"), layer.quant_bias_bit_width() if layer.quant_bias_bit_width() is not None else "0"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_BIAS_INT_BITS"), "9999999999999999"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_WEIGHT_BITS"), layer.quant_weight_bit_width()))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_WEIGHT_INT_BITS"), "9999999999999999"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_IA_BITS"), "RELU_{}_OA_BITS".format(counter-1)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_IA_INT_BITS"), "RELU_{}_OA_INT_BITS".format(counter-1) if counter > 0 else 0 ))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_IFM_DIM"), "RELU_{}_OFM_DIM".format(counter-1) if counter > 0 else "(SEQUENCE_LENGTH / CONV1D_0_IFM_CH)"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_MUL_BITS"), "9999999999999999"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_MUL_INT_BITS"), "9999999999999999"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_ACC_BITS"), "9999999999999999"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_ACC_INT_BITS"), "9999999999999999"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_OA_BITS"), "9999999999999999"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_OA_INT_BITS"), "9999999999999999"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_BMEM"), "(CONV2D_1_OFM_CH / CONV2D_1_PE)"))
    file.write("{:<48}{}\n\n".format("{:s}{:d}{:s}".format("#define CONV2D_", counter, "_WMEM"), "((1 * CONV2D_1_K * CONV2D_1_IFM_CH * CONV2D_1_OFM_CH) / (CONV2D_1_PE * CONV2D_1_SIMD)) "))
    
def maxpool2d_parser(maxpool2d, file, counter):
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", counter, "_K"), str(maxpool2d.kernel_size)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", counter, "_IFM_CH"), "RELU_{}_OFM_CH".format(counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", counter, "_IFM_DIM"), "RELU_{}_OFM_DIM".format(counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", counter, "_STRIDE"), "MAXPOOL_{}_K".format(counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", counter, "_PADDING"), str(maxpool2d.padding)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", counter, "_OFM_CH"), "MAXPOOL_{}_IFM_CH".format(counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", counter, "_OFM_CH"), "(MAXPOOL_{}_IFM_DIM / MAXPOOL_{}_K)".format(counter,counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", counter, "_SIMD"), "1"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", counter, "_IA_BITS"), "RELU_{}_OA_BITS".format(counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", counter, "_IA_INT_BITS"), "RELU_{}_OA_INT_BITS".format(counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", counter, "_OA_BITS"), "MAXPOOL_{}_IA_BITS".format(counter)))
    file.write("{:<48}{}\n\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", counter, "_OA_INT_BITS"), "MAXPOOL_{}_IA_INT_BITS".format(counter)))

def quantReLU_parser(layer, file, counter):
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", counter, "_IFM_CH"), "CONV2D_{}_OFM_CH".format(counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", counter, "_IFM_DIM"), "CONV2D_{}_OFM_DIM".format(counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", counter, "_OFM_CH"), "RELU_{}_IFM_CH".format(counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", counter, "_OFM_DIM"), "RELU_{}_IFM_DIM".format(counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", counter, "_SIMD"), "1"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", counter, "_IA_BITS"), "CONV2D_{}_OA_BITS".format(counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", counter, "_IA_INT_BITS"), "CONV2D_{}_OA_INT_BITS".format(counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", counter, "_OA_BITS"), "9999999999999999999999999999999"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", counter, "_OA_INT_BITS"), "9999999999999999999999999999999"))

def main():
    pth_path = "/home/habi/TUK/HIWI/Hiwi_Task1/saved/models/checkpoint-epoch150.pth"
    model=QuantVGG(VGG_type='A', batch_norm=True, bit_width=8, num_classes=1000)
    try:
        model.load_state_dict(torch.load(pth_path)['state_dict'])
    except RuntimeError:
         print("RuntimeError")
    # print("model.children()")
    # for i in model.children():
    #     print("->"+str(type(i)))
    #     try:
    #         for j in i:
    #             print(type(j))
    #     except:
    #         print("Error")
    #         pass            
    # print("---------------------------------------------------")
    # print("model.modules()")
    # for i in model.modules():
    #     print(type(i))
    # print("---------------------------------------------------")
    # print("model.parameters()")
    # for i in model.parameters():
    #     print(type(i))
    # exit()
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