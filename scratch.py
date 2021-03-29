from shutil import Error
from typing import Counter
import torch
import pathlib
import brevitas
from model.quant_vgg import QuantVGG

conv2d_counter = 0
maxpool2d_counter = 0
quantrelue_counter = 0
fullyconn_counter=0
pre_layer=None
def quant_conv2d_parser(layer, file):
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_K"), layer.kernel_size))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_IFM_CH"), layer.in_channels))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_STRIDE"), layer.stride))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_PADDING"), layer.padding))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_OFM_CH"), layer.out_channels))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_OFM_DIM"), "((CONV2D_{0}_IFM_DIM - CONV2D_{0}_K + 2 * CONV2D_{0}_PADDING) / CONV2D_{0}_STRIDE + 1)".format(conv2d_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_PE"), "1"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_SIMD"), "1"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_BIAS_BITS"), layer.quant_bias_bit_width() if layer.quant_bias_bit_width() is not None else "0"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_BIAS_INT_BITS"), ))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_WEIGHT_BITS"), layer.quant_weight_bit_width()))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_WEIGHT_INT_BITS"), "1!"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_IA_BITS"), "{}OA_BITS".format(pre_layer) if conv2d_counter > 0 else "8!" ))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_IA_INT_BITS"), "{}OA_INT_BITS".format(pre_layer) if conv2d_counter > 0 else "4!" ))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_IFM_DIM"), "{}OFM_DIM".format(pre_layer) if conv2d_counter > 0 else "(SEQUENCE_LENGTH / CONV1D_0_IFM_CH)"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_MUL_BITS"), "16!"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_MUL_INT_BITS"), "8!"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_ACC_BITS"), "16!"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_ACC_INT_BITS"), "8!"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_OA_BITS"), "16!"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_OA_INT_BITS"), "8!"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_BMEM"), "(CONV2D_{0}_OFM_CH / CONV2D_{0}_PE)".format(conv2d_counter)))
    file.write("{:<48}{}\n\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_WMEM"), "((1 * CONV2D_{0}_K * CONV2D_{0}_IFM_CH * CONV2D_{0}_OFM_CH) / (CONV2D_{0}_PE * CONV2D_{0}_SIMD))".format(conv2d_counter)))
    
def maxpool2d_parser(layer, file):
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_K"), layer.kernel_size))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_IFM_CH"), "{}OFM_CH".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_IFM_DIM"), "{}OFM_DIM".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_STRIDE"), "MAXPOOL2D_{}_K".format(maxpool2d_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_PADDING"), layer.padding))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_OFM_CH"), "MAXPOOL2D_{}_IFM_CH".format(maxpool2d_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_OFM_DIM"), "(MAXPOOL2D_{0}_IFM_DIM / MAXPOOL2D_{0}_K)".format(maxpool2d_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_SIMD"), "1"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_IA_BITS"), "{}OA_BITS".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_IA_INT_BITS"), "{}OA_INT_BITS".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_OA_BITS"), "MAXPOOL2D_{}_IA_BITS".format(maxpool2d_counter)))
    file.write("{:<48}{}\n\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_OA_INT_BITS"), "MAXPOOL2D_{}_IA_INT_BITS".format(maxpool2d_counter)))

def quantReLU_parser(layer, file):
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelue_counter, "_IFM_CH"), "{}OFM_CH".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelue_counter, "_IFM_DIM"), "{}OFM_DIM".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelue_counter, "_OFM_CH"), "RELU_{}_IFM_CH".format(quantrelue_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelue_counter, "_OFM_DIM"), "RELU_{}_IFM_DIM".format(quantrelue_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelue_counter, "_SIMD"), "1"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelue_counter, "_IA_BITS"), "{}OA_BITS".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelue_counter, "_IA_INT_BITS"), "{}OA_INT_BITS".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelue_counter, "_OA_BITS"), "999"))
    file.write("{:<48}{}\n\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelue_counter, "_OA_INT_BITS"), "999"))

def fullyconn_parser(layer,file):
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_IFM_CH"), "({}OFM_CH)".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_IFM_DIM"), "1!" if fullyconn_counter > 0 else "({}OFM_DIM".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_IA_BITS"), "{}OA_BITS".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_IA_INT_BITS"), "{}OA_INT_BITS".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_OFM_CH"), layer.out_channels))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_PE"), "CONV2D_{}_OFM_CH".format(fullyconn_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_SIMD"), "CONV2D_{}_OFM_CH".format(fullyconn_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_BIAS_BITS"), "CONV2D_{}_OFM_CH".format(fullyconn_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_BIAS_INT_BITS"), "CONV2D_{}_OFM_CH".format(fullyconn_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_WEIGHT_BITS"), "CONV2D_{}_OFM_CH".format(fullyconn_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_WEIGHT_INT_BITS"), "CONV2D_{}_OFM_CH".format(fullyconn_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_MUL_BITS"), "CONV2D_{}_OFM_CH".format(fullyconn_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_MUL_INT_BITS"), "CONV2D_{}_OFM_CH".format(fullyconn_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_ACC_BITS"), "CONV2D_{}_OFM_CH".format(fullyconn_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_ACC_INT_BITS"), "CONV2D_{}_OFM_CH".format(fullyconn_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_OA_BITS"), "CONV2D_{}_OFM_CH".format(fullyconn_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_OA_INT_BITS"), "CONV2D_{}_OFM_CH".format(fullyconn_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_BMEM"), "CONV2D_{}_OFM_CH".format(fullyconn_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_WMEM"), "CONV2D_{}_OFM_CH".format(fullyconn_counter)))


def main():
    pth_path = "/home/habi/TUK/HIWI/Hiwi_Task1/saved/models/checkpoint-epoch150.pth"
    model=QuantVGG(VGG_type='A', batch_norm=True, bit_width=8, num_classes=1000)
    try:
        model.load_state_dict(torch.load(pth_path)['state_dict'])
    except RuntimeError:
         print("RuntimeError")
    model.eval()
    model(torch.randn(1,3,32,32))
    with open('config.h', 'w') as file_object:
        global conv2d_counter
        global maxpool2d_counter
        global quantrelue_counter
        global fullyconn_counter
        global pre_layer
        
        file_object.write("#ifndef CONFIG_H_\n#define CONFIG_H_\n\n")
        # Extract Features layers Data
        for i in model.features:
            if isinstance(i,brevitas.nn.quant_conv.QuantConv2d):
                quant_conv2d_parser(i, file_object)
                pre_layer="CONV2D_{}_".format(conv2d_counter)
                conv2d_counter += 1
            elif isinstance(i,torch.nn.modules.pooling.MaxPool2d):
                maxpool2d_parser(i, file_object)
                pre_layer="MAXPOOL2D_{}_".format(maxpool2d_counter)
                maxpool2d_counter += 1
            elif isinstance(i,brevitas.nn.quant_activation.QuantReLU):
                quantReLU_parser(i,file_object)
                pre_layer="RELU_{}_".format(quantrelue_counter)
                quantrelue_counter+=1
        # Extract classifier layers Data
        for i in model.classifier:
            if isinstance(i,brevitas.nn.QuantLinear):
                fullyconn_parser(i, file_object)
                pre_layer="FC_{}_".format(fullyconn_counter)
                fullyconn_counter += 1
            elif isinstance(i,brevitas.nn.quant_activation.QuantReLU):
                quantReLU_parser(i,file_object)
                pre_layer="RELU_{}_".format(quantrelue_counter)
                quantrelue_counter+=1
        # Extract Conv layers Weight & Bias   
        conv2d_counter=0
        for i in model.features:
            if isinstance(i,brevitas.nn.quant_conv.QuantConv2d):
                file_object.write("QuantConv2d_{}_WEIGHT\n".format(conv2d_counter))
                file_object.write(str(i.weight.data))
                if i.bias is not None:
                    file_object.write("QuantConv2d_{}_BIAS\n".format(conv2d_counter))
                    file_object.write(str(i.bias.data))
                conv2d_counter += 1

        # Extract Fully Connected layers Weight & Bias   
        fullyconn_counter=0
        for i in model.features:
            if isinstance(i,brevitas.nn.quant_conv.QuantConv2d):
                file_object.write("FullyConnected_{}_WEIGHT\n".format(fullyconn_counter))
                file_object.write(str(i.weight.data))
                file_object.write("\n")
                if i.bias is not None:
                    file_object.write("FullyConnected_{}_BIAS\n".format(fullyconn_counter))
                    file_object.write(str(i.bias.data))
                    file_object.write("\n")
                fullyconn_counter += 1

    print("Result:")
    print(str(pathlib.Path(__file__).parent.absolute())+"/config.h")


if __name__ == '__main__':
    main()