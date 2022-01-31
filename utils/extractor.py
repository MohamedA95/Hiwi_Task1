import torch
import pathlib
import brevitas
from brevitas.nn.utils import merge_bn
from torch.nn.utils.fusion import fuse_conv_bn_eval
from model.quant_vgg import QuantVGG
from pathlib import Path
from tqdm import tqdm
from .util import bin_digits

FINN_STRUCTURES=False
conv2d_counter = 0
maxpool2d_counter = 0
quantrelu_counter = 0
fullyconn_counter=0
pre_layer=None

def quant_conv2d_parser(layer, file,ext_config):
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_K"), layer.kernel_size[0]))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_IFM_CH"), layer.in_channels))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_STRIDE"), layer.stride[0]))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_PADDING"), layer.padding[0]))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_OFM_CH"), layer.out_channels))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_OFM_DIM"), "(((CONV2D_{0}_IFM_DIM + 2 * CONV2D_{0}_PADDING - 1 * (CONV2D_{0}_K - 1)-1) / CONV2D_{0}_STRIDE) + 1)".format(conv2d_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_PE"), ext_config['PE']))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_SIMD"), ext_config['SIMD']))
    if layer.is_bias_quant_enabled:
        file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_BIAS_BITS"), int(layer.quant_bias_bit_width())))
        file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_BIAS_INT_BITS"), int(layer.quant_bias_bit_width()+torch.log2(layer.quant_bias_scale()))))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_WEIGHT_BITS"), int(layer.quant_weight_bit_width())))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_WEIGHT_INT_BITS"), int(layer.quant_weight_bit_width()+torch.log2(layer.quant_weight_scale()))))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_IA_BITS"), "{}OA_BITS".format(pre_layer) if conv2d_counter > 0 else str(ext_config['IA_BITS']) ))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_IA_INT_BITS"), "{}OA_INT_BITS".format(pre_layer) if conv2d_counter > 0 else str(ext_config['IA_INT_BITS']) ))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_IFM_DIM"), "{}OFM_DIM".format(pre_layer) if conv2d_counter > 0 else "(SEQUENCE_LENGTH / CONV2D_0_IFM_CH)"))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_MUL_BITS"), ext_config['MUL_BITS']))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_MUL_INT_BITS"), ext_config['MUL_INT_BITS']))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_ACC_BITS"), ext_config['ACC_BITS']))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_ACC_INT_BITS"), ext_config['ACC_INT_BITS']))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_OA_BITS"), int(layer.quant_output_bit_width())))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_OA_INT_BITS"), int(layer.quant_output_bit_width()+torch.log2(layer.quant_output_scale()))))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_BMEM"), "(CONV2D_{0}_OFM_CH / CONV2D_{0}_PE)".format(conv2d_counter)))
    file.write("{:<48}{}\n\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_WMEM"), "((CONV2D_{0}_K * CONV2D_{0}_K * CONV2D_{0}_IFM_CH * CONV2D_{0}_OFM_CH) / (CONV2D_{0}_PE * CONV2D_{0}_SIMD))".format(conv2d_counter)))
    
def maxpool2d_parser(layer, file,ext_config):
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_K"), layer.kernel_size))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_IFM_CH"), "{}OFM_CH".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_IFM_DIM"), "{}OFM_DIM".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_STRIDE"), "MAXPOOL2D_{}_K".format(maxpool2d_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_PADDING"), layer.padding))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_OFM_CH"), "MAXPOOL2D_{}_IFM_CH".format(maxpool2d_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_OFM_DIM"), "(((MAXPOOL2D_{0}_IFM_DIM + 2 * MAXPOOL2D_{0}_PADDING - 1 * (MAXPOOL2D_{0}_K - 1)-1) / MAXPOOL2D_{0}_STRIDE) + 1)".format(maxpool2d_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_SIMD"), ext_config['SIMD']))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_IA_BITS"), "{}OA_BITS".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_IA_INT_BITS"), "{}OA_INT_BITS".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_OA_BITS"), "MAXPOOL2D_{}_IA_BITS".format(maxpool2d_counter)))
    file.write("{:<48}{}\n\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_OA_INT_BITS"), "MAXPOOL2D_{}_IA_INT_BITS".format(maxpool2d_counter)))

def quantReLU_parser(layer, file,ext_config):
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelu_counter, "_IFM_CH"), "{}OFM_CH".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelu_counter, "_IFM_DIM"), "{}OFM_DIM".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelu_counter, "_OFM_CH"), "RELU_{}_IFM_CH".format(quantrelu_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelu_counter, "_OFM_DIM"), "RELU_{}_IFM_DIM".format(quantrelu_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelu_counter, "_SIMD"), ext_config['SIMD']))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelu_counter, "_IA_BITS"), "{}OA_BITS".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelu_counter, "_IA_INT_BITS"), "{}OA_INT_BITS".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelu_counter, "_OA_BITS"), int(layer.quant_output_bit_width())))
    file.write("{:<48}{}\n\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelu_counter, "_OA_INT_BITS"), int(layer.quant_output_bit_width()+torch.log2(layer.quant_output_scale()))))

def fullyconn_parser(layer,file,ext_config):
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_IFM_CH"), "{}OFM_CH".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_IFM_DIM"), "1" if fullyconn_counter > 0 else "{}OFM_DIM".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_IA_BITS"), "{}OA_BITS".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_IA_INT_BITS"), "{}OA_INT_BITS".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_OFM_CH"), layer.out_channels))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_PE"), ext_config['PE']))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_SIMD"), ext_config['SIMD']))
    if layer.is_bias_quant_enabled:
        file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_BIAS_BITS"), int(layer.quant_bias_bit_width())))
        file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_BIAS_INT_BITS"), int(layer.quant_bias_bit_width()+torch.log2(layer.quant_bias_scale()))))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_WEIGHT_BITS"), int(layer.quant_weight_bit_width())))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_WEIGHT_INT_BITS"), int(layer.quant_weight_bit_width()+torch.log2(layer.quant_weight_scale()))))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_MUL_BITS"), ext_config['MUL_BITS']))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_MUL_INT_BITS"), ext_config['MUL_INT_BITS']))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_ACC_BITS"), ext_config['ACC_BITS']))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_ACC_INT_BITS"), ext_config['ACC_INT_BITS']))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_OA_BITS"), int(layer.quant_output_bit_width())))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_OA_INT_BITS"), int(layer.quant_output_bit_width()+torch.log2(layer.quant_output_scale()))))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_BMEM"), "(FC_{0}_OFM_CH / FC_{0}_PE)".format(fullyconn_counter)))
    file.write("{:<48}{}\n\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_WMEM"), "((FC_{0}_IFM_DIM * FC_{0}_IFM_DIM * FC_{0}_IFM_CH * FC_{0}_OFM_CH) / (FC_{0}_PE * FC_{0}_SIMD))".format(fullyconn_counter)))

def conv_weight_bias_finn(layer,file_object,weight_bit_width,bias_bit_width):
    file_object.write("static FixedPointWeights<CONV2D_{0}_SIMD, conv2d_{0}_weight_dtype,CONV2D_{0}_PE,CONV2D_{0}_WMEM> conv2d_{0}_weights =\n".format(conv2d_counter))
    file_object.write("{{{\n")
    for j in layer.int_weight():
        for k in j:
            for r in k:
                for m in r: 
                    file_object.write(('"0x{0}", ').format(bin_digits(m,weight_bit_width)))
    file_object.write("}}};\n")
    if layer.is_bias_quant_enabled:
        bias=layer.int_bias()
        str_format='"0x{0}", '
    else:
        bias=layer.bias
        str_format='{:f}, '
    file_object.write("static BiasActivation<CONV2D_{0}_BMEM,CONV2D_{0}_PE, conv2d_{0}_activation_dtype,conv2d_{0}_bias_dtype> conv2d_{0}_bias =\n".format(conv2d_counter))
    file_object.write("{{\n")
    for j in bias:
        file_object.write((str_format).format(bin_digits(j,bias_bit_width)))
    file_object.write("}};\n")

def linear_weight_bias_finn(layer,file_object,weight_bit_width,bias_bit_width):
    file_object.write("static FixedPointWeights<FC_{0}_SIMD, fc_{0}_weight_dtype,FC_{0}_PE,FC_{0}_WMEM> fc_{0}_weights =\n".format(fullyconn_counter))
    file_object.write("{{{\n")
    for row in layer.int_weight():
        for val in row:
            file_object.write(('"0x{0}", ').format(bin_digits(val,weight_bit_width)))
    file_object.write("\n}}};\n")

    if layer.is_bias_quant_enabled:
        bias=layer.int_bias()
        str_format='"0x{0}", '
    else:
        bias=layer.bias
        str_format='{:f}, '
    file_object.write("static BiasActivation<FC_{0}_BMEM,FC_{0}_PE, fc_{0}_activation_dtype,fc_{0}_bias_dtype> fc_{0}_bias =\n".format(fullyconn_counter))
    file_object.write("{{\n")
    for val in bias:
        file_object.write((str_format).format(bin_digits(val,bias_bit_width)))
    file_object.write("\n}};\n")

def conv_weight_bias_array(layer,file_object,weight_bit_width,bias_bit_width):
    file_object.write("static ap_uint<CONV2D_{0}_WEIGHT_BITS> conv2d_{0}_weight [CONV2D_{0}_PE] [CONV2D_{0}_SIMD] [CONV2D_{0}_WMEM] =\n".format(conv2d_counter))
    file_object.write("{\n")
    for j in layer.int_weight():
        file_object.write("{\n")
        for k in j:
            file_object.write("{\n")
            for r in k:
                file_object.write("{")
                for m in r[0:-1]: 
                    file_object.write(("0x{0}, ").format(bin_digits(m,weight_bit_width)))
                file_object.write(("0x{0}}},\n").format(bin_digits(r[-1],weight_bit_width)))
            file_object.write("},\n")
        file_object.write("}\n")
    file_object.write("};\n")


    file_object.write("static ap_uint<CONV2D_{0}_BIAS_BITS> conv2d_{0}_bias [CONV2D_{0}_PE][CONV2D_{0}_BMEM] =\n".format(conv2d_counter))
    file_object.write("{\n{\n")
    if layer.is_bias_quant_enabled:
        for j in layer.int_bias()[0:-1]:
            file_object.write(('0x{0}').format(bin_digits(j,bias_bit_width)))
            file_object.write(",\n")
        file_object.write(('0x{0}').format(bin_digits(layer.int_bias()[-1],bias_bit_width)))
    else:
        for j in layer.bias[0:-1]:
            file_object.write(('{:f}').format(j))
            file_object.write(",\n")
        file_object.write(('{:f}').format(layer.bias[-1]))
    file_object.write("\n}\n};\n")


def linear_weight_bias_array(layer,file_object,weight_bit_width,bias_bit_width):
    file_object.write("static ap_uint<FC_{0}_WEIGHT_BITS> fc_{0}_weight [FC_{0}_PE] [FC_{0}_SIMD] [FC_{0}_WMEM] =\n".format(fullyconn_counter))
    file_object.write("{\n")
    for row in layer.int_weight():
        file_object.write("{")
        for val in row[0:-1]:
            file_object.write(("0x{0}, ").format(bin_digits(val,weight_bit_width)))
        file_object.write(("0x{0}}},\n").format(bin_digits(row[-1],weight_bit_width)))
    file_object.write("};\n")

    file_object.write("static ap_uint<FC_{0}_BIAS_BITS> fc_{0}_bias [FC_{0}_PE][FC_{0}_BMEM] =".format(fullyconn_counter))
    file_object.write("{\n{")

    if layer.is_bias_quant_enabled:
        for val in layer.int_bias()[0:-1]:
            file_object.write(('0x{0}').format(bin_digits(val,bias_bit_width)))
            file_object.write(", ")
        file_object.write(('0x{0}').format(bin_digits(layer.int_bias()[-1],bias_bit_width)))
    else:
        for val in layer.bias[0:-1]:
            file_object.write(('{:f}').format(val))
            file_object.write(", ")
        file_object.write(('{:f}').format(layer.bias[-1]))
    file_object.write("}\n};\n")

def parameters_extractor(model,ext_config,result_path="",fuse=False):
    """
    Extracts layers properties, weight & bias and writes the result to .h file  with the model name
    under the same path, 

    Keyword arguments:
    model -- The model object
    config -- config dictionary, contaning special parameters 

    """
    res_path=None
    global conv2d_counter
    global maxpool2d_counter
    global quantrelu_counter
    global fullyconn_counter
    global pre_layer
    with open(Path(result_path) / "ann_config.hpp", 'w') as file_object:
        file_object.write("#ifndef ANN_CONFIG_H_\n#define ANN_CONFIG_H_\n\n")
        file_object.write("{:<48}{}\n".format("#define DATAWIDTH",ext_config['DATAWIDTH']))
        file_object.write("{:<48}{}\n".format("#define CLASS_LABEL_BITS",ext_config['CLASS_LABEL_BITS']))
        file_object.write("{:<48}{}\n\n\n".format("#define SEQUENCE_LENGTH",ext_config['SEQUENCE_LENGTH']))

        # Extract Features layers Data
        features_iter=model.features.children()
        conv_layer_list=[]
        with tqdm(total=len(model.features),desc='Extracting features parameters') as pbar:
            i=next(features_iter,None)
            while i != None:
                if isinstance(i,brevitas.nn.quant_conv.QuantConv2d):
                    if fuse:
                        bn=next(features_iter)
                        i = fuse_conv_bn_eval(i, bn)
                        # merge_bn(i,bn)
                        pbar.update()
                        print("Fusing BatchNorm2d with Conv2d layer:{}".format(conv2d_counter))
                    quant_conv2d_parser(i, file_object,ext_config)
                    conv_layer_list.append(i)
                    pre_layer="CONV2D_{}_".format(conv2d_counter)
                    conv2d_counter += 1
                elif isinstance(i,torch.nn.modules.pooling.MaxPool2d):
                    maxpool2d_parser(i, file_object,ext_config)
                    pre_layer="MAXPOOL2D_{}_".format(maxpool2d_counter)
                    maxpool2d_counter += 1
                elif isinstance(i,brevitas.nn.quant_activation.QuantReLU):
                    quantReLU_parser(i,file_object,ext_config)
                    pre_layer="RELU_{}_".format(quantrelu_counter)
                    quantrelu_counter+=1
                else:
                    print("Faced an Unknown layer:\n",type(i))
                i=next(features_iter,None)
                pbar.update()

        # Extract classifier layers Data
        for i in tqdm(model.classifier,desc='Extracting classifer parameters'):
            if isinstance(i,brevitas.nn.QuantLinear):
                fullyconn_parser(i, file_object,ext_config)
                pre_layer="FC_{}_".format(fullyconn_counter)
                fullyconn_counter += 1
            elif isinstance(i,brevitas.nn.quant_activation.QuantReLU):
                quantReLU_parser(i,file_object,ext_config)
                pre_layer="RELU_{}_".format(quantrelu_counter)
                quantrelu_counter+=1
        file_object.write("#endif")
        res_path=file_object.name

    with open(Path(result_path) / "ann_weight_bias_config.hpp", 'w') as file_object:
        file_object.write("#ifndef ANN_WEIGHT_BIAS_CONFIG_H_\n#define ANN_WEIGHT_BIAS_CONFIG_H_\n")
        file_object.write('#include "ann_config.hpp"\n\n')
        # Extract Conv layers Weight & Bias   
        conv2d_counter=0
        for i in tqdm(conv_layer_list,desc='Extracting conv layers weight & bias'):
            if FINN_STRUCTURES:
                conv_weight_bias_finn(i,file_object,int(i.quant_weight_bit_width()),int(i.quant_bias_bit_width()))
            else:
                conv_weight_bias_array(i,file_object,int(i.quant_weight_bit_width()),int(i.quant_bias_bit_width()))
            conv2d_counter += 1

        # Extract linear layers Weight & Bias
        fullyconn_counter=0
        for i in tqdm(model.classifier,desc='Extracting linear layers weight & bias'):
            if isinstance(i,brevitas.nn.quant_linear.QuantLinear):
                if FINN_STRUCTURES:
                    linear_weight_bias_finn(i,file_object,int(i.quant_weight_bit_width()),int(i.quant_bias_bit_width()))
                else:
                    linear_weight_bias_array(i,file_object,int(i.quant_weight_bit_width()),int(i.quant_bias_bit_width()))
                fullyconn_counter += 1
        file_object.write("#endif")
        res_path=res_path+"\n"+file_object.name

    return res_path
