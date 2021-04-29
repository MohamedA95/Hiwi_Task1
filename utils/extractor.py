import torch
import pathlib
import brevitas
from model.quant_vgg import QuantVGG

conv2d_counter = 0
maxpool2d_counter = 0
quantrelue_counter = 0
fullyconn_counter=0
pre_layer=None
def quant_conv2d_parser(layer, file,ext_config):
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_K"), layer.kernel_size[0]))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_IFM_CH"), layer.in_channels))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_STRIDE"), layer.stride[0]))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_PADDING"), layer.padding[0]))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_OFM_CH"), layer.out_channels))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_OFM_DIM"), "((CONV2D_{0}_IFM_DIM - CONV2D_{0}_K + 2 * CONV2D_{0}_PADDING) / CONV2D_{0}_STRIDE + 1)".format(conv2d_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_PE"), ext_config['PE']))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_SIMD"), ext_config['SIMD']))
    if layer.is_bias_quant_enabled:
        file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_BIAS_BITS"), layer.quant_bias_bit_width() if layer.quant_bias_bit_width() is not None else "0"))
        file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_BIAS_INT_BITS"), int(layer.quant_bias_bit_width()+torch.log2(layer.quant_bias_scale()))))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_WEIGHT_BITS"), layer.quant_weight_bit_width()))
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
    file.write("{:<48}{}\n\n".format("{:s}{:d}{:s}".format("#define CONV2D_", conv2d_counter, "_WMEM"), "((1 * CONV2D_{0}_K * CONV2D_{0}_IFM_CH * CONV2D_{0}_OFM_CH) / (CONV2D_{0}_PE * CONV2D_{0}_SIMD))".format(conv2d_counter)))
    
def maxpool2d_parser(layer, file,ext_config):
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_K"), layer.kernel_size))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_IFM_CH"), "{}OFM_CH".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_IFM_DIM"), "{}OFM_DIM".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_STRIDE"), "MAXPOOL2D_{}_K".format(maxpool2d_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_PADDING"), layer.padding))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_OFM_CH"), "MAXPOOL2D_{}_IFM_CH".format(maxpool2d_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_OFM_DIM"), "(MAXPOOL2D_{0}_IFM_DIM / MAXPOOL2D_{0}_K)".format(maxpool2d_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_SIMD"), ext_config['SIMD']))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_IA_BITS"), "{}OA_BITS".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_IA_INT_BITS"), "{}OA_INT_BITS".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_OA_BITS"), "MAXPOOL2D_{}_IA_BITS".format(maxpool2d_counter)))
    file.write("{:<48}{}\n\n".format("{:s}{:d}{:s}".format("#define MAXPOOL2D_", maxpool2d_counter, "_OA_INT_BITS"), "MAXPOOL2D_{}_IA_INT_BITS".format(maxpool2d_counter)))

def quantReLU_parser(layer, file,ext_config):
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelue_counter, "_IFM_CH"), "{}OFM_CH".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelue_counter, "_IFM_DIM"), "{}OFM_DIM".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelue_counter, "_OFM_CH"), "RELU_{}_IFM_CH".format(quantrelue_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelue_counter, "_OFM_DIM"), "RELU_{}_IFM_DIM".format(quantrelue_counter)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelue_counter, "_SIMD"), ext_config['SIMD']))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelue_counter, "_IA_BITS"), "{}OA_BITS".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelue_counter, "_IA_INT_BITS"), "{}OA_INT_BITS".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelue_counter, "_OA_BITS"), int(layer.quant_output_bit_width())))
    file.write("{:<48}{}\n\n".format("{:s}{:d}{:s}".format("#define RELU_", quantrelue_counter, "_OA_INT_BITS"), int(layer.quant_output_bit_width()+torch.log2(layer.quant_output_scale()))))

def fullyconn_parser(layer,file):
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_IFM_CH"), "({}OFM_CH)".format(pre_layer)))
    file.write("{:<48}{}\n".format("{:s}{:d}{:s}".format("#define FC_", fullyconn_counter, "_IFM_DIM"), "1" if fullyconn_counter > 0 else "({}OFM_DIM".format(pre_layer)))
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


def parameters_extractor(model,ext_config):
    """
    Extracts layers properites, weight & bias and writes the result to .h file  with the model name
    under the same path, 

    Keyword arguments:
    model -- The model object
    config -- config dictionary, contaning special parameters 

    """
    with open( "{}_config.h".format(type(model).__name__), 'w') as file_object:
        global conv2d_counter
        global maxpool2d_counter
        global quantrelue_counter
        global fullyconn_counter
        global pre_layer
        
        file_object.write("#ifndef CONFIG_H_\n#define CONFIG_H_\n\n\n\n")
        file_object.write("{:<48}{}\n".format("#define DATAWIDTH",ext_config['DATAWIDTH']))
        file_object.write("{:<48}{}\n".format("#define CLASS_LABEL_BITS",ext_config['CLASS_LABEL_BITS']))
        file_object.write("{:<48}{}\n\n\n".format("#define SEQUENCE_LENGTH",ext_config['SEQUENCE_LENGTH']))

        # Extract Features layers Data
        for i in model.features:
            if isinstance(i,brevitas.nn.quant_conv.QuantConv2d):
                quant_conv2d_parser(i, file_object,ext_config)
                pre_layer="CONV2D_{}_".format(conv2d_counter)
                conv2d_counter += 1
            elif isinstance(i,torch.nn.modules.pooling.MaxPool2d):
                maxpool2d_parser(i, file_object,ext_config)
                pre_layer="MAXPOOL2D_{}_".format(maxpool2d_counter)
                maxpool2d_counter += 1
            elif isinstance(i,brevitas.nn.quant_activation.QuantReLU):
                quantReLU_parser(i,file_object,ext_config)
                pre_layer="RELU_{}_".format(quantrelue_counter)
                quantrelue_counter+=1
        print("Extracted Features parameters successfully, Working on classifer parameters")
        # Extract classifier layers Data
        for i in model.classifier:
            if isinstance(i,brevitas.nn.QuantLinear):
                fullyconn_parser(i, file_object)
                pre_layer="FC_{}_".format(fullyconn_counter)
                fullyconn_counter += 1
            elif isinstance(i,brevitas.nn.quant_activation.QuantReLU):
                quantReLU_parser(i,file_object,ext_config)
                pre_layer="RELU_{}_".format(quantrelue_counter)
                quantrelue_counter+=1
        print("Extracting features weight & bias")
        # Extract Conv layers Weight & Bias   
        conv2d_counter=0
        for i in model.features:
            if isinstance(i,brevitas.nn.quant_conv.QuantConv2d):
                file_object.write("static ap_uint<CONV2D_{0}_WEIGHT_BITS> conv2d_{0}_weight [CONV2D_{0}_PE] [CONV2D_{0}_SIMD] [CONV2D_{0}_WMEM] =\n".format(conv2d_counter))
                file_object.write("{\n")
                for j in i.int_weight():
                    file_object.write("{\n")
                    for k in j:
                        file_object.write("{\n")
                        for r in k:
                            file_object.write("{")
                            for m in r[0:-1]: 
                                file_object.write(("{:02x}, " if m>0 else "{:03x}, ").format(m))
                            file_object.write(("{:02x}}},\n" if r[-1]>0 else "{:03x}}},\n").format(r[-1]))
                        file_object.write("};\n")
                    file_object.write("};\n")
                file_object.write("};\n")

                if i.is_bias_quant_enabled:
                    file_object.write("static ap_uint<CONV2D_{0}_BIAS_BITS> conv2d_{0}_bias [CONV2D_{0}_PE][CONV2D_{0}_BMEM] =\n".format(conv2d_counter))
                    file_object.write("{\n{\n")
                    for j in i.int_bias()[0:-1]:
                        file_object.write(("{:02x},\n" if j>0 else "{:03x},\n").format(j))
                    file_object.write(("{:02x}\n" if i.int_bias()[-1]>0 else "{:03x}\n").format(i.int_bias()[-1]))
                    file_object.write("}\n};\n")
                conv2d_counter += 1

        # Extract Fully Connected layers Weight & Bias
        print("Extracting classifier weight & bias")
        fullyconn_counter=0
        for i in model.classifier:
            if isinstance(i,brevitas.nn.quant_linear.QuantLinear):
                file_object.write("FullyConnected_{}_WEIGHT\n".format(fullyconn_counter))
                file_object.write("{\n")
                for row in i.int_weight():
                    file_object.write("{")
                    for val in row[0:-1]:
                        file_object.write(("{:02x}, " if val>0 else "{:03x}, ").format(val))
                    file_object.write(("{:02x}}};\n" if row[-1]>0 else "{:03x}}};\n").format(row[-1]))
                file_object.write("\n")
                if i.is_bias_quant_enabled:
                    file_object.write("FullyConnected_{}_BIAS\n{\n".format(fullyconn_counter))
                    for val in i.int_bias()[0:-1]:
                        file_object.write(("{:02x}, " if val>0 else "{:03x}, ").format(val))
                    file_object.write(("{:02x}}}\n" if i.int_bias()[-1]>0 else "{:03x}}}\n").format(i.int_bias()[-1]))
                    file_object.write(";\n")
                fullyconn_counter += 1
    return "{}_config.h".format(type(model).__name__)
