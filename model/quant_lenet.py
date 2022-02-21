# Based on brevitas examples
import brevitas
import brevitas.nn as qnn
import torch.nn.functional as F
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.stats import StatsOp
from brevitas.quant import (Int8ActPerTensorFixedPoint, Int8ActPerTensorFloat,
                            Int8Bias,
                            Int8BiasPerTensorFixedPointInternalScaling,
                            Int8BiasPerTensorFloatInternalScaling,
                            Int8WeightPerTensorFixedPoint,
                            Int8WeightPerTensorFloat, Int16Bias, IntBias,
                            Uint8ActPerTensorFixedPoint, Uint8ActPerTensorFloat)
from torch import nn
from torch.nn import Module

QUANT_TYPE = QuantType.INT
SCALING_MIN_VAL = 2e-16
INPUT_QUANTIZER = Int8ActPerTensorFixedPoint
ACT_SCALING_RESTRICT_SCALING_TYPE = RestrictValueType.LOG_FP
ACT_MAX_VAL = 6.0
ACT_QUANTIZER = Uint8ActPerTensorFixedPoint

WEIGHT_SCALING_PER_OUTPUT_CHANNEL = False
WEIGHT_QUANTIZER = Int8WeightPerTensorFixedPoint

ACT_RETURN_QUANT_TENSOR = True
CONV_RETURN_QUANT_TENSOR = True
LINEAR_RETURN_QUANT_TENSOR = True
ENABLE_BIAS_QUANT = True
BIAS_QUANTIZER = Int8BiasPerTensorFixedPointInternalScaling


class QuantLeNet(Module):
    def __init__(self, bit_width=8, batchnorm=False, enable_bias_quant=ENABLE_BIAS_QUANT):
        super(QuantLeNet, self).__init__()
        self.features = nn.Sequential(
            *make_quant_conv2d(3, 32, 3, bit_width, bias=True, return_quant_tensor=not batchnorm,
                               batchnorm=batchnorm, enable_bias_quant=enable_bias_quant),
            make_quant_relu(bit_width, INPUT_QUANTIZER,
                            act_quant=ACT_QUANTIZER),
            qnn.QuantMaxPool2d(kernel_size=2, stride=2,
                               return_quant_tensor=True),
            *make_quant_conv2d(32, 64, 3, bit_width, bias=True, input_quant=None,
                               return_quant_tensor=not batchnorm, batchnorm=batchnorm, enable_bias_quant=enable_bias_quant),
            make_quant_relu(bit_width, INPUT_QUANTIZER,
                            act_quant=ACT_QUANTIZER),
            qnn.QuantMaxPool2d(kernel_size=2, stride=2,
                               return_quant_tensor=True)
        )
        self.classifier = nn.Sequential(
            make_quant_linear(in_features=64*6*6, out_features=120,
                              bit_width=bit_width, bias=True, enable_bias_quant=enable_bias_quant),
            make_quant_relu(bit_width, INPUT_QUANTIZER,
                            act_quant=ACT_QUANTIZER),
            make_quant_linear(in_features=120, out_features=84, bit_width=bit_width,
                              bias=True, enable_bias_quant=enable_bias_quant),
            make_quant_relu(bit_width, INPUT_QUANTIZER,
                            act_quant=ACT_QUANTIZER),
            make_quant_linear(in_features=84, out_features=10, bit_width=bit_width, bias=True, return_quant_tensor=False, enable_bias_quant=enable_bias_quant))
        self.features[0].cache_inference_quant_bias = True
        self.features[4].cache_inference_quant_bias = True
        self.classifier[0].cache_inference_quant_bias = True
        self.classifier[2].cache_inference_quant_bias = True
        self.classifier[4].cache_inference_quant_bias = True

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.shape[0], -1)
        out = self.classifier(out)
        return out


def make_quant_conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      bit_width,
                      stride=1,
                      padding=0,
                      dilation=1,
                      groups=1,
                      bias=True,
                      batchnorm=False,
                      weight_quant=WEIGHT_QUANTIZER,
                      bias_quant=BIAS_QUANTIZER,
                      input_quant=INPUT_QUANTIZER,
                      output_quant=INPUT_QUANTIZER,
                      return_quant_tensor=CONV_RETURN_QUANT_TENSOR,
                      enable_bias_quant=ENABLE_BIAS_QUANT,
                      weight_scaling_per_output_channel=WEIGHT_SCALING_PER_OUTPUT_CHANNEL,
                      weight_scaling_min_val=SCALING_MIN_VAL):
    bias_quant_type = QUANT_TYPE if enable_bias_quant else QuantType.FP
    layers = [qnn.QuantConv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias,
                              weight_quant=weight_quant,
                              bias_quant=bias_quant,
                              input_quant=input_quant,
                              output_quant=output_quant,
                              weight_bit_width=bit_width,
                              input_bit_width=bit_width,
                              return_quant_tensor=return_quant_tensor,
                              bias_quant_type=bias_quant_type,
                              compute_output_bit_width=bias and enable_bias_quant,
                              compute_output_scale=bias and enable_bias_quant,
                              weight_scaling_per_output_channel=weight_scaling_per_output_channel,
                              weight_scaling_min_val=weight_scaling_min_val)]

    if(batchnorm):
        layers.append(nn.BatchNorm2d(out_channels))
    return layers


def make_quant_linear(in_features,
                      out_features,
                      bit_width,
                      bias,
                      weight_quant=WEIGHT_QUANTIZER,
                      bias_quant=BIAS_QUANTIZER,
                      return_quant_tensor=LINEAR_RETURN_QUANT_TENSOR,
                      output_quant=ACT_QUANTIZER,
                      enable_bias_quant=ENABLE_BIAS_QUANT,
                      weight_scaling_min_val=SCALING_MIN_VAL):
    bias_quant_type = QUANT_TYPE if enable_bias_quant else QuantType.FP
    return qnn.QuantLinear(in_features, out_features,
                           weight_quant=weight_quant,
                           bias_quant=bias_quant,
                           return_quant_tensor=return_quant_tensor,
                           output_quant=output_quant,
                           output_bit_width=bit_width,
                           bias=bias,
                           bias_quant_type=bias_quant_type,
                           compute_output_bit_width=bias and enable_bias_quant,
                           compute_output_scale=bias and enable_bias_quant,
                           weight_bit_width=bit_width,
                           weight_scaling_min_val=weight_scaling_min_val)


def make_quant_relu(bit_width,
                    input_quant,
                    act_quant=ACT_QUANTIZER,
                    restrict_scaling_type=ACT_SCALING_RESTRICT_SCALING_TYPE,
                    scaling_min_val=SCALING_MIN_VAL,
                    max_val=ACT_MAX_VAL,
                    return_quant_tensor=ACT_RETURN_QUANT_TENSOR):
    return qnn.QuantReLU(bit_width=bit_width,
                         input_quant=input_quant,
                         act_quant=act_quant,
                         restrict_scaling_type=restrict_scaling_type,
                         scaling_min_val=scaling_min_val,
                         max_val=max_val,
                         return_quant_tensor=return_quant_tensor)
