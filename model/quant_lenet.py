# Based on brevitas examples
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
import brevitas
import brevitas.nn as qnn
from brevitas.core.scaling import ScalingImplType
from brevitas.core.quant import QuantType
from brevitas.core.stats import StatsOp
from brevitas.core.restrict_val import RestrictValueType
from brevitas.quant import Int8ActPerTensorFixedPoint, Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFixedPoint, Int8WeightPerTensorFloat
from brevitas.quant import Int8Bias, Int8BiasPerTensorFloatInternalScaling, Int8BiasPerTensorFixedPointInternalScaling, Int16Bias, IntBias

QUANT_TYPE = QuantType.INT
SCALING_MIN_VAL = 2e-16

ACT_SCALING_IMPL_TYPE = ScalingImplType.PARAMETER
ACT_SCALING_PER_CHANNEL = True # custom 200 epoch
ACT_SCALING_RESTRICT_SCALING_TYPE = RestrictValueType.LOG_FP
ACT_MAX_VAL = 6.0
ACT_QUANTIZER = Int8ActPerTensorFixedPoint
ACT_PER_CHANNEL_BROADCASTABLE_SHAPE = None

WEIGHT_SCALING_IMPL_TYPE = ScalingImplType.STATS
WEIGHT_SCALING_PER_OUTPUT_CHANNEL = False # custom 200 epoch
WEIGHT_SCALING_STATS_OP = StatsOp.MAX
WEIGHT_QUANTIZER = Int8WeightPerTensorFixedPoint
WEIGHT_RESTRICT_SCALING_TYPE = RestrictValueType.LOG_FP
WEIGHT_NARROW_RANGE = False # custom 200 epoch

ACT_RETURN_QUANT_TENSOR = True 
CONV_RETURN_QUANT_TENSOR = True
LINEAR_RETURN_QUANT_TENSOR = True
ENABLE_BIAS_QUANT = True # enable bias quantization, default: False
BIAS_QUANTIZER = Int8BiasPerTensorFixedPointInternalScaling

class QuantLeNet(Module):
    def __init__(self, bit_width=8,batchnorm=False,enable_bias_quant=ENABLE_BIAS_QUANT):
        super(QuantLeNet, self).__init__()
        self.features = nn.Sequential(
            *make_quant_conv2d(3, 32, 3,bit_width,bias=True,return_quant_tensor=not batchnorm,batchnorm=batchnorm,enable_bias_quant=enable_bias_quant),
            make_quant_relu(bit_width=bit_width,input_quant= ACT_QUANTIZER if batchnorm else None),
            qnn.QuantMaxPool2d(kernel_size=2, stride=2, return_quant_tensor=True),
            *make_quant_conv2d(32, 64, 3,bit_width,bias=True,input_quant=None,return_quant_tensor=not batchnorm,batchnorm=batchnorm,enable_bias_quant=enable_bias_quant),
            make_quant_relu(bit_width=bit_width,input_quant= ACT_QUANTIZER if batchnorm else None),
            qnn.QuantMaxPool2d(kernel_size=2, stride=2, return_quant_tensor=True)
            )
        self.classifier = nn.Sequential(
            make_quant_linear(in_features=64*6*6,out_features=120,bit_width=bit_width,bias=True,enable_bias_quant=enable_bias_quant),
            make_quant_relu(bit_width=bit_width,input_quant=None),
            make_quant_linear(in_features=120,out_features=84,bit_width=bit_width,bias=True,enable_bias_quant=enable_bias_quant),
            make_quant_relu(bit_width=bit_width,input_quant=None),
            make_quant_linear(in_features=84,out_features=10,bit_width=bit_width,bias=True,return_quant_tensor=False,enable_bias_quant=enable_bias_quant))
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
                      groups=1,
                      bias=True,
                      batchnorm=False,
                      weight_quant=WEIGHT_QUANTIZER,
                      bias_quant=BIAS_QUANTIZER,
                      input_quant=ACT_QUANTIZER,
                      output_quant=ACT_QUANTIZER,
                      return_quant_tensor=CONV_RETURN_QUANT_TENSOR, #Custom
                      enable_bias_quant=ENABLE_BIAS_QUANT,
                      weight_quant_type=QUANT_TYPE,
                      weight_scaling_impl_type=WEIGHT_SCALING_IMPL_TYPE,
                      weight_scaling_stats_op=WEIGHT_SCALING_STATS_OP,
                      weight_scaling_per_output_channel=WEIGHT_SCALING_PER_OUTPUT_CHANNEL,
                      weight_restrict_scaling_type=WEIGHT_RESTRICT_SCALING_TYPE,
                      weight_narrow_range=WEIGHT_NARROW_RANGE,
                      weight_scaling_min_val=SCALING_MIN_VAL):
    bias_quant_type = QUANT_TYPE if enable_bias_quant else QuantType.FP
    layers=[qnn.QuantConv2d(in_channels,
                           out_channels,
                           groups=groups,
                           kernel_size=kernel_size,
                           padding=padding,
                           stride=stride,
                           bias=bias,
                           weight_quant=weight_quant,
                           bias_quant=bias_quant,
                           input_quant=input_quant,
                           input_bit_width=bit_width,
                           return_quant_tensor=return_quant_tensor,
                           output_quant=output_quant,
                           output_bit_width=bit_width,
                           bias_quant_type=bias_quant_type,
                           compute_output_bit_width=bias and enable_bias_quant,
                           compute_output_scale=bias and enable_bias_quant,
                           weight_bit_width=bit_width,
                           weight_quant_type=weight_quant_type,
                           weight_scaling_impl_type=weight_scaling_impl_type,
                           weight_scaling_stats_op=weight_scaling_stats_op,
                           weight_scaling_per_output_channel=weight_scaling_per_output_channel,
                           weight_restrict_scaling_type=weight_restrict_scaling_type,
                           weight_narrow_range=weight_narrow_range,
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
                      weight_quant_type=QUANT_TYPE,
                      weight_scaling_impl_type=WEIGHT_SCALING_IMPL_TYPE,
                      weight_scaling_stats_op=WEIGHT_SCALING_STATS_OP,
                      weight_scaling_per_output_channel=WEIGHT_SCALING_PER_OUTPUT_CHANNEL,
                      weight_restrict_scaling_type=WEIGHT_RESTRICT_SCALING_TYPE,
                      weight_narrow_range=WEIGHT_NARROW_RANGE,
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
                           weight_quant_type=weight_quant_type,
                           weight_scaling_impl_type=weight_scaling_impl_type,
                           weight_scaling_stats_op=weight_scaling_stats_op,
                           weight_scaling_per_output_channel=weight_scaling_per_output_channel,
                           weight_restrict_scaling_type=weight_restrict_scaling_type,
                           weight_narrow_range=weight_narrow_range,
                           weight_scaling_min_val=weight_scaling_min_val
                           )


def make_quant_relu(bit_width,
                    quant_type=QUANT_TYPE,
                    input_quant=ACT_QUANTIZER,
                    scaling_impl_type=ACT_SCALING_IMPL_TYPE,
                    scaling_per_channel=ACT_SCALING_PER_CHANNEL,
                    restrict_scaling_type=ACT_SCALING_RESTRICT_SCALING_TYPE,
                    scaling_min_val=SCALING_MIN_VAL,
                    max_val=ACT_MAX_VAL,
                    return_quant_tensor=ACT_RETURN_QUANT_TENSOR,
                    per_channel_broadcastable_shape=ACT_PER_CHANNEL_BROADCASTABLE_SHAPE):
    return qnn.QuantReLU(bit_width=bit_width,
                         input_quant=input_quant,
                         input_bit_width=bit_width,
                         quant_type=quant_type,
                         scaling_impl_type=scaling_impl_type,
                         scaling_per_channel=scaling_per_channel,
                         restrict_scaling_type=restrict_scaling_type,
                         scaling_min_val=scaling_min_val,
                         max_val=max_val,
                         return_quant_tensor=return_quant_tensor,
                         output_quant=ACT_QUANTIZER,
                         output_bit_width=bit_width,
                         per_channel_broadcastable_shape=per_channel_broadcastable_shape
                         )
