import math
import numpy as np
from torch.autograd import Function, Variable
import torch
import logging
logger = logging.getLogger(__name__) 

def symmetric_linear_quantization_params(tensor, range_max, range_min, num_bits=8, s_bit=1, e_bits=4):
    e_bias = 2 ** (e_bits - 1) -1
    m_bits = num_bits - s_bit -e_bits
    saturation_max = (2 - 2 ** (- m_bits)) * (2 ** (2 ** e_bits - 1 - e_bias))
    saturation_min = - saturation_max
    p_min = 1 - e_bias - m_bits
    range_scale = (saturation_max - saturation_min) / (range_max - range_min)
    if tensor.min() < 0:
        offset = tensor.min() * range_scale - saturation_min
    else:
        offset = tensor.min() * range_scale
    return p_min, saturation_min, saturation_max, range_scale, offset, m_bits

def linear_quantize(tensor, p_min, saturation_min, saturation_max, range_scale, offset, m_bits): 
    tensor = tensor * range_scale - offset
    p = 2 ** (torch.floor(torch.log2(tensor.abs())) - m_bits)
    p = torch.clamp(p, p_min)
    element_scale = 2 ** p 
    assert not (element_scale == 0).any()   
    tensor = tensor / element_scale
    tensor = torch.clamp(tensor, saturation_min, saturation_max)
    return tensor, element_scale


def linear_dequantize(tensor, element_scale, range_scale, offset):
    return  (element_scale * tensor + offset) / range_scale 



