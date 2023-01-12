# *
# @file Different utility functions
# Copyright (c) Cong Guo, Yuxian Qiu, Jingwen Leng, Xiaotian Gao, 
# Chen Zhang, Yunxin Liu, Fan Yang, Yuhao Zhu, Minyi Guo
# All rights reserved.
# This file is part of SQuant repository.
#
# SQuant is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SQuant is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SQuant repository.  If not, see <http://www.gnu.org/licenses/>.
# *
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from quant_affine import *
import warnings

try:
    from quant_cuda import rounding_loop as SQuant_func
except ImportError:
    warnings.warn("CUDA-based SQuant is not installed! PyTorch-based SQuant will lead to a prolonged quantization process.")
    from squant_function import SQuant_func

logger = logging.getLogger(__name__) 
class TensorQuantizer(nn.Module):
    def __init__(self, mode="base", is_signed=False, is_enable=False, is_input=False, args=None):
        super().__init__() 
        self.mode = mode
        self.register_buffer('bit', torch.tensor(1)) 
        self.is_signed = is_signed
        self.is_enable = is_enable
        self.is_enable_input = is_enable
        self.is_input = is_input
        self.args = args
        self.percent = self.args.percent / 100
        self.is_sigma = False
        if args.sigma > 0:
            self.percent = args.sigma
            self.is_sigma = True
        self.quant_weight_tensor = None
        self.quant_input_tensor = None
        self.register_buffer('x_max', torch.tensor(1.0))
        self.register_buffer('x_min', torch.tensor(1.0))
        self.has_inited_quant_para = False

        self.squant_k = True
        self.squant_c = True

    def disable_input_quantization(self):
        self.is_enable_input = False
    
    def enable_quantization(self, name):
        self.name = name
        self.is_enable = True

    def disable_quantization(self, name):
        self.name = name
        self.is_enable = False


    def _sigma(self, tensor):
        if self.is_signed:
            return tensor[tensor > 0].std() #tensor[tensor > 0].std() -> tensor 中大于0的部分的标准差std(standard deviation)
        return tensor.std()
    

    'Algorithm 1: Progressive SQuant Algorithm.'
    def adaptive_round(self, x, t_min = None, t_max = None):
        # Get the rounding integer and fraction.
        '''
        rounding_number[]:输入x四舍五入到最近的整数
        rounding_error[]:输入x四舍五入后造成的舍入误差
        up_error[]:针对输入x中四舍五入向下取整元素,计算变为向上取整相应的偏差;本身即是向上取整元素up_error=0
        up_number[]:输入x向上取整的结果
        up_priority[]:针对输入x中四舍五入向下取整元素,向上取整的优先级 = (x.round()-x).abs();本身即是向上取整元素up_priority=0
        '''
        rounding_number = x.round()
        rounding_error  = rounding_number - x
            
        up_number = rounding_number.clone()
        up_error  = rounding_error.clone()
        up_error[x >= t_max]  = 0.0 # x up_error为shape相等的tensor，up_error[x >= t_max]=0.0 -> 找出x >= t_max的元素索引,并将up_error相应索引位置元素赋0
        up_error[up_error > 0]  = 0.0
        up_priority = up_error.clone().abs()

        up_error[up_error != 0]  += 1
        up_number[up_error != 0] += 1
        '''
        down_error[]:针对输入x中四舍五入向上取整元素,计算变为向下取整相应的错误偏差;本身即是向下取整元素down_error=0
        down_number[]:输入x向下取整的结果
        down_priority[]:针对输入x中四舍五入向上取整元素,向下取整的优先级 = (x.round()-x).abs();本身即是向下取整元素down_priority=0
        '''
        down_number = rounding_number.clone()
        down_error  = rounding_error.clone()
        down_error[x <= t_min]  = 0.0
        down_error[down_error < 0]  = 0.0
        down_priority = down_error.clone().abs()

        down_error[down_error != 0]  -= 1
        down_number[down_error != 0] -= 1

        flip_number = torch.tensor([0.0], device=x.device) # 在x所在的device上创造一个叶子张量(没有autograd历史记录),flip_number=tensor([0.0])
        flip_up_number = torch.tensor([0.0], device=x.device)
        flip_down_number = torch.tensor([0.0], device=x.device)

        
        conver_shape = rounding_number.view(x.shape[0], x.shape[1], -1).shape
        if conver_shape[2] == 1:
            self.squant_k = False

        if self.squant_k:
            rounding_error_sum = rounding_error.view(conver_shape).sum(-1) # sum(-1) 对最后一个维度即dim=2求和，压缩掉此维度
            _, up_order = torch.sort(up_priority.view(conver_shape), descending=True) # 不指定dim,则对最后一个dim进行降序排序
            _, down_order = torch.sort(down_priority.view(conver_shape), descending=True)
            up_priority *= 0.0
            down_priority *= 0.0

            # SQuant Flip Algorithm used in SQuant-K
            SQuant_func(
                flip_number,
                flip_up_number,
                flip_down_number,
                
                rounding_error_sum,
                rounding_number.view(conver_shape), 
                rounding_error.view(conver_shape), 

                up_number.view(conver_shape), 
                up_error.view(conver_shape), 
                up_priority.view(conver_shape), 
                up_order, 

                down_number.view(conver_shape), 
                down_error.view(conver_shape), 
                down_priority.view(conver_shape),
                down_order,
            )
        
        if self.squant_c:
            conver_shape = rounding_number.view(1, x.shape[0], -1).shape
            rounding_error_sum = rounding_error.view(conver_shape).sum(-1)
            _, up_order = torch.sort(up_priority.view(conver_shape), descending=True)
            _, down_order = torch.sort(down_priority.view(conver_shape), descending=True)

            # SQuant Flip Algorithm used in SQuant-C
            SQuant_func(
                flip_number,
                flip_up_number,
                flip_down_number,
                
                rounding_error_sum,
                rounding_number.view(conver_shape), 
                rounding_error.view(conver_shape), 

                up_number.view(conver_shape), 
                up_error.view(conver_shape), 
                up_priority.view(conver_shape), 
                up_order, 

                down_number.view(conver_shape), 
                down_error.view(conver_shape), 
                down_priority.view(conver_shape),
                down_order
            )
        return rounding_number

    @torch.no_grad()
    def update_activation_clip_range(self, data):
            data_max = data.max()
            alpha = self.percent * data.abs().max()
            if self.is_sigma:
                sigma = self._sigma(data)#计算activation 标准差σ
                alpha = self.percent * sigma
                if self.is_signed:
                    # We also consider the signed activation. Other framworks will skip this tensor.
                    alpha = self.percent * sigma / 1.25
                
                # For a higher bit-width, using a wider range still will not cause accuracy loss.
                if self.bit < 6:
                    # For small bit, need clip.
                    alpha = min(alpha, data_max)

            # Activation min
            if self.is_signed:
                self.x_min = -alpha
            else:
                self.x_min.data = torch.zeros_like(alpha)
            # Activation max
            self.x_max = alpha

    def quant_weight_or_activation(self, tensor, p_min, saturation_min, saturation_max, range_scale, offset, m_bits):
        if self.has_inited_quant_para == False:
            logger.info(f"def quant_weight_or_activation: tensor's max/min: {tensor.max()}/{tensor.min()}")

        if self.is_input == False:
            quant_tensor, element_scale = linear_quantize(tensor, p_min, saturation_min, saturation_max, range_scale, offset, m_bits)
            logger.info(f"def quant_weight_or_activation: weight_tensor's max/min after linear_quantize: {quant_tensor.max()}/{quant_tensor.min()}")
            #quant_tensor = quant_tensor.round()
            quant_tensor = self.adaptive_round(quant_tensor, saturation_min, saturation_max)
            logger.info(f"def quant_weight_or_activation: weight_tensor's max/min after Adaround:{quant_tensor.max()}/{quant_tensor.min()}")
            quant_tensor = linear_dequantize(quant_tensor, element_scale, range_scale, offset)
            logger.info(f"def quant_weight_or_activation: weight_tensor's max/min after linear_dequantize: {quant_tensor.max()}/{quant_tensor.min()}")
            
            
        else:
            quant_tensor, element_scale = linear_quantize(tensor, p_min, saturation_min, saturation_max, range_scale, offset, m_bits)
            if self.has_inited_quant_para == False:
                logger.info(f"def quant_weight_or_activation: activation_tensor's max/min: {quant_tensor.max()}/{quant_tensor.min()}")
            quant_tensor = quant_tensor.round()
            if self.has_inited_quant_para == False:
                logger.info(f"def quant_weight_or_activation: activation_tensor's max/min after round: {quant_tensor.max()}/{quant_tensor.min()}")
            quant_tensor = linear_dequantize(quant_tensor, element_scale, range_scale, offset)
            if self.has_inited_quant_para == False:
                logger.info(f"def quant_weight_or_activation: activation_tensor's max/min after linear_dequantize: {quant_tensor.max()}/{quant_tensor.min()}")
        return quant_tensor

    def _init_quant_para(self, data):
        if self.is_input == True:
            self.update_activation_clip_range(data)
            p_min, saturation_min, saturation_max, range_scale, offset, m_bits = symmetric_linear_quantization_params(data,self.x_max,self.x_min)
        else:
            p_min, saturation_min, saturation_max, range_scale, offset, m_bits = symmetric_linear_quantization_params(data,data.max(),data.min())    
        if self.has_inited_quant_para == False:
            logger.info("QUANT FP8: %s " % (self.name))
            if self.mode == "squant-e":
                self.squant_k = False
                self.squant_c = False
                self.mode="squant"
            elif self.mode == "squant-k":
                self.squant_c = False
                self.mode="squant"
            elif self.mode == "squant-c":
                self.squant_k = False
                self.mode="squant"

            if self.mode == "squant":
                                                     
                if self.is_input == False:
                    #Weight quantization
                    start_w = time.perf_counter()
                    self.quant_weight_tensor = self.quant_weight_or_activation(data, p_min, saturation_min, saturation_max, range_scale, offset, m_bits)
                    elapsed_w = (time.perf_counter() - start_w)
                    logger.info("Weight Quantzation Time: %f ms" %(elapsed_w * 1000))
                else:
                    #Activation quantization
                    start_a = time.perf_counter()                                      
                    self.quant_input_tensor = self.quant_weight_or_activation(data, p_min, saturation_min, saturation_max, range_scale, offset, m_bits)
                    elapsed_a = (time.perf_counter() - start_a)
                    logger.info("Activation Quantzation Time: %f ms" %(elapsed_a * 1000))                    
            else:
                raise RuntimeError("Unsupported mode: " + self.mode) 
            self.has_inited_quant_para = True
        else:
            if self.is_input:
                self.quant_input_tensor = self.quant_weight_or_activation(data, p_min, saturation_min, saturation_max, m_bits, range_scale, offset)


    def tensor_forward(self, tensor):
        if self.mode == "base":
            return tensor
        if not self.is_enable:
            return tensor

        with torch.no_grad():
            self._init_quant_para(tensor)        
            if self.is_input:
                return self.quant_input_tensor
            else:
                return self.quant_weight_tensor

    def forward(self, tensor):
        return self.tensor_forward(tensor)

class LinearQuantizer(nn.Module):
    """
    Class to quantize given linear layer weights
    """
    def __init__(self, mode=None, wbit=None, abit=None, args=None):
        """
        weight: bit-setting for weight
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super().__init__()
        assert mode is not None,'Quantizer is not initilized!'
        self.quant_input  = TensorQuantizer(mode=mode, is_signed=False, is_enable=True, args=args, is_input=True)
        self.quant_weight = TensorQuantizer(mode=mode, is_signed=True, is_enable=True, args=args)

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.data.clone())
        try:
            self.bias = nn.Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, input):
        input = self.quant_input(input)
        weight = self.quant_weight(self.weight)
        return F.linear(input, weight, self.bias)


class Conv2dQuantizer(nn.Module):
    """
    Class to quantize given convolutional layer weights
    """
    def __init__(self, mode=None, wbit=None, abit=None, args=None):
        super().__init__()
        assert mode is not None,'Quantizer is not initilized!'
        self.quant_input  = TensorQuantizer(mode=mode, is_signed=False, is_enable=True, args=args, is_input=True)
        self.quant_weight = TensorQuantizer(mode=mode, is_signed=True, is_enable=True, args=args)

    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = nn.Parameter(conv.weight.data.clone())
        try:
            self.bias = nn.Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None

    def _conv_forward(self, input, weight):
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        input = self.quant_input(input)       
        weight = self.quant_weight(self.weight) 
        return self._conv_forward(input, weight)