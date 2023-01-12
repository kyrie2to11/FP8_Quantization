# *
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
# *
import torch
import torch.nn as nn
import copy
from quant_modules import  Conv2dQuantizer, LinearQuantizer 
from quant_utils import quant_args


def quantize_model(model):
    """
    Recursively quantize a pretrained single-precision model to integer quantized model
    model: pretrained single-precision model
    """
    # quantize convolutional and linear layers to 8-bit
    if type(model) == nn.Conv2d:
        quant_mod = Conv2dQuantizer(**quant_args) # 创建Conv2dQuantizer实例 quant_mod
        quant_mod.set_param(model)# 根据原model设置quant_mod 的weight 和 bias
        return quant_mod
    elif type(model) == nn.Linear:
        quant_mod = LinearQuantizer(**quant_args)
        quant_mod.set_param(model)
        return quant_mod

    # recursively use the quantized module to replace the single-precision module
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            mods.append(quantize_model(m))
        return nn.Sequential(*mods)
    else:
        '''
        a = copy.copy(c) -> 浅复制,相当于给已存在的数据块c加新标签a
        b = copy.deepcopy(c) -> 深复制,将数据块c完全复制一遍作为一个单独存在的新个体,对新个体加标签b
        '''
        q_model = copy.deepcopy(model)
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                setattr(q_model, attr, quantize_model(mod))
        return q_model

def set_first_last_layer(model):
    module_list = []
    for m in model.modules():
        if isinstance(m, Conv2dQuantizer):
            module_list += [m]
        if isinstance(m, LinearQuantizer):
            module_list += [m]
    module_list[0].quant_input.is_enable = False
    module_list[-1].quant_input.bit = torch.tensor(8)