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
import os
import torch
import logging
import uuid
from quant_modules import TensorQuantizer

quant_args = {}#定义quant_args字典dict
def set_quantizer(args):
    global quant_args #在函数内部对函数外的变量进行操作，就需要在函数内部声明其为global
    quant_args.update({'mode' : args.mode, 'args' : args})#字典更新键值

logger = logging.getLogger(__name__)

def set_util_logging(filename):
    'logging.FileHandler输出目录文件名'
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(filename),
            logging.StreamHandler()
        ]
    )

def get_log_path(args):
    path=args.log_path     #原作代码为 path='squant_log'
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, args.model+"_"+args.dataset)
    if not os.path.isdir(path):
        os.mkdir(path)
    num = int(uuid.uuid4().hex[0:4], 16)#UUID: 通用唯一标识符 ( Universally Unique Identifier ), 对于所有的UUID它可以保证在空间和时间上的唯一性.
    pathname = str(num)+ '_' + args.trial_run # 自己添加的改进实验的名称args.trial_run
    path = os.path.join(path, pathname)
    if not os.path.isdir(path):
        os.mkdir(path)    
    return path

def disable_input_quantization(model):
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            module.disable_input_quantization()

def enable_quantization(model):
    for name, module in model.named_modules():
        # print("Enabling module:", name)
        if isinstance(module, TensorQuantizer):
            # print("Enabling module:", name)
            module.enable_quantization(name)

def disable_quantization(model):
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            # print("Disabling module:", name)
            module.disable_quantization(name)