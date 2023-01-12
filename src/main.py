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
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as models

from pytorchcv.model_provider import get_model
from dataloader import *
from quant_utils import *
from quant_model import *


parser = argparse.ArgumentParser(description='SQuant') #创建parser
'''
添加参数
'''
parser.add_argument('--mode', default='squant', type=str, choices=['base', 'squant', 'squant-k', 'squant-c', 'squant-e'],
                    help='quantizer mode')
parser.add_argument('--dataset', default='imagenet', type=str, 
                    help='dataset name')
parser.add_argument('--dataset_path', default='/liyawei/DataSets/imagenet/val', type=str, 
                    help='dataset path')
parser.add_argument('--log_path', default='/liyawei/FP8_Quantization_Simulation/FP8_Quantization_Prototype_0/fp8_quant_log', type=str, 
                    help='log_path')
parser.add_argument('--model', default='ResNet18', type=str, choices=['resnet18', 'resnet50', 'inceptionV3', 'sqnxt23_w1', 'sqnxt23_w2', 'shufflenet_g1_w1'],
                    help='model name')
parser.add_argument('--model_state_dict_path', default='/liyawei/Models/pretrained_models/resnet18-0896-77a56f15.pth', type=str, 
                    help='pretrained model parameter path')
parser.add_argument('--batch_size', default=256, type=int, 
                    help='batch_size num')
parser.add_argument('--disable_quant', "-dq", default=False, action='store_true', 
                    help='disable quant')
parser.add_argument('--disable_activation_quant', "-daq", default=False, action='store_true', 
                    help='quant_activation')
parser.add_argument('--percent', '-p', default='100', type=int, 
                    help='percent')
parser.add_argument('--sigma', '-s', default='0', type=float, 
                    help='Init activation range with Batchnorm Sigma')
parser.add_argument('--trial_run', '-tr', default='Base', type=str,
                    help='improvement or mend on SQuant baseline')

args = parser.parse_args()

### Logging Setting
output_path = get_log_path(args) 

'quant_utils.py logging config setting'
set_util_logging(output_path+"/squant.log") # quant_utils logger config setting

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(output_path+"/squant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__) #创建logger
logger.info(output_path)#logger打印等级为INFO的信息
logger.info(args)

### Model Loading
logger.info('==> Building model..')
model = get_model(args.model, pretrained=True)#从pytorchcv.model_provider导入的get_model, pretrained=True 服务器开发环境无法下载模型的参数

'加载下载好预训练模型参数'
model_state_dict = torch.load(args.model_state_dict_path) #提前下载好的模型参数的路径
model.load_state_dict(model_state_dict)#从上述路径加载模型参数

### Random Input
if args.model.startswith('inception'):#Python method startswith(): 用于检查字符串是否是以指定子字符串开头，如果是则返回True,否则返回False -> 检查是否是inception模型，inception输入单独处理
    rand_input = torch.rand([args.batch_size, 3, 299, 299], dtype=torch.float, requires_grad=False).cuda()# torch.rand(): Returns a tensor filled with random numbers from a uniform distribution on the interval [0,1)
else:
    rand_input = torch.rand([args.batch_size, 3, 224, 224], dtype=torch.float, requires_grad=False).cuda()

'将rand_input替换为ZeroQ 蒸馏出的32张高斯图片的tensor'
rand_input = torch.load("/liyawei/ZeroQ/classification/refined_gaussian_imgs/gaussian_img_tensor.pt").cuda()
### Set Quantizer
logger.info('==> Setting quantizer..')
set_quantizer(args)#设置Quantizer参数
quantized_model = quantize_model(model)

if args.disable_quant:
    disable_quantization(quantized_model)
else:
    enable_quantization(quantized_model)

if args.disable_activation_quant:
    disable_input_quantization(quantized_model)
     
set_first_last_layer(quantized_model)
quantized_model.cuda()


logger.info("SQuant Start!")
quantized_model.eval()
quantized_model(rand_input)
logger.info("SQuant has Done!")


### define validation/test function
# 定义test,测试量化后模型掉点情况
@torch.no_grad() # @/with torch.no_grad() 之后代码不会计算梯度，但BN 和 Dropout 保持正常; model.eval() 计算梯度，但BN 和 Dropout 关闭
def test(quantized_model_):
    quantized_model_.eval()
    test_loss = 0
    correct = 0
    correct_5 = 0
    total = 0    
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda() # inputs.Size([256, 3, 224, 224]) targets.Size([256])
        outputs = quantized_model_(inputs) # outputs.size([256,1000]) 
        criterion = nn.CrossEntropyLoss() 
        loss = criterion(outputs, targets) # 计算 outputs 和 targets 的交叉熵损失
        test_loss += loss.item() # .item() 取出单元素张量的元素值并返回该值 累积到test_loss

        '''
        1. 计算Top-1正确率
        '''

        _, predicted = outputs.max(1) # 等同于_, predicted = torch.max(outputs,1), outputs.size([256,1000]) -> predicted.Size([256])
        total += targets.size(0) # 此batch label 个数
        correct += predicted.eq(targets).sum().item() # 预测正确的图片个数

        '''
        2. 计算Top-5正确率
        '''

        _, predicted_5 = outputs.topk(5, 1, True, True) # outputs.topk(5,1,True,True) -> 返回在outputs dim=1中最大的5个元素,且已排好序 _接收元素(value),predicted_5接收指数(indices) _.size([256,5]) predicted_5.size([256,5])
        predicted_5 = predicted_5.t() # predicted_5.t() == transpose(predicted_5,0,1) -> .t() 的期望输入<=2维, 转置后返回, 对于0维和1维输入返回和输入一样的tensor predicted_5.size([5,256])
        correct_ = predicted_5.eq(targets.view(1, -1).expand_as(predicted_5))
        '''
        targets.size([256]) ->
        targets.view(1, -1).size([1,256]) ->
        targets.view(1, -1).expand_as(predicted_5).size([5,256]) ->
        predicted_5.eq(targets.view(1, -1).expand_as(predicted_5)).size([5,256]) ->
        corect_.size([5,256]) -> boolean[5,256]
        '''
        correct_5 += correct_[:5].reshape(-1).float().sum(0, keepdim=True).item()
        '''
        correct_[:5].size([5,256]) == correct_[0:5,:].size([5,256]) ->
        correct_[:5].reshape(-1).size([1280]) !=  correct_[:5].reshape(1,-1).size([1,1280]) ->
        correct_[:5].reshape(-1).float(True -> 1,0; False -> 0.0) ->
        correct_[:5].reshape(-1).float().sum(0, keepdim=True).size([1]) ->
        correct_[:5].reshape(-1).float().sum(0, keepdim=True).item() -> float (取单元素tensor的value,累积到correct_5)
        '''

        '''
        3. log 并 print 相关信息
        '''

        if batch_idx % 10 == 0 or batch_idx == len(testloader) - 1:
            logger.info("test: [batch: %d/%d ] | Loss: %.3f | Acc: %.3f%% (%d/%d) / %.3f%% (%d/%d)"
                        % (batch_idx, len(testloader), test_loss/(batch_idx+1), 100.*correct/total, correct, total, 100.*correct_5/total, correct_5, total))
        ''' 
        " test: [batch: 当前batch索引/总batch个数] | Loss: 平均每个batch的交叉熵损失 | Acc: Top-1正确率 / Top-5正确率 "
        '''
        # ave_loss = test_loss/total 无效参数计算注释掉了

    '''
    log 并 print 最终Top-1正确率
    '''
    acc = 100.*correct/total
    logger.info("Final accuracy: %.3f" % acc)

### Load validation data
logger.info('==> Preparing data..')
'''
from torch.utils.data DataLoader 通过调用Dataloader 返回一个可迭代对象，DataLoader本质上就是一个iterable
(跟python的内置类型list等一样)，并利用多进程来加速batch data的处理，使用yield来使用有限的内存
'''
testloader = getTestData(dataset=args.dataset,
                        batch_size=args.batch_size,
                        path=args.dataset_path,
                        for_inception=args.model.startswith('inception'))

### validation
test(quantized_model)