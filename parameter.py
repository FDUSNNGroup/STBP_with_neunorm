from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import time
from spiking_model import*
import torch.nn.utils.prune as prune
import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.inf)
snn = SCNN()
snn2 = SCNN()
checkpoint = torch.load('./checkpoint/4ckptspiking_model.t7')
checkpoint2 = torch.load('./checkpoint/3prune_ckptspiking_model.t7')
snn.load_state_dict(checkpoint['net'])
snn2.load_state_dict(checkpoint2['net'])

fp11 = open(file='./conv1.txt',mode='w')
fp12 = open(file='./aux1.txt',mode='w')
fp21 = open(file='./conv2.txt',mode='w')
fp22 = open(file='./aux2.txt', mode='w')
fp31 = open(file = './fc1.txt',mode='w')
fp32 = open(file='./aux3.txt', mode='w')
fp4 = open(file='./fc2.txt',mode='w')

pd.set_option('display.max_columns', None)   #显示完整的列
pd.set_option('display.max_rows', None)  #显示完整的行

print(snn2.conv1.weight.detach().numpy(),file=fp11)
print(snn2.aux1.weight.detach().numpy(),file=fp12)
print(snn2.conv2.weight.detach().numpy(),file=fp21)
print(snn2.aux2.weight.detach().numpy(),file=fp22)
print(snn2.fc1.weight.detach().numpy(),file=fp31)
print(snn2.aux3.weight.detach().numpy(),file=fp32)
print(snn2.fc2.weight.detach().numpy(),file=fp4)

fp11.close()
fp12.close()
fp21.close()
fp22.close()
fp31.close()
fp32.close()
fp4.close()

