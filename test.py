from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import time
from spiking_model import*
import torch.nn.utils.prune as prune
from visdom import Visdom
import numpy as np
import pandas as pd
data_path = './raw'
test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
snn_unquant = SCNN()
checkpoint = torch.load('./checkpoint/3prune_ckptspiking_model.t7')
snn_unquant.load_state_dict(checkpoint['net'])
snn_unquant.to("cpu")

snn = torch.quantization.quantize_dynamic(
    snn_unquant,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)
correct = 0
total = 0
start_time = time.time()
viz = Visdom()
viz.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='test loss&acc.',legend=['loss', 'acc.']))
global_step = 0
test_loss = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to("cpu")

        optimizer.zero_grad()
        outputs = snn(inputs)
        labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
        loss = criterion(outputs.cpu(), labels_)
        _, predicted = outputs.cpu().max(1)
        pred = outputs.argmax(dim=1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        global_step += 1
        if batch_idx %100 ==0:
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader),' Acc: %.5f' % acc)

        viz.line([[test_loss, correct / global_step]],
                    [global_step], win='test', update='append')
        viz.images(inputs.view(-1, 1, 28, 28), win='x')
        viz.text(str(pred.detach().cpu().numpy()), win='pred',opts=dict(title='pred'))
        

print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
print('total time: %.3f seconds' % (time.time() - start_time))


fp11 = open(file='./conv1.txt',mode='w')
fp12 = open(file='./aux1.txt',mode='w')
fp21 = open(file='./conv2.txt',mode='w')
fp22 = open(file='./aux2.txt', mode='w')
fp31 = open(file = './fc1.txt',mode='w')
fp32 = open(file='./aux3.txt', mode='w')
fp4 = open(file='./fc2.txt',mode='w')

pd.set_option('display.max_columns', None)   #显示完整的列
pd.set_option('display.max_rows', None)  #显示完整的行
print(snn.conv1.weight.detach().numpy(),file=fp11)
print(snn.aux1.weight.detach().numpy(),file=fp12)
print(snn.conv2.weight.detach().numpy(),file=fp21)
print(snn.aux2.weight.detach().numpy(),file=fp22)
print(snn.fc1.weight.detach().numpy(),file=fp31)
print(snn.aux3.weight.detach().numpy(),file=fp32)
print(snn.fc2.weight.detach().numpy(),file=fp4)
fp11.close()
fp12.close()
fp21.close()
fp22.close()
fp31.close()
fp32.close()
fp4.close()
