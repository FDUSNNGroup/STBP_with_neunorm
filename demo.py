from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import time
from spiking_model import*
import torch.nn.utils.prune as prune
from visdom import Visdom
import cv2

data_path = './raw'
test_set = cv2.imread('D:/vscode/SNN_with_pruning/testfile.jpg',flags=cv2.IMREAD_GRAYSCALE)
print(type(test_set))
test_set = torch.from_numpy(test_set).view(1,1,28,28)
print(test_set.size())
snn = SCNN()
checkpoint = torch.load('./checkpoint/prune_ckptspiking_model.t7')
snn.load_state_dict(checkpoint['net'])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)
correct = 0
total = 0
start_time = time.time()
output = snn(test_set)
print(output)