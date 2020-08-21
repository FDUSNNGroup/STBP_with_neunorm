from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
import os
import time
from spiking_model import*
snn = SCNN()
data_path = './raw'
checkpoint = torch.load('./checkpoint/prune_ckptspiking_model.t7')
snn.load_state_dict(checkpoint['net'])

test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

for i,data in enumerate(test_loader):
    inputs,labels = data

script_module = torch.jit.trace(snn,inputs)
script_module.save("torch_script_module.pt")