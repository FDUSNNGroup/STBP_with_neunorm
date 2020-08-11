from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import time
from spiking_model import*
import torch.nn.utils.prune as prune
data_path = './raw'
test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
snn = SCNN()
checkpoint = torch.load('./checkpoint/ckptspiking_model.t7')
snn.load_state_dict(checkpoint['net'])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)
correct = 0
total = 0

#pruning
parameters_to_prune = (
    (snn.conv1, 'weight'),
    (snn.conv2, 'weight'),
    (snn.fc1, 'weight'),
    (snn.fc2, 'weight'),
)

prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.3)
#pruning completed
start_time = time.time()
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = snn(inputs)
        labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
        loss = criterion(outputs.cpu(), labels_)
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        if batch_idx %100 ==0:
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader),' Acc: %.5f' % acc)


print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
print('total time: %.3f seconds' % (time.time() - start_time))