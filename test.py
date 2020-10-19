from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import time
from spiking_model import*
import torch.nn.utils.prune as prune
from visdom import Visdom

data_path = './raw'
test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
snn = SCNN()
checkpoint = torch.load('./checkpoint/prune_ckptspiking_model.t7')
snn.load_state_dict(checkpoint['net'])
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
        inputs = inputs.to(device)
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
