from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import time
from spiking_model import*
#from visdom import Visdom
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
names = 'spiking_model'
data_path =  './cifar10/' #todo: input your data path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#transform.ToTensor: [0,255]numpy array->[0,1]tensor
test_set = torchvision.datasets.CIFAR10(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])
checkpoint = torch.load('./checkpoint/4ckptspiking_model.t7')
snn = SCNN()
snn.load_state_dict(checkpoint['net'])


snn.to(device)
#define loss function and optimizer
#snn.apply(para_init)
criterion = nn.MSELoss()
conv1_params = list(map(id, snn.conv1.parameters()))
conv2_params = list(map(id, snn.conv2.parameters()))
conv3_params = list(map(id, snn.conv3.parameters()))
linear_params = filter(lambda p: id(p) not in conv1_params + conv2_params + conv3_params, snn.parameters())
params = [{'params':linear_params},
        {'params':snn.conv1.parameters(), 'lr':learning_rate},
        {'params':snn.conv2.parameters(), 'lr':learning_rate/10},
        {'params':snn.conv3.parameters(), 'lr':learning_rate/100}]
optimizer = torch.optim.Adam(params, lr=learning_rate)
global_step = 0
#train network
for epoch in range(num_epochs):
    running_loss = 0
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        snn.zero_grad()
        optimizer.zero_grad()
        images = images.float().to(device)
        outputs = snn(images)
        labels_ = torch.zeros(batch_size, 10).scatter_(1, labels.view(-1, 1), 1)
        # print(labels)
        # print(labels_)
        # print("########\n",outputs-labels_)
        loss = criterion(outputs.cpu(), labels_)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        global_step += 1
        if (i+1)%100 == 0:
             print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                    %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size,running_loss ))
             running_loss = 0
             print('Time elasped:', time.time()-start_time)
        # if (i+1) %1000 == 0:
        #         print('Saving...')
        #         state = {
        #             'net': snn.state_dict()
        #         }
        #         torch.save(state, './checkpoint/4ckpt'+names+'.t7')
    correct = 0
    total = 0
    optimizer = lr_scheduler(optimizer, epoch, learning_rate, 5)

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
            
    print('Iters:', epoch,'\n\n\n')
    print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)
    if epoch % 1 == 0:
        print(acc)
        print('Saving..')
        state = {
            'net': snn.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_record,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/4ckpt' + names + '.t7')
        best_acc = acc
