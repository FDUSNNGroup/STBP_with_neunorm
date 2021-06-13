import torch
import torch.nn as nn
import torch.nn.functional as F
#from visdom import Visdom
import numpy as np
#viz = Visdom()
device = torch.device( "cpu")
thresh = 0.5 # neuronal threshold
lens = 0.5 # hyper-parameters of approximate function
decay = 0.6 # decay constants
num_classes = 10
batch_size  = 16
learning_rate = 1e-4
num_epochs = 20 # max epoch
#aux_decay = 0.25 # decay constant for auxiliary neurons

# define approximate firing function
class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply
# membrane potential update

def mem_update(ops, aux_ops, x, mem, spike):
    tmp = ops(x) - 0.1*aux_ops(x)
    tmp = torch.relu(tmp)
    mem = mem * decay * (1. - spike) + tmp
    spike = act_fun(mem) # act_fun : approximation firing function
    return mem, spike

def original_mem_update(ops, x, mem, spike): #无辅助层情况下的膜电位更新，仅输出层使用
    mem = mem * decay * (1. - spike) + ops(x)
    spike = act_fun(mem) # act_fun : approximation firing function
    return mem, spike
# cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
cfg_cnn = [(3, 32, 1, 1, 3),
           (32, 64, 1, 1, 3),
           (64, 32, 1, 1, 3),
           (32, 1, 1, 0, 1)]
# kernel size
cfg_kernel = [32, 16, 8, 8]
# fc layer
cfg_fc = [10]

# Dacay learning_rate

def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.5 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5
    return optimizer

class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.aux1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.aux2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[2]
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.aux3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[3]
        self.conv4 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.aux4 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.fc = nn.Linear(cfg_cnn[-1][1]*cfg_kernel[-1]*cfg_kernel[-1], cfg_fc[0], bias=False)
 
    def forward(self, input, time_window = 20):
        c1_mem = 0.5*torch.ones(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_mem = 0.5*torch.ones(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)
        c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)
        #c3_mem = c3_spike = torch.zeros(batch_size, cfg_cnn[2][1], cfg_kernel[2], cfg_kernel[2], device=device)
        c3_spike = torch.zeros(batch_size, cfg_cnn[2][1], cfg_kernel[2], cfg_kernel[2], device=device)
        c3_mem = 0.5*torch.ones(batch_size, cfg_cnn[2][1], cfg_kernel[2], cfg_kernel[2], device=device)
        c4_spike = torch.zeros(batch_size, cfg_cnn[3][1], cfg_kernel[2], cfg_kernel[2], device=device)
        c4_mem = 0.5*torch.ones(batch_size, cfg_cnn[3][1], cfg_kernel[2], cfg_kernel[2], device=device)
        h_spike = h_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h_mem = 0.5*torch.ones(batch_size, cfg_fc[0], device=device)
        '''
        fid1 = open(file='D:/vscode/SNN_with_pruning/conv1_mem.txt',mode='w')
        fid2 = open(file='D:/vscode/SNN_with_pruning/conv1_spike.txt',mode='w')
        fid3 = open(file='D:/vscode/SNN_with_pruning/conv2_mem.txt',mode='w')
        fid4 = open(file='D:/vscode/SNN_with_pruning/conv2_spike.txt',mode='w')
        '''
        for step in range(time_window): # simulation time steps
            #encode the input
            x = input > torch.rand(input.size(), device=device) # prob. firing
            
            c1_mem, c1_spike = mem_update(self.conv1, self.aux1, x.float(), c1_mem, c1_spike)
            '''
            print('step: %d'%step, file=fid1)
            print('step: %d'%step, file=fid2)
            print('step: %d'%step, file=fid3)
            print('step: %d'%step, file=fid4)
            print(c1_mem.detach().numpy(),file=fid1)
            print(c1_spike.detach().numpy(),file=fid2)
            '''
            x = F.max_pool2d(c1_spike, 2)
            
            c2_mem, c2_spike = mem_update(self.conv2, self.aux2, x, c2_mem,c2_spike)
            '''
            print(c2_mem.detach().numpy(),file=fid3)
            print(c2_spike.detach().numpy(),file=fid4)
            '''
            
            x = F.max_pool2d(c2_spike, 2)
            
            c3_mem, c3_spike = mem_update(self.conv3, self.aux3, x, c3_mem, c3_spike)
            c4_mem, c4_spike = mem_update(self.conv4, self.aux4, c3_spike, c4_mem, c4_spike)
            x = c4_spike
            x = x.view(batch_size, -1)

           
            
            h_mem, h_spike = original_mem_update(self.fc, x, h_mem,h_spike)
            h_sumspike += h_spike
            """
            for i in [10]:
                emit1_single = h1_spike[i].numpy()[0]
                emit1_idx = (np.where(h1_spike[i].numpy() == 1))[0]
                emit2_idx = np.where(h2_spike[i].numpy() == 1)[0]
                num_1 = len(emit1_idx)
                num_2 = len(emit2_idx)
                emit1_step = step*(np.ones(num_1))
                emit2_step = step*(np.ones(num_2))
                emit1_idx = np.r_[emit1_idx,emit1_step]
                emit2_idx = np.r_[emit2_idx,emit2_step].reshape(2,num_2).T
                emit1_idx = emit1_idx.reshape(2,num_1).T
                emit1_plt = torch.from_numpy(emit1_idx)
                emit2_plt = torch.from_numpy(emit2_idx)
                
                viz.scatter(emit1_plt,opts={'markersize':2, 'title':'fc_spike','layoutopts': {'plotly': {'xaxis': { 'range': [1, 128],'autorange': False},'yaxis': {'range': [0, 20],'autorange': False}}}},win='h1_spike',update='append')
                viz.scatter(emit2_plt,opts={'markersize':2, 'title':'output_spike','layoutopts': {'plotly': {'xaxis': { 'range': [1, 10],'autorange': False},'yaxis': {'range': [0, 20],'autorange': False}}}},win='h2_spike',update='append')
        """
        outputs = h_sumspike / time_window
        '''
        fid1.close()
        fid2.close()
        fid3.close()
        fid4.close()
        '''
        """
        clearset = torch.zeros(1,2)  
        viz.scatter(clearset,opts={'markersize':2, 'title':'fc1_spike generated by the 11th image','layoutopts': {'plotly': {'xaxis': { 'range': [1, 128],'autorange': False},'yaxis': {'range': [0, 20],'autorange': False}}}},win='h1_spike')
        viz.scatter(clearset,opts={'markersize':2, 'title':'output spike','layoutopts': {'plotly': {'xaxis': { 'range': [1, 10],'autorange': False},'yaxis': {'range': [0, 20],'autorange': False}}}},win='h2_spike')
        """
        return outputs
