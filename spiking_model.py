import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = 0.5 # neuronal threshold
lens = 0.5 # hyper-parameters of approximate function
decay = 0.2 # decay constants
num_classes = 10
batch_size  = 100
learning_rate = 1e-3
num_epochs = 1 # max epoch
aux_decay = 0.2 # decay constant for auxiliary neurons
v = 0.9#v & f are constants defined for the updates of aux-neurons. Note that aux_decay + vf == 1.
f = 0.9
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
    mem = mem * decay * (1. - spike) + ops(x) - aux_ops(x) #待修改！！
    spike = act_fun(mem) # act_fun : approximation firing function
    return mem, spike

# auxiliary neurons membrane potential update/待修改！不一定管用
def aux_mem_update(ops ,x, mem):
    mem = mem * aux_decay + (v/f) * ops(x)
    return mem
    
def original_mem_update(ops, x, mem, spike):
    mem = mem * decay * (1. - spike) + ops(x)
    spike = act_fun(mem) # act_fun : approximation firing function
    return mem, spike
# cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
cfg_cnn = [(1, 32, 1, 1, 3),
           (32, 32, 1, 1, 3),]
# kernel size
cfg_kernel = [28, 14, 7]
# fc layer
cfg_fc = [128, 10]

# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer

class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.aux1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.aux2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        self.fc1 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])
        self.aux3 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.aux4 = nn.Linear(cfg_fc[0], cfg_fc[1])

    def forward(self, input, time_window = 20):
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)
        a1_mem = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        a2_mem = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)
        a3_mem = torch.zeros(batch_size, cfg_fc[0], device=device)
        #所有a开头的膜电位都是aux neuron，待修改！
        for step in range(time_window): # simulation time steps
            x = input > torch.rand(input.size(), device=device) # prob. firing
            tmp = x
            #待修改！！
            a1_mem = aux_mem_update(self.aux1, tmp.float(), a1_mem)
            
            c1_mem, c1_spike = mem_update(self.conv1, self.aux1, x.float(), c1_mem, c1_spike)

            x = F.avg_pool2d(c1_spike, 2)

            tmp = x
            #待修改↕
            a2_mem = aux_mem_update(self.aux2, tmp.float(), a2_mem)

            c2_mem, c2_spike = mem_update(self.conv2, self.aux2, x, c2_mem,c2_spike)

            x = F.avg_pool2d(c2_spike, 2)
            x = x.view(batch_size, -1)

            #待修改！！
            tmp = x
            a3_mem = aux_mem_update(self.aux3, tmp.float(), a3_mem)

            h1_mem, h1_spike = mem_update(self.fc1, self.aux3, x, h1_mem, h1_spike)
            h1_sumspike += h1_spike

            h2_mem, h2_spike = original_mem_update(self.fc2, h1_spike, h2_mem,h2_spike)
            h2_sumspike += h2_spike

        outputs = h2_sumspike / time_window
        return outputs