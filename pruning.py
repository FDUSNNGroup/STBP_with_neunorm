import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from spiking_model import *
names = 'spiking_model'
snn = SCNN()
checkpoint = torch.load('./checkpoint/4ckptspiking_model.t7')
snn.load_state_dict(checkpoint['net'])
#pruning
parameters_to_prune = (
    (snn.conv1, 'weight'),
    (snn.conv2, 'weight'),
    (snn.fc1, 'weight'),
    (snn.fc2, 'weight'),
)

prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2)
for layers in parameters_to_prune:
    prune.remove(layers[0], layers[1])
#pruning completed
state = {
    'net':snn.state_dict()
}
torch.save(state, './checkpoint/3prune_ckpt' + names + '.t7')
