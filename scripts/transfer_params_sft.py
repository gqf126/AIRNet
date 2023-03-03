import torch
from torch.nn import init

pretrained_net = torch.load('../../experiments/pretrained_models/latest_G.pth')
# should run train debug mode first to get an initial model
crt_net = torch.load('../../experiments/pretrained_models/72_G.pth')

for k, v in crt_net.items():
    if 'weight' in k:
        print(k, 'weight')
        init.kaiming_normal(v, a=0, mode='fan_in')
        v *= 0.1
    elif 'bias' in k:
        print(k, 'bias')
        v.fill_(0)
for k, v in pretrained_net.items():
    if 'weight' in k:
        print(k, 'weight')
        init.kaiming_normal(v, a=0, mode='fan_in')
        v *= 0.1
    elif 'bias' in k:
        print(k, 'bias')
        v.fill_(0)

crt_net['conv0.weight'] = pretrained_net['fea_conv.0.weight']
crt_net['conv0.bias'] = pretrained_net['fea_conv.0.bias']
crt_net['conv1.weight'] = pretrained_net['fea_conv1.0.weight']
crt_net['conv1.bias'] = pretrained_net['fea_conv1.0.bias']
crt_net['conv2.weight'] = pretrained_net['fea_conv2.0.weight']
crt_net['conv2.bias'] = pretrained_net['fea_conv2.0.bias']
# residual blocks
for i in range(16):
    crt_net['sft_branch.{:d}.conv0.weight'.format(i)] = pretrained_net['res_conv.{:d}.res.0.weight'.format(i)]
    crt_net['sft_branch.{:d}.conv0.bias'.format(i)] = pretrained_net['res_conv.{:d}.res.0.bias'.format(i)]
    crt_net['sft_branch.{:d}.conv1.weight'.format(i)] = pretrained_net['res_conv.{:d}.res.2.weight'.format(i)]
    crt_net['sft_branch.{:d}.conv1.bias'.format(i)] = pretrained_net['res_conv.{:d}.res.2.bias'.format(i)]

crt_net['sft_branch.17.weight'] = pretrained_net['res_conv.16.weight']
crt_net['sft_branch.17.bias'] = pretrained_net['res_conv.16.bias']

# HR
crt_net['HR_branch1.0.weight'] = pretrained_net['upsampler1.0.weight']
crt_net['HR_branch1.0.bias'] = pretrained_net['upsampler1.0.bias']
crt_net['HR_branch2.0.weight'] = pretrained_net['upsampler2.0.weight']
crt_net['HR_branch2.0.bias'] = pretrained_net['upsampler2.0.bias']
crt_net['HR_branch3.0.weight'] = pretrained_net['HR_conv0.0.weight']
crt_net['HR_branch3.0.bias'] = pretrained_net['HR_conv0.0.bias']
crt_net['HR_branch3.2.weight'] = pretrained_net['HR_conv1.0.weight']
crt_net['HR_branch3.2.bias'] = pretrained_net['HR_conv1.0.bias']

print('OK. \n Saving model...')
torch.save(crt_net, '../../experiments/pretrained_models/sft_net_ini.pth')