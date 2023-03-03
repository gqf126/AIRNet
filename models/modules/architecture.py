import math
import torch
import torch.nn as nn
import torchvision
import functools
from . import block as B
from . import spectral_norm as SN

####################
# Generator
####################

class SRResNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=1, norm_type='batch', act_type='relu', \
            mode='NAC', res_scale=1, upsample_mode='upconv'):
        super(SRResNet, self).__init__()
        # n_upscale = int(math.log(upscale, 2))
        n_upscale = 2
        if upscale == 3:
            n_upscale = 1

        self.fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        self.fea_conv1 = B.conv_block(nf, nf, kernel_size=4, stride=2, norm_type=None, act_type=None)
        self.fea_conv2 = B.conv_block(nf, nf, kernel_size=4, stride=2, norm_type=None, act_type=None)
        resnet_blocks = [B.ResNetBlock(nf, nf, nf, norm_type=norm_type, act_type=act_type,\
            mode=mode, res_scale=res_scale) for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)
        self.res_conv = B.sequential(*resnet_blocks, LR_conv)
        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            # upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
            self.upsampler1 = upsample_block(nf, nf, act_type=act_type)
            self.upsampler2 = upsample_block(nf, nf, act_type=act_type)
        self.HR_conv0 = B.conv_block(nf*2, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)
        
        # self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks, LR_conv)),\
        #     *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        # x = self.model(x)
        fea = self.fea_conv(x)
        fea1 = self.fea_conv1(fea)
        fea2 = self.fea_conv2(fea1)
        res = self.res_conv(fea2)
        out = fea2 + res
        out1 = torch.cat((fea2,out),1)
        up_out1 = self.upsampler1(out1)
        out2 = torch.cat((fea1,up_out1),1)
        up_out2 = self.upsampler2(out2) 
        hr_out = torch.cat((fea,up_out2),1)
        hout0 = self.HR_conv0(hr_out)
        hout1 = self.HR_conv1(hout0) 

        return hout1


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, \
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv'):
        super(RRDBNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x


####################
# Discriminator
####################


# VGG style Discriminator with input size 128*128
class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_128, self).__init__()
        # features
        # hxw, c
        # 128, 64
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 64, 64
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 32, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 16, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 8, 512
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 4, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# VGG style Discriminator with input size 128*128, Spectral Normalization
class Discriminator_VGG_128_SN(nn.Module):
    def __init__(self):
        super(Discriminator_VGG_128_SN, self).__init__()
        # features
        # hxw, c
        # 128, 64
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.conv0 = SN.spectral_norm(nn.Conv2d(3, 64, 3, 1, 1))
        self.conv1 = SN.spectral_norm(nn.Conv2d(64, 64, 4, 2, 1))
        # 64, 64
        self.conv2 = SN.spectral_norm(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv3 = SN.spectral_norm(nn.Conv2d(128, 128, 4, 2, 1))
        # 32, 128
        self.conv4 = SN.spectral_norm(nn.Conv2d(128, 256, 3, 1, 1))
        self.conv5 = SN.spectral_norm(nn.Conv2d(256, 256, 4, 2, 1))
        # 16, 256
        self.conv6 = SN.spectral_norm(nn.Conv2d(256, 512, 3, 1, 1))
        self.conv7 = SN.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 8, 512
        self.conv8 = SN.spectral_norm(nn.Conv2d(512, 512, 3, 1, 1))
        self.conv9 = SN.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 4, 512

        # classifier
        self.linear0 = SN.spectral_norm(nn.Linear(512 * 4 * 4, 100))
        self.linear1 = SN.spectral_norm(nn.Linear(100, 1))

    def forward(self, x):
        x = self.lrelu(self.conv0(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = self.lrelu(self.conv8(x))
        x = self.lrelu(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear0(x))
        x = self.linear1(x)
        return x


class Discriminator_VGG_96(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_96, self).__init__()
        # features
        # hxw, c
        # 96, 64
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 48, 64
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 24, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 12, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 6, 512
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Discriminator_VGG_192(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_192, self).__init__()
        # features
        # hxw, c
        # 192, 64
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 96, 64
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 48, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 24, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 12, 512
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 6, 512
        conv10 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv11 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9, conv10, conv11)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Discriminator_VGG_256(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_256, self).__init__()
        # features
        # hxw, c
        # 192, 64
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 96, 64
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 48, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 24, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 12, 512
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 6, 512
        conv10 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv11 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9, conv10, conv11)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

####################
# Perceptual Network
####################


# Assume input range is [0, 1]
class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
            # model = torchvision.models.vgg19(pretrained=False)
            # model.load_state_dict(torch.load('/home/lsl/.torch/models/vgg19-dcbb9e9d.pth'))
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


# Assume input range is [0, 1]
class ResNet101FeatureExtractor(nn.Module):
    def __init__(self, use_input_norm=True, device=torch.device('cpu')):
        super(ResNet101FeatureExtractor, self).__init__()
        model = torchvision.models.resnet101(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.children())[:8])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


class MINCNet(nn.Module):
    def __init__(self):
        super(MINCNet, self).__init__()
        self.ReLU = nn.ReLU(True)
        self.conv11 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv21 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv22 = nn.Conv2d(128, 128, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv31 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv32 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv33 = nn.Conv2d(256, 256, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv41 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv42 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv43 = nn.Conv2d(512, 512, 3, 1, 1)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv51 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv52 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv53 = nn.Conv2d(512, 512, 3, 1, 1)

    def forward(self, x):
        out = self.ReLU(self.conv11(x))
        out = self.ReLU(self.conv12(out))
        out = self.maxpool1(out)
        out = self.ReLU(self.conv21(out))
        out = self.ReLU(self.conv22(out))
        out = self.maxpool2(out)
        out = self.ReLU(self.conv31(out))
        out = self.ReLU(self.conv32(out))
        out = self.ReLU(self.conv33(out))
        out = self.maxpool3(out)
        out = self.ReLU(self.conv41(out))
        out = self.ReLU(self.conv42(out))
        out = self.ReLU(self.conv43(out))
        out = self.maxpool4(out)
        out = self.ReLU(self.conv51(out))
        out = self.ReLU(self.conv52(out))
        out = self.conv53(out)
        return out


# Assume input range is [0, 1]
class MINCFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True, \
                device=torch.device('cpu')):
        super(MINCFeatureExtractor, self).__init__()

        self.features = MINCNet()
        self.features.load_state_dict(
            torch.load('../experiments/pretrained_models/VGG16minc_53.pth'), strict=True)
        self.features.eval()
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        output = self.features(x)
        return output

# Assume input range is [0, 1]
class ResNet18_Y(nn.Module):
    def __init__(self, output_dim, use_input_norm=True, device=torch.device('cpu')):
        super(ResNet18_Y, self).__init__()
        model = torchvision.models.resnet18(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.children())[:8])
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
        self.cls = nn.Sequential(
            nn.Linear(in_features=512, out_features=128, bias=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=output_dim, bias=True),
            # nn.ReLU(True)
            )
        # No need to BP to variable
        # for k, v in self.features.named_parameters():
        #     v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        fea = self.features(x)
        fea = self.avgpool(fea)
        output = self.cls(fea.view(fea.shape[0], -1))
        return output

class ResNet50_Y(nn.Module):
    def __init__(self, output_dim, use_input_norm=True, device=torch.device('cpu')):
        super(ResNet50_Y, self).__init__()
        model = torchvision.models.resnet50(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.children())[:8])
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
        self.cls = nn.Sequential(
            nn.Linear(in_features=2048, out_features=128, bias=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=output_dim, bias=True)
            )
        # No need to BP to variable
        # for k, v in self.features.named_parameters():
        #     v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        fea = self.features(x)
        fea = self.avgpool(fea)
        output = self.cls(fea.view(fea.shape[0], -1))
        return output

# Assume input range is [0, 1]
class ResNet18_UV(nn.Module):
    def __init__(self, output_dim, use_input_norm=False, device=torch.device('cpu')):
        super(ResNet18_UV, self).__init__()
        model = torchvision.models.resnet18(pretrained=False)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(model.children())[1:8])
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
        self.cls = nn.Sequential(
            nn.Linear(in_features=512, out_features=128, bias=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=output_dim, bias=True)
            )
        # No need to BP to variable
        # for k, v in self.features.named_parameters():
        #     v.requires_grad = False

    def forward(self, x):
        # if self.use_input_norm:
        #     x = (x - self.mean) / self.std
        # print(x[0].shape)
        # fds
        fea = self.features(x)
        # fea = self.features(torch.cat((x[0], x[1]), 1))
        fea = self.avgpool(fea)
        output = self.cls(fea.view(fea.shape[0], -1))
        return output


# Assume input range is [0, 1]
class ResNet18_Gamma(nn.Module):
    def __init__(self, output_dim, use_input_norm=True, device=torch.device('cpu')):
        super(ResNet18_Gamma, self).__init__()
        model = torchvision.models.resnet18(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.children())[:8])
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
        self.cls = nn.Sequential(
            nn.Linear(in_features=512, out_features=128, bias=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=output_dim, bias=True),
            nn.Sigmoid()
            )
        # No need to BP to variable
        # for k, v in self.features.named_parameters():
        #     v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        fea = self.features(x)
        fea = self.avgpool(fea)
        output = self.cls(fea.view(fea.shape[0], -1))
        return output


# Assume input range is [0, 1]
class ResNet_Y(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, k_avg, norm_type, out_act=None, device=torch.device('cpu')):
        super(ResNet_Y, self).__init__()
        # model = torchvision.models.resnet18(pretrained=False)
        # self.use_input_norm = use_input_norm
        # if self.use_input_norm:
        #     mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        #     # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
        #     std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        #     # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
        #     self.register_buffer('mean', mean)
        #     self.register_buffer('std', std)
        self.features1 = nn.Sequential(
            nn.Conv2d(in_nc, nf, kernel_size=7, stride=2, padding=3, bias=False),
            norm_type(nf),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        nf_max = 512
        # rb_block = [B.BasicResBlock_conv1_maxpool(min(nf*2**iter_temp, nf_max),
        #     min(nf*2**(iter_temp+1), nf_max), stride=2, downsample=2, norm_layer=norm_type) for iter_temp in range(nb)]
        
        iter_temp = 0
        rb_block = [B.BasicResBlock_conv1_maxpool(min(nf*2**iter_temp, nf_max),
            min(nf*2**(iter_temp+1), nf_max), stride=2, downsample=2, norm_layer=norm_type, pool_type='maxpool')]
        iter_temp = 1
        rb_block += [B.BasicResBlock_conv1(min(nf*2**iter_temp, nf_max),
            min(nf*2**(iter_temp+1), nf_max), stride=2, downsample=2, norm_layer=norm_type)]
        
        iter_temp = nb - 1
        nf_conv = min(nf*2**(iter_temp+1), nf_max)
        # print(rb_block)
        self.features2 = nn.Sequential(*rb_block)
        self.avgpool = nn.AvgPool2d(kernel_size=k_avg, stride=k_avg)
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if out_act == 'sigmoid':
            self.cls = nn.Sequential(
                # nn.Linear(in_features=nf_conv, out_features=256, bias=True),
                # nn.Dropout(0.5),
                nn.Linear(in_features=nf_conv, out_features=64, bias=True),
                nn.Dropout(0.5),
                nn.Linear(in_features=64, out_features=out_nc),
                nn.Sigmoid()
            )
        else:
            self.cls = nn.Sequential(
                # nn.Linear(in_features=nf_conv, out_features=256, bias=True),
                # nn.Dropout(0.5),
                nn.Linear(in_features=nf_conv, out_features=64, bias=True),
                nn.Dropout(0.5),
                nn.Linear(in_features=64, out_features=out_nc)
                )
        # No need to BP to variable
        # for k, v in self.features.named_parameters():
        #     v.requires_grad = False

    def forward(self, x):
        # if self.use_input_norm:
        #     x = (x - self.mean) / self.std
        # print(x[0].shape)
        # fds
        fea = self.features1(x)
        fea = self.features2(fea)
        # fea = self.features(torch.cat((x[0], x[1]), 1))
        fea = self.avgpool(fea)
        output = self.cls(fea.view(fea.shape[0], -1))
        return output


# Assume input range is [0, 1]
class ResNet_UV(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, k_avg, norm_type, out_act=None, device=torch.device('cpu')):
        super(ResNet_UV, self).__init__()
        # model = torchvision.models.resnet18(pretrained=False)
        # self.use_input_norm = use_input_norm
        # if self.use_input_norm:
        #     mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        #     # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
        #     std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        #     # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
        #     self.register_buffer('mean', mean)
        #     self.register_buffer('std', std)
        self.features1 = nn.Sequential(
            nn.Conv2d(in_nc, nf, kernel_size=7, stride=2, padding=3, bias=False),
            norm_type(nf),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        nf_max = 512
        # rb_block = [B.BasicResBlock_conv1_maxpool(min(nf*2**iter_temp, nf_max),
        #     min(nf*2**(iter_temp+1), nf_max), stride=2, downsample=2, norm_layer=norm_type) for iter_temp in range(nb)]

        iter_temp = 0
        rb_block = [B.BasicResBlock_conv1_maxpool(min(nf*2**iter_temp, nf_max),
            min(nf*2**(iter_temp+1), nf_max), stride=2, downsample=2, norm_layer=norm_type, pool_type='maxpool')]
        iter_temp = 1
        rb_block += [B.BasicResBlock_conv1(min(nf*2**iter_temp, nf_max),
            min(nf*2**(iter_temp+1), nf_max), stride=2, downsample=2, norm_layer=norm_type)]      

        iter_temp = nb - 1
        nf_conv = min(nf*2**(iter_temp+1), nf_max)
        # print(rb_block)
        self.features2 = nn.Sequential(*rb_block)
        self.avgpool = nn.AvgPool2d(kernel_size=k_avg, stride=k_avg)
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if out_act == 'sigmoid':
            self.cls = nn.Sequential(
                # nn.Linear(in_features=nf_conv, out_features=256, bias=True),
                # nn.Dropout(0.5),
                nn.Linear(in_features=nf_conv, out_features=64, bias=True),
                nn.Dropout(0.5),
                nn.Linear(in_features=64, out_features=out_nc),
                nn.Sigmoid()
            )
        else:
            self.cls = nn.Sequential(
                # nn.Linear(in_features=nf_conv, out_features=256, bias=True),
                # nn.Dropout(0.5),
                nn.Linear(in_features=nf_conv, out_features=64, bias=True),
                nn.Dropout(0.5),
                nn.Linear(in_features=64, out_features=out_nc)
                )
        # No need to BP to variable
        # for k, v in self.features.named_parameters():
        #     v.requires_grad = False

    def forward(self, x):
        # if self.use_input_norm:
        #     x = (x - self.mean) / self.std
        # print(x[0].shape)
        # fds
        fea = self.features1(x)
        fea = self.features2(fea)
        # fea = self.features(torch.cat((x[0], x[1]), 1))
        fea = self.avgpool(fea)
        output = self.cls(fea.view(fea.shape[0], -1))
        return output   


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [B.ResnetBlock2(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)             