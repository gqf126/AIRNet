'''
architecture for sft
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools


class SFTLayer(nn.Module):
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        # print(x[0].shape, x[1].shape)
        return x[0] * (scale + 1) + shift


class ResBlock_SFT(nn.Module):
    def __init__(self):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer()
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.sft1 = SFTLayer()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = self.sft0(x)
        # fea = self.conv0(x[0])
        fea = F.relu(self.conv0(fea), inplace=True)
        fea = self.sft1((fea, x[1]))
        fea = self.conv1(fea)
        return (x[0] + fea, x[1])  # return a tuple containing features and conditions
        # return (x[0] + fea)


class SFT_Net(nn.Module):
    def __init__(self):
        super(SFT_Net, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv1 = nn.Conv2d(64, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 64, 4, 2, 1)

        sft_branch = []
        for i in range(16):
            sft_branch.append(ResBlock_SFT())
        sft_branch.append(SFTLayer())
        sft_branch.append(nn.Conv2d(64, 64, 3, 1, 1))
        self.sft_branch = nn.Sequential(*sft_branch)

        self.HR_branch1 = nn.Sequential(
            nn.Conv2d(64*2, 64*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(True)
        )
        self.HR_branch2 = nn.Sequential(
            nn.Conv2d(64*2, 64*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(True)
        )
        self.HR_branch3 = nn.Sequential(
            nn.Conv2d(64*2, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, 1, 1)
        )
        # self.HR_branch = nn.Sequential(
        #     nn.Conv2d(64, 256, 3, 1, 1),
        #     nn.PixelShuffle(2),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 256, 3, 1, 1),
        #     nn.PixelShuffle(2),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 64, 3, 1, 1),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 3, 3, 1, 1)
        # )

        self.CondNet = nn.Sequential(
            nn.Conv2d(8, 128, 4, 4),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 32, 1)
        )

    def forward(self, x):
        # x[0]: img; x[1]: seg
        cond = self.CondNet(x[1])
        fea = self.conv0(x[0])
        fea1 = self.conv1(fea)
        fea2 = self.conv2(fea1)
        # print('#########',fea2.shape,cond.shape)
        # res = self.sft_branch((fea, cond))
        res = self.sft_branch((fea2, cond))
        # res = self.sft_branch(fea)
        # fea = fea + res
        fea3 = fea2 + res
        # out = self.HR_branch(fea)
        out = torch.cat((fea2,fea3),1)
        out1 = self.HR_branch1(out)
        out2 = torch.cat((fea1,out1),1)
        out3 = self.HR_branch2(out2)
        out_hr = torch.cat((fea,out3),1)
        out4 = self.HR_branch3(out_hr)
        return out4

###SFT_UV
class SFT_UV_Layer(nn.Module):
    def __init__(self, inplanes):
        super(SFT_UV_Layer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(128, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, inplanes, 1)
        self.SFT_shift_conv0 = nn.Conv2d(128, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, inplanes, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        # print(x[0].shape, x[1].shape)
        # print(x[0].shape, scale.shape, shift.shape)
        return x[0] * (scale + 1) + shift


class ResBlock_UV_SFT(nn.Module):
    def __init__(self, inplanes):
        super(ResBlock_UV_SFT, self).__init__()
        self.sft0 = SFT_UV_Layer(inplanes)
        self.conv0 = nn.Sequential(nn.Conv2d(inplanes, inplanes, 3, 1, 1),
                                   nn.ReLU(True),
                                   nn.BatchNorm2d(inplanes))
        self.sft1 = SFT_UV_Layer(inplanes)
        self.conv1 = nn.Sequential(nn.Conv2d(inplanes, inplanes, 3, 1, 1),
                                   nn.BatchNorm2d(inplanes))

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = self.sft0(x)
        # fea = self.conv0(x[0])
        fea = self.conv0(fea)
        fea = self.sft1((fea, x[1]))
        fea = self.conv1(fea)
        return (x[0] + fea, x[1])  # return a tuple containing features and conditions
        # return (x[0] + fea)

class ResBlock_strided2_UV_SFT(nn.Module):
    def __init__(self, inplanes):
        super(ResBlock_strided2_UV_SFT, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(inplanes, inplanes*2, 3, 2, 1),
                                   nn.ReLU(True),
                                   nn.BatchNorm2d(inplanes*2))
        self.sft0 = SFT_UV_Layer(inplanes*2)
        self.conv1 = nn.Sequential(nn.Conv2d(inplanes*2, inplanes*2, 3, 1, 1),
                                   nn.ReLU(True),
                                   nn.BatchNorm2d(inplanes*2))
        self.sft1 = SFT_UV_Layer(inplanes*2)
        self.conv2 = nn.Sequential(nn.Conv2d(inplanes*2, inplanes*2, 3, 1, 1),
                                   nn.BatchNorm2d(inplanes*2))
        self.downsample = nn.Sequential(nn.Conv2d(inplanes, inplanes*2, 1, 2),
                                        nn.BatchNorm2d(inplanes*2))

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = self.conv0(x[0])
        fea = self.sft0((fea, x[1]))
        # fea = self.conv0(x[0])
        fea = self.conv1(fea)
        fea = self.sft1((fea, x[1]))
        fea = self.conv2(fea)
        identity = self.downsample(x[0])
        return (identity + fea, x[1])  # return a tuple containing features and conditions
        # return (x[0] + fea)

class SFT_UV_Net(nn.Module):
    def __init__(self):
        super(SFT_UV_Net, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.CondNet_down4 = nn.Sequential(
            nn.Conv2d(8, 128, 4, 4),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
        )
        self.sft_branch1 = ResBlock_UV_SFT(64)
        self.sft_branch2 = ResBlock_UV_SFT(64)
        self.CondNet_down8 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
        )
        self.sft_branch3 = ResBlock_strided2_UV_SFT(64)
        self.sft_branch4 = ResBlock_UV_SFT(128)
        self.CondNet_down16 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
        )
        self.sft_branch5 = ResBlock_strided2_UV_SFT(128)
        self.sft_branch6 = ResBlock_UV_SFT(256)
        self.CondNet_down32 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
        )
        self.sft_branch7 = ResBlock_strided2_UV_SFT(256)
        self.sft_branch8 = ResBlock_UV_SFT(512)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
        self.cls = nn.Sequential(
            nn.Linear(in_features=512, out_features=128, bias=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=6, bias=True)
            )


    def forward(self, x):
        # x[0]: img; x[1]: seg
        fea = self.conv0(x[0])
        cond = self.CondNet_down4(x[1])
        fea = self.sft_branch1((fea, cond))
        fea = self.sft_branch2(fea)
        # print(fea[0].shape)
        cond = self.CondNet_down8(cond)
        # print(cond.shape)
        fea = self.sft_branch3((fea[0], cond))
        fea = self.sft_branch4(fea)

        cond = self.CondNet_down16(cond)
        fea = self.sft_branch5((fea[0], cond))
        fea = self.sft_branch6(fea)

        cond = self.CondNet_down32(cond)
        fea = self.sft_branch7((fea[0], cond))
        fea = self.sft_branch8(fea)
        fea = self.avgpool(fea[0])
        out = self.cls(fea.view(fea.shape[0], -1))
        return out



# Auxiliary Classifier Discriminator
class ACD_VGG_BN_96(nn.Module):
    def __init__(self):
        super(ACD_VGG_BN_96, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.1, True),
            
            nn.AvgPool2d(kernel_size=5, stride=5)
#             nn.Conv2d(512, 512, 4, 2, 1),
#             nn.BatchNorm2d(512, affine=True),
#             nn.LeakyReLU(0.1, True),

#             nn.Conv2d(512, 512, 4, 2, 1),
#             nn.BatchNorm2d(512, affine=True),
#             nn.LeakyReLU(0.1, True),

#             nn.Conv2d(512, 512, 4, 2, 1),
#             nn.BatchNorm2d(512, affine=True),
#             nn.LeakyReLU(0.1, True)
        )

        # gan
        self.gan = nn.Sequential(
            nn.Linear(512*1*1, 100),
            nn.LeakyReLU(0.1, True),
            nn.Linear(100, 1)
        )

        self.cls = nn.Sequential(
            nn.Linear(512*1*1, 100),
            nn.LeakyReLU(0.1, True),
            nn.Linear(100, 8)
        )

    def forward(self, x):
        fea = self.feature(x)
        fea = fea.view(fea.size(0), -1)
        gan = self.gan(fea)
        cls = self.cls(fea)
        return [gan, cls]


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

#############################################
# below is the sft arch for the torch version
#############################################


class SFTLayer_torch(nn.Module):
    def __init__(self):
        super(SFTLayer_torch, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.01, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.01, inplace=True))
        return x[0] * scale + shift


class ResBlock_SFT_torch(nn.Module):
    def __init__(self):
        super(ResBlock_SFT_torch, self).__init__()
        self.sft0 = SFTLayer_torch()
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.sft1 = SFTLayer_torch()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = F.relu(self.sft0(x), inplace=True)
        fea = self.conv0(fea)
        fea = F.relu(self.sft1((fea, x[1])), inplace=True)
        fea = self.conv1(fea)
        return (x[0] + fea, x[1])  # return a tuple containing features and conditions


class SFT_Net_torch(nn.Module):
    def __init__(self):
        super(SFT_Net_torch, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, 3, 1, 1)

        sft_branch = []
        for i in range(16):
            sft_branch.append(ResBlock_SFT_torch())
        sft_branch.append(SFTLayer_torch())
        sft_branch.append(nn.Conv2d(64, 64, 3, 1, 1))
        self.sft_branch = nn.Sequential(*sft_branch)

        self.HR_branch = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, 1, 1)
        )

        # Condtion network
        self.CondNet = nn.Sequential(
            nn.Conv2d(8, 128, 4, 4),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 32, 1)
        )

    def forward(self, x):
        # x[0]: img; x[1]: seg
        cond = self.CondNet(x[1])
        fea = self.conv0(x[0])
        res = self.sft_branch((fea, cond))
        fea = fea + res
        out = self.HR_branch(fea)
        return out
