import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import itertools
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from IPython import display
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
        #print('no display found. Using non-interactive Agg backend')
        mpl.use('Agg')
import matplotlib.pyplot as plt
import geffnet


def conv_org(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)

######################################################################################################################
######################################################################################################################

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

# pre-activation based Upsampling conv block
class upConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, norm, act, num_groups):
        super(upConvLayer, self).__init__()
        conv = conv_org
        if act == 'ELU':
            act = nn.ELU()
        else:
            act = nn.ReLU(True)
        self.conv = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if norm == 'GN':
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        else:
            self.norm = nn.BatchNorm2d(in_channels, momentum=0.01, affine=True, track_running_stats=True)
        self.act = act
        self.scale_factor = scale_factor
    def forward(self, x, shape=None):
        x = self.norm(x)
        x = self.act(x)     #pre-activation
        if shape is None:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        else:
            x = F.interpolate(x, shape, mode='bilinear')
        x = self.conv(x)
        return x

# pre-activation based conv block
class myConv(nn.Module):
    def __init__(self, in_ch, out_ch, kSize, stride=1, 
                    padding=0, dilation=1, bias=True, norm='GN', act='ELU', num_groups=32):
        super(myConv, self).__init__()
        conv = conv_org
        if act == 'ELU':
            act = nn.ELU()
        else:
            act = nn.ReLU(True)
        module = []
        if norm == 'GN': 
            module.append(nn.GroupNorm(num_groups=num_groups, num_channels=in_ch))
        else:
            module.append(nn.BatchNorm2d(in_ch, momentum=0.01, affine=True, track_running_stats=True))
        module.append(act)
        module.append(conv(in_ch, out_ch, kernel_size=kSize, stride=stride, 
                            padding=padding, dilation=dilation, bias=bias))
        self.module = nn.Sequential(*module)
    def forward(self, x):
        out = self.module(x)
        return out

# Deep Feature Fxtractor
class ResNet101_KD(nn.Module):
    def __init__(self,args):
        super(ResNet101_KD, self).__init__()
        self.args = args
        # after passing ReLU   : H/2  x W/2
        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        self.encoder = models.resnet101(pretrained=True)
        self.layerList = ['bn1','layer1','layer2','layer3']
        self.dimList = [64, 256, 512, 1024]
        del self.encoder.layer4
        del self.encoder.fc
    def forward(self, x):
        out_featList = []
        feature = x
        for k, v in self.encoder._modules.items():
            if k == 'avgpool':
                break
            feature = v(feature)
            #feature = v(features[-1])
            #features.append(feature)
            if any(x in k for x in self.layerList):
                out_featList.append(feature) 
        return out_featList
    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

class ResNet50_KD(nn.Module):
    def __init__(self,args):
        super(ResNet50_KD, self).__init__()
        self.args = args
        # after passing ReLU   : H/2  x W/2
        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        self.encoder = models.resnet50(pretrained=True)
        del self.encoder.fc
        self.layerList = ['relu','layer1','layer2','layer3', 'layer4']
        dimList = [64, 256, 512, 1024, 2048]
        out_dimList = [64, 128, 256, 512, 1024]
        self.out_dimList = out_dimList
        norm = 'BN'
        act = 'ReLU'

        convert1 = myConv(dimList[0], out_dimList[0], kSize=1, stride=1, padding=0, bias=False, 
                           norm=norm, act=act, num_groups=dimList[0]//16)
        convert2 = myConv(dimList[1], out_dimList[1], kSize=1, stride=1, padding=0, bias=False, 
                           norm=norm, act=act, num_groups=dimList[1]//16)
        convert3 = myConv(dimList[2], out_dimList[2], kSize=1, stride=1, padding=0, bias=False, 
                           norm=norm, act=act, num_groups=dimList[2]//16)
        convert4 = myConv(dimList[3], out_dimList[3], kSize=1, stride=1, padding=0, bias=False, 
                           norm=norm, act=act, num_groups=dimList[3]//16)
        convert5 = myConv(dimList[4], out_dimList[4], kSize=1, stride=1, padding=0, bias=False, 
                           norm=norm, act=act, num_groups=dimList[4]//16)
        self.conv_convert_list = nn.ModuleList([convert1,convert2,convert3,convert4,convert5])
        
    def forward(self, x):
        out_featList = []
        feature = x
        cnt = 0
        for k, v in self.encoder._modules.items():
            if k == 'avgpool':
                break
            feature = v(feature)
            #feature = v(features[-1])
            #features.append(feature)
            if any(x in k for x in self.layerList):
                converted_feat = self.conv_convert_list[cnt](feature)
                out_featList.append(converted_feat)
                cnt = cnt + 1 
        return out_featList
    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

class ResNext101_KD(nn.Module):
    def __init__(self,args):
        super(ResNext101_KD, self).__init__()
        self.args = args
        # after passing ReLU   : H/2  x W/2     (88 x 176)
        # after passing Layer1 : H/4  x W/4     (44 x 88)
        # after passing Layer2 : H/8  x W/8     (22 x 44)
        # after passing Layer3 : H/16 x W/16    (11 x 22)
        # after passing Layer4 : H/32 x W/32    (6 x 11)
        self.encoder = models.resnext101_32x8d(pretrained=True)
        del self.encoder.fc
        self.layerList = ['relu','layer1','layer2','layer3', 'layer4']
        dimList = [64, 256, 512, 1024,2048]
        out_dimList = [64, 128, 256, 512, 1024]
        norm = 'BN'
        act = 'ReLU'
        self.out_dimList = out_dimList
        self.fixList = ['layer1.0','layer1.1']
        convert1 = myConv(dimList[0], out_dimList[0], kSize=1, stride=1, padding=0, bias=False, norm=norm, act=act, num_groups=dimList[0]//16)
        convert2 = myConv(dimList[1], out_dimList[1], kSize=1, stride=1, padding=0, bias=False, norm=norm, act=act, num_groups=dimList[1]//16)
        convert3 = myConv(dimList[2], out_dimList[2], kSize=1, stride=1, padding=0, bias=False, norm=norm, act=act, num_groups=dimList[2]//16)
        convert4 = myConv(dimList[3], out_dimList[3], kSize=1, stride=1, padding=0, bias=False, norm=norm, act=act, num_groups=dimList[3]//16)
        convert5 = myConv(dimList[4], out_dimList[4], kSize=1, stride=1, padding=0, bias=False, norm=norm, act=act, num_groups=dimList[4]//16)
        self.conv_convert_list = nn.ModuleList([convert1,convert2,convert3,convert4,convert5])
    def forward(self, x):
        out_featList = []
        feature = x
        cnt = 0
        for k, v in self.encoder._modules.items():
            if k == 'avgpool':
                break
            feature = v(feature)
            if any(x in k for x in self.layerList):
                converted_feat = self.conv_convert_list[cnt](feature)
                out_featList.append(converted_feat)
                cnt = cnt + 1
        return out_featList
    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        '''
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable
        '''
        for name, parameters in self.encoder.named_parameters():
            if name == 'conv1.weight':
                parameters.requires_grad = False
            if any(x in name for x in self.fixList):
                parameters.requires_grad = False

class MobileNetV2_KD(nn.Module):
    def __init__(self,args):
        super(MobileNetV2_KD, self).__init__()
        self.args = args
        # after passing 1th : H/2  x W/2
        # after passing 2th : H/4  x W/4
        # after passing 3th : H/8  x W/8
        # after passing 4th : H/16 x W/16
        # after passing 5th : H/32 x W/32
        self.encoder = models.mobilenet_v2(pretrained=True)
        #self.encoder = models.mobilenet_v2(pretrained=False)
        #print(self.encoder)
        del self.encoder.classifier
        self.layerList = [1, 3, 6, 13, 18]
        dimList = [32, 144, 192, 576, 1280]
        out_dimList = [64, 128, 256, 512, 1024]
        self.out_dimList = out_dimList
        #self.fixList = args.fixlist
        norm = 'BN'
        act = 'ReLU'

        convert1 = myConv(dimList[0], out_dimList[0], kSize=1, stride=1, padding=0, bias=False, norm=norm, act=act, num_groups=dimList[0]//16)
        convert2 = myConv(dimList[1], out_dimList[1], kSize=1, stride=1, padding=0, bias=False, norm=norm, act=act, num_groups=dimList[1]//16)
        convert3 = myConv(dimList[2], out_dimList[2], kSize=1, stride=1, padding=0, bias=False, norm=norm, act=act, num_groups=dimList[2]//16)
        convert4 = myConv(dimList[3], out_dimList[3], kSize=1, stride=1, padding=0, bias=False, norm=norm, act=act, num_groups=dimList[3]//16)
        convert5 = myConv(dimList[4], out_dimList[4], kSize=1, stride=1, padding=0, bias=False, norm=norm, act=act, num_groups=dimList[4]//16)
        self.conv_convert_list = nn.ModuleList([convert1,convert2,convert3,convert4,convert5])
    def forward(self, x):
        out_featList = []
        feature = x
        cnt = 0
        for i in range(len(self.encoder.features)):
            if i in self.layerList:
                if i != 18:
                    for j in range(len(self.encoder.features[i].conv)):
                        feature = self.encoder.features[i].conv[j](feature)
                        if i == 1: 
                            if j == 0:
                                #print("i: ", i, "cnt: ",cnt, feature.shape)
                                converted_feat = self.conv_convert_list[cnt](feature)
                                out_featList.append(converted_feat)
                                cnt = cnt + 1
                        else:
                            if j == 1:
                                #print("i: ", i, "cnt: ",cnt, feature.shape)
                                converted_feat = self.conv_convert_list[cnt](feature)
                                out_featList.append(converted_feat)
                                cnt = cnt + 1
                else:
                    feature = self.encoder.features[i](feature)
                    converted_feat = self.conv_convert_list[cnt](feature)
                    out_featList.append(converted_feat)
            else:
                feature = self.encoder.features[i](feature)
            
        return out_featList

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

class ResNet18_KD(nn.Module):
    def __init__(self,args):
        super(ResNet18_KD, self).__init__()
        self.args = args
        # after passing ReLU   : H/2  x W/2
        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        self.encoder = models.resnet18(pretrained=True)
        del self.encoder.fc
        self.layerList = ['relu','layer1','layer2','layer3', 'layer4']
        dimList = [64, 64, 128, 256, 512]
        out_dimList = [64, 128, 256, 512, 1024]
        self.out_dimList = out_dimList
        norm = 'BN'
        act = 'ReLU'

        convert1 = myConv(dimList[0], out_dimList[0], kSize=1, stride=1, padding=0, bias=False, norm=norm, act=act, num_groups=dimList[0]//16)
        convert2 = myConv(dimList[1], out_dimList[1], kSize=1, stride=1, padding=0, bias=False, norm=norm, act=act, num_groups=dimList[1]//16)
        convert3 = myConv(dimList[2], out_dimList[2], kSize=1, stride=1, padding=0, bias=False, norm=norm, act=act, num_groups=dimList[2]//16)
        convert4 = myConv(dimList[3], out_dimList[3], kSize=1, stride=1, padding=0, bias=False, norm=norm, act=act, num_groups=dimList[3]//16)
        convert5 = myConv(dimList[4], out_dimList[4], kSize=1, stride=1, padding=0, bias=False, norm=norm, act=act, num_groups=dimList[4]//16)
        self.conv_convert_list = nn.ModuleList([convert1,convert2,convert3,convert4,convert5])
        
    def forward(self, x):
        out_featList = []
        feature = x
        cnt = 0
        for k, v in self.encoder._modules.items():
            if k == 'avgpool':
                break
            feature = v(feature)
            #feature = v(features[-1])
            #features.append(feature)
            if any(x in k for x in self.layerList):
                converted_feat = self.conv_convert_list[cnt](feature)
                out_featList.append(converted_feat)
                cnt = cnt + 1 
        return out_featList
    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

class EfficientNet_KD(nn.Module):
    def __init__(self,args, architecture="EfficientNet-B0"):
        super(EfficientNet_KD, self).__init__()
        self.args = args
        assert architecture in ["EfficientNet-B0", "EfficientNet-B1", "EfficientNet-B2", "EfficientNet-B3", 
                                    "EfficientNet-B4", "EfficientNet-B5", "EfficientNet-B6", "EfficientNet-B7"]
        
        if architecture == "EfficientNet-B0":
            self.encoder = geffnet.tf_efficientnet_b0_ns(pretrained=True)
            self.dimList = [16, 24, 40, 112, 1280] #5th feature is extracted after conv_head or bn2
            #self.dimList = [16, 24, 40, 112, 320] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B1":
            self.encoder = geffnet.tf_efficientnet_b1_ns(pretrained=True)
            self.dimList = [16, 24, 40, 112, 1280] #5th feature is extracted after conv_head or bn2
            #self.dimList = [16, 24, 40, 112, 320] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B2":
            self.encoder = geffnet.tf_efficientnet_b2_ns(pretrained=True)
            self.dimList = [16, 24, 48, 120, 1408] #5th feature is extracted after conv_head or bn2
            #self.dimList = [16, 24, 48, 120, 352] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B3":
            self.encoder = geffnet.tf_efficientnet_b3_ns(pretrained=True)
            self.dimList = [24, 32, 48, 136, 1536] #5th feature is extracted after conv_head or bn2
            #self.dimList = [24, 32, 48, 136, 384] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B4":
            self.encoder = geffnet.tf_efficientnet_b4_ns(pretrained=True)
            self.dimList = [24, 32, 56, 160, 1792] #5th feature is extracted after conv_head or bn2
            #self.dimList = [24, 32, 56, 160, 448] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B5":
            self.encoder = geffnet.tf_efficientnet_b5_ns(pretrained=True)
            self.dimList = [24, 40, 64, 176, 2048] #5th feature is extracted after conv_head or bn2
            #self.dimList = [24, 40, 64, 176, 512] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B6":
            self.encoder = geffnet.tf_efficientnet_b6_ns(pretrained=True)
            self.dimList = [32, 40, 72, 200, 2304] #5th feature is extracted after conv_head or bn2
            #self.dimList = [32, 40, 72, 200, 576] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B7":
            self.encoder = geffnet.tf_efficientnet_b7_ns(pretrained=True)
            self.dimList = [32, 48, 80, 224, 2560] #5th feature is extracted after conv_head or bn2
            #self.dimList = [32, 48, 80, 224, 640] #5th feature is extracted after blocks[6]

        del self.encoder.global_pool
        del self.encoder.classifier
        #self.block_idx = [3, 4, 5, 7, 9] #5th feature is extracted after blocks[6]
        #self.block_idx = [3, 4, 5, 7, 10] #5th feature is extracted after conv_head
        self.block_idx = [3, 4, 5, 7, 11] #5th feature is extracted after bn2
        # after passing blocks[3]    : H/2  x W/2
        # after passing blocks[4]    : H/4  x W/4
        # after passing blocks[5]    : H/8  x W/8
        # after passing blocks[7]    : H/16 x W/16
        # after passing conv_stem    : H/32 x W/32
        out_dimList = [64, 128, 256, 512, 1024]
        self.out_dimList = out_dimList
        norm = 'BN'
        act = 'ReLU'

        convert1 = myConv(self.dimList[0], out_dimList[0], kSize=1, stride=1, padding=0, bias=False, 
                           norm=norm, act=act, num_groups=self.dimList[0]//16)
        convert2 = myConv(self.dimList[1], out_dimList[1], kSize=1, stride=1, padding=0, bias=False, 
                           norm=norm, act=act, num_groups=self.dimList[1]//16)
        convert3 = myConv(self.dimList[2], out_dimList[2], kSize=1, stride=1, padding=0, bias=False, 
                           norm=norm, act=act, num_groups=self.dimList[2]//16)
        convert4 = myConv(self.dimList[3], out_dimList[3], kSize=1, stride=1, padding=0, bias=False, 
                           norm=norm, act=act, num_groups=self.dimList[3]//16)
        convert5 = myConv(self.dimList[4], out_dimList[4], kSize=1, stride=1, padding=0, bias=False, 
                           norm=norm, act=act, num_groups=self.dimList[4]//16)
        self.conv_convert_list = nn.ModuleList([convert1,convert2,convert3,convert4,convert5])
        
    def forward(self, x):
        out_featList = []
        feature = x
        cnt = 0
        block_cnt = 0
        for k, v in self.encoder._modules.items():
            if k == 'act2':
                break
            if k == 'blocks':
                for m, n in v._modules.items():
                    feature = n(feature)
                    if self.block_idx[block_cnt] == cnt:
                        converted_feat = self.conv_convert_list[block_cnt](feature)
                        out_featList.append(converted_feat)
                        block_cnt += 1
                    cnt += 1
            else:
                feature = v(feature)
                if self.block_idx[block_cnt] == cnt:
                    converted_feat = self.conv_convert_list[block_cnt](feature)
                    out_featList.append(converted_feat)
                    block_cnt += 1
                cnt += 1            
            
        return out_featList

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()
                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

class Basic_decoder_shallow(nn.Module):
    def __init__(self, args, dimList):
        super(Basic_decoder_shallow, self).__init__()
        norm = args.norm
        if norm == 'GN':
            print("==> Norm: GN")
        else:
            print("==> Norm: BN")
        
        if args.act == 'ELU':
            act = 'ELU'
        else:
            act = 'ReLU'
        act = 'ReLU'
        kSize = 3
        self.max_depth = args.max_depth
        self.dimList = dimList

        # decoder0 : 1 x H/32 x W/32 (Level 6)
        self.decoder_0_up = upConvLayer(dimList[4], dimList[4]//2, 2, norm, act, dimList[4]//16)

        # decoder1 : 1 x H/16 x W/16 (Level 5)
        self.decoder_1_reduc = myConv(dimList[4]//2 + dimList[3], dimList[3]//2, 1, stride=1, padding=0, bias=False, norm=norm, act=act, num_groups=(dimList[4]//2 + dimList[3])//16)
        self.decoder_1_up = upConvLayer(dimList[3]//2, dimList[3]//4, 2, norm, act, (dimList[3]//2)//16)
        
        # decoder2 : 1 x H/8 x W/8 (Level 4) - concat with output of 'Layer2'
        self.decoder_2_reduc = myConv(dimList[3]//4 + dimList[2], dimList[3]//4, 1, stride=1, padding=0, bias=False, norm=norm, act=act, num_groups=(dimList[3]//4 + dimList[2])//16)
        self.decoder_2_up = upConvLayer(dimList[3]//4, dimList[3]//8, 2, norm, act, (dimList[3]//4)//16)

        # decoder3 : 1 x H/4 x W/4 (Level 3) - concat with output of 'Layer1'
        self.decoder_3_reduc = myConv(dimList[3]//8 + dimList[1], dimList[3]//8, 1, stride=1, padding=0, bias=False, norm=norm, act=act, num_groups=(dimList[3]//8 + dimList[1])//16)
        self.decoder_3_up = upConvLayer(dimList[3]//8, dimList[3]//16, 2, norm, act, (dimList[3]//8)//16)

        # decoder4 : 1 x H/2 x W/2 (Level 2) - concat with output of 'bn1'
        self.decoder_4_reduc = myConv(dimList[3]//16 + dimList[0], dimList[3]//16, 1, stride=1, padding=0, bias=False, norm=norm, act=act, num_groups=(dimList[3]//16 + dimList[0])//16)
        self.decoder_4_up = upConvLayer(dimList[3]//16, dimList[3]//32, 2, norm, act, (dimList[3]//16)//16)

        self.decoder_5 = myConv(dimList[3]//32, 1, kSize, stride=1, padding=kSize//2, bias=False, norm=norm, act=act, num_groups=(dimList[3]//32)//16)

        self.drop0 = ReplaceBlock(drop_prob = 0., block_size = 2)
        self.drop1 = ReplaceBlock(drop_prob = 0., block_size = 2)
        self.drop2 = ReplaceBlock(drop_prob = 0., block_size = 7)
        self.drop3 = ReplaceBlock(drop_prob = 0., block_size = 7)

    def forward(self, x, keep_prob = None, T_dec_feat_list = None, dense_feat_T = None):
        new_S_feat_list = None
        drop_stud_list = None
        drop_tchr_list = None

        cat0, cat1, cat2, cat3, dense_feat = x[0], x[1], x[2], x[3], x[4]
        if T_dec_feat_list is not None:
            dec1_T, dec2_T, dec3_T, dec4_T, dec5_T = T_dec_feat_list
            drop_stud_dec_list = []
            drop_tchr_dec_list = []

        if T_dec_feat_list is not None:
            dense_feat, drop_S_feat_dec, drop_T_feat_dec = self.drop0(dense_feat, dense_feat_T)
            drop_stud_dec_list.append(drop_S_feat_dec)
            drop_tchr_dec_list.append(drop_T_feat_dec)

        out = self.decoder_0_up(dense_feat, cat3.shape[2:])
        dec0 = out
        if T_dec_feat_list is not None:
            out, drop_S_feat_dec, drop_T_feat_dec = self.drop1(out, dec1_T)
            drop_stud_dec_list.append(drop_S_feat_dec)
            drop_tchr_dec_list.append(drop_T_feat_dec)

        out = self.decoder_1_reduc(torch.cat([out, cat3],dim=1))
        out = self.decoder_1_up(out)
        dec1 = out
        if T_dec_feat_list is not None:
            out, drop_S_feat_dec, drop_T_feat_dec = self.drop2(out, dec2_T)
            drop_stud_dec_list.append(drop_S_feat_dec)
            drop_tchr_dec_list.append(drop_T_feat_dec)

        out = self.decoder_2_reduc(torch.cat([out, cat2],dim=1))
        out = self.decoder_2_up(out)
        dec2 = out
        if T_dec_feat_list is not None:
            out, drop_S_feat_dec, drop_T_feat_dec = self.drop3(out, dec3_T)
            #out, drop_S_feat_dec, drop_T_feat_dec = out, torch.tensor(0.).cuda(), torch.tensor(0.).cuda()
            drop_stud_dec_list.append(drop_S_feat_dec)
            drop_tchr_dec_list.append(drop_T_feat_dec)

        out = self.decoder_3_reduc(torch.cat([out, cat1],dim=1))
        out = self.decoder_3_up(out)
        dec3 = out
        if T_dec_feat_list is not None:
            #out, drop_S_feat_dec, drop_T_feat_dec = self.drop(out, dec4_T)
            out, drop_S_feat_dec, drop_T_feat_dec = out, torch.tensor(0.).cuda(), torch.tensor(0.).cuda()
            drop_stud_dec_list.append(drop_S_feat_dec)
            drop_tchr_dec_list.append(drop_T_feat_dec)

        out = self.decoder_4_reduc(torch.cat([out, cat0],dim=1))
        out = self.decoder_4_up(out)
        dec4 = out
        if T_dec_feat_list is not None:
            #out, drop_S_feat_dec, drop_T_feat_dec = self.drop(out, dec5_T)
            out, drop_S_feat_dec, drop_T_feat_dec = out, torch.tensor(0.).cuda(), torch.tensor(0.).cuda()
            drop_stud_dec_list.append(drop_S_feat_dec)
            drop_tchr_dec_list.append(drop_T_feat_dec)

        out = self.decoder_5(out)
        final_depth = torch.sigmoid(out)
        if T_dec_feat_list is not None:
            return [dec0, dec1, dec2, dec3, dec4], drop_stud_dec_list, drop_tchr_dec_list, final_depth*self.max_depth
        else:
            return [dec0, dec1, dec2, dec3, dec4], final_depth*self.max_depth

    def set_drop_prob(self, drop_prob):
        self.drop0.drop_prob = drop_prob
        self.drop1.drop_prob = drop_prob
        self.drop2.drop_prob = drop_prob
        self.drop3.drop_prob = drop_prob

# Knowledge Distillation Depth Network
class KDDN_encoder(nn.Module):
    def __init__(self, args, model = 'ResNext', dimList = None):
        super(KDDN_encoder, self).__init__()

        if model == 'ResNet101':
            self.encoder = ResNet101_KD(args)
        elif model == 'ResNet50':
            self.encoder = ResNet50_KD(args)
        elif model == 'ResNext101':
            self.encoder = ResNext101_KD(args)
        elif model == 'MobileNetV2':
            self.encoder = MobileNetV2_KD(args)
        elif model == 'ResNet18':
            self.encoder = ResNet18_KD(args)
        elif 'EfficientNet' in model:
            self.encoder = EfficientNet_KD(args, model)
    def forward(self, x):
        out_featList = self.encoder(x)
        return out_featList    

class ReplaceBlock(nn.Module):
    r"""Randomly replace 2D spatial blocks of the Stucent feature 
    with corresponding blokcs of the Teature feature.
    the code part of randomly extracting block area is based on
    "DropBlock: A regularization method for convolutional networks":
       https://arxiv.org/abs/1810.12890
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. 
    """

    def __init__(self, drop_prob, block_size):
        super(ReplaceBlock, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, S_feat, T_feat=None):
        # shape: (bsize, channels, height, width)

        assert S_feat.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if T_feat is None:
            drop_S_feat = torch.tensor(0.).cuda()
            drop_T_feat = torch.tensor(0.).cuda()
            return S_feat, drop_S_feat, drop_T_feat
        else:
            # get gamma value
            gamma = self._compute_gamma(S_feat)

            # sample mask
            mask = (torch.rand(S_feat.shape[0], *S_feat.shape[2:]) < gamma).float()

            # compute block mask
            block_mask = self._compute_block_mask(mask)
            block_mask = block_mask.expand_as(S_feat).bool()
            block_mask = block_mask.cuda()

            if S_feat[block_mask].shape[0] != 0:
                drop_S_feat = S_feat[block_mask]
                drop_T_feat = T_feat[block_mask]
                valid = (S_feat > 0)
                valid = block_mask & valid
                S_feat[valid] = S_feat[valid]*(T_feat[valid]/S_feat[valid])
            else:
                drop_S_feat = torch.tensor(0.).cuda()
                drop_T_feat = torch.tensor(0.).cuda()

            return S_feat, drop_S_feat, drop_T_feat

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        return block_mask

    def _compute_gamma(self, x):
        h, w = x.shape[2], x.shape[3]
        return (self.drop_prob / (self.block_size ** 2))

