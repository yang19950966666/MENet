import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from net.pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2_li, pvt_v2_b3
from kan_conv.KANConv import KAN_Convolution
#CA注意力
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
class KConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(KConvBNR, self).__init__()

        self.block = nn.Sequential(
            KAN_Convolution(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), device="cuda"),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
class CAEM(nn.Module):
    def __init__(self, channel1, channel2):
        super(CAEM, self).__init__()
        self.ca1 = CoordAtt(channel1, channel1)
        self.ca2 = CoordAtt(channel2, channel2)
        self.reduce1 = Conv1x1(channel1, 256)
        self.reduce4 = Conv1x1(channel2, 256)

        self.block = nn.Sequential(
            ConvBNR(256 + 256, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))

    def forward(self, x4, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(self.ca1(x1))
        x4 = self.reduce4(self.ca2(x4))
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)

        return out
class BGM(nn.Module):
    def __init__(self, channel):
        super(BGM, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        x = c * att + c
        x = self.conv2d(x)
        wei = self.avg_pool(x)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = x * wei

        return x
class FFM(nn.Module):
    def __init__(self, hchannel, channel):
        super(FFM, self).__init__()
        self.conv1_1 = Conv1x1(hchannel + channel, channel)
        self.conv3_1 = ConvBNR(channel // 4, channel // 4, 3)
        self.dconv5_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=2)
        self.dconv7_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=3)
        self.dconv9_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=4)
        self.conv1_2 = Conv1x1(channel, channel)
        self.conv3_3 = ConvBNR(channel, channel, 3)

    def forward(self, lf, hf):
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((lf, hf), dim=1)
        x = self.conv1_1(x)
        xc = torch.chunk(x, 4, dim=1)
        x0 = self.conv3_1(xc[0] + xc[1])
        x1 = self.dconv5_1(xc[1] + x0 + xc[2])
        x2 = self.dconv7_1(xc[2] + x1 + xc[3])
        x3 = self.dconv9_1(xc[3] + x2)
        xx = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.conv3_3(x + xx)

        return x
class KFFM(nn.Module):
    def __init__(self, hchannel, channel):
        super(KFFM, self).__init__()
        self.conv1_1 = Conv1x1(hchannel + channel, channel)
        self.conv3_1 = ConvBNR(channel // 4, channel // 4, 3)
        self.dconv5_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=2)
        self.dconv7_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=3)
        self.dconv9_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=4)
        self.conv1_2 = Conv1x1(channel, channel)
        self.conv3_3 = KConvBNR(channel, channel, 3)

    def forward(self, lf):
        x = torch.cat((lf, lf), dim=1)
        x = self.conv1_1(x)
        xc = torch.chunk(x, 4, dim=1)
        x0 = self.conv3_1(xc[0] + xc[1])
        x1 = self.dconv5_1(xc[1] + x0 + xc[2])
        x2 = self.dconv7_1(xc[2] + x1 + xc[3])
        x3 = self.dconv9_1(xc[3] + x2)
        xx = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.conv3_3(x + xx)
        return x
class MENet(nn.Module):
    def __init__(self, fun_str = 'pvt_v2_b3'):
        super(MENet, self).__init__()
        #self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.backbone, embedding_dims = eval(fun_str)()

        self.caem = CAEM(embedding_dims[0], embedding_dims[3])

        self.bgm1 = BGM(embedding_dims[0])
        self.bgm2 = BGM(embedding_dims[1])
        self.bgm3 = BGM(embedding_dims[2])
        self.bgm4 = BGM(embedding_dims[3])

        self.reduce1 = Conv1x1(embedding_dims[0], embedding_dims[0]//4)
        self.reduce2 = Conv1x1(embedding_dims[1], embedding_dims[1]//4)
        self.reduce3 = Conv1x1(embedding_dims[2], embedding_dims[3]//8)
        self.reduce4 = Conv1x1(embedding_dims[3], embedding_dims[3]//8)

        self.ffm1 = FFM(embedding_dims[1]//4, embedding_dims[0]//4)
        self.ffm2 = FFM(embedding_dims[3]//8, embedding_dims[1]//4)
        self.ffm3 = FFM(embedding_dims[3]//8, embedding_dims[3]//8)
        self.kffm = KFFM(embedding_dims[3]//8, embedding_dims[3]//8)
        self.ffm5 = FFM(embedding_dims[0] // 4, embedding_dims[3] // 8)
        self.ffm6 = FFM(embedding_dims[1] // 4, embedding_dims[3] // 8)
        self.ffm7 = FFM(embedding_dims[3] // 8, embedding_dims[0] // 4)

        self.predictor1 = nn.Conv2d(embedding_dims[0]//4, 1, 1)
        self.predictor2 = nn.Conv2d(embedding_dims[3]//8, 1, 1)
        self.predictor3 = nn.Conv2d(embedding_dims[3]//8, 1, 1)

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)

        edge4 = self.caem(x4, x1)
        edge_att4 = torch.sigmoid(edge4)

        x1a = self.bgm1(x1, edge_att4)
        x2a = self.bgm2(x2, edge_att4)
        x3a = self.bgm3(x3, edge_att4)
        x4a = self.bgm4(x4, edge_att4)

        x1r = self.reduce1(x1a)
        x2r = self.reduce2(x2a)
        x3r = self.reduce3(x3a)
        x4r = self.reduce4(x4a)

        x4y = self.kffm(x4r)  #torch.Size([5, 64, 13, 13])
        #print(x4y.size())
        x34 = self.ffm3(x3r, x4y)  #torch.Size([5, 64, 26, 26])
        #print(x34.size())
        x234 = self.ffm2(x2r, x34)  #torch.Size([5, 32, 52, 52])
        #print(x234.size())
        x1234 = self.ffm1(x1r, x234)  #torch.Size([5, 16, 104, 104])
        #print(x1234.size())

        x3 = self.ffm5(x1234, x34)
        #print(x3.size())
        x2 = self.ffm6(x3, x234)
        #print(x2.size())
        x1 = self.ffm7(x2, x1234)
        #print(x1.size())

        o3 = self.predictor3(x3)
        #print(o3.size())
        o3 = F.interpolate(o3, scale_factor=4, mode='bilinear', align_corners=False)
        #print(o3.size())
        o2 = self.predictor2(x2)
        #print(o2.size())
        o2 = F.interpolate(o2, scale_factor=4, mode='bilinear', align_corners=False)
        #print(o2.size())
        o1 = self.predictor1(x1)
        #print(o1.size())
        o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=False)
        #print(o1.size())

        oe = F.interpolate(edge_att4, scale_factor=4, mode='bilinear', align_corners=False)
        #print(oe.size())


        return o3, o2, o1, oe
