import pywt
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from theano import tensor
from theano.tensor import basic
from dwtmodel.waveletpro import *
# from pylearn2.utils.rng import make_theano_rng
# from nspt import nsbp, nssum
from six.moves import xrange

pool_size = None
stride_size = None

def B_LPPool2d(Input, P, kernel):
    # b, c, w, h = Input.shape
    # R = 1 / (w * h)
    lPPool2d = nn.LPPool2d(norm_type=P, kernel_size=kernel, stride=2, ceil_mode=False)
    R = kernel * kernel
    output = lPPool2d(Input) * (1 / R) * ((1 - R) / P + R)
    return output


# def Battle(f1, f2, f3):
#     battle = torch.gt(f1, f2)
#     battle = battle.cpu().numpy()
#     if np.sum(battle == True) >= np.sum(battle == False):
#         battle1 = torch.gt(f2, f3)
#         battle1 = battle1.cpu().numpy()
#         if np.sum(battle1 == True) >= np.sum(battle1 == False):
#             flag = 1
#         else:
#             flag = 2
#     else:
#         battle1 = torch.gt(f1, f3)
#         battle1 = battle1.cpu().numpy()
#         if np.sum(battle1 == True) >= np.sum(battle1 == False):
#             flag = 3
#         else:
#             flag = 4
#     return flag
def Battle(f1, f2, f3):

    battle = torch.gt(f1, f2)
    num_true = torch.sum(battle).item()
    num_false = torch.sum(~battle).item()

    if num_true >= num_false:
        battle1 = torch.gt(f2, f3)
        num_true1 = torch.sum(battle1).item()
        num_false1 = torch.sum(~battle1).item()

        if num_true1 >= num_false1:
            flag = 1
        else:
            flag = 2
    else:
        battle1 = torch.gt(f1, f3)
        num_true1 = torch.sum(battle1).item()
        num_false1 = torch.sum(~battle1).item()

        if num_true1 >= num_false1:
            flag = 3
        else:
            flag = 4

    return flag

# def BI(Input):
#     # Input = Input / 1.0
#     # Input = torch.FloatTensor(Input)
#     Output = torch.nn.functional.interpolate(Input, scale_factor=2, mode="bilinear", align_corners=False)
#     upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#     # upMS = np.array(upMS)
#     # print('ms4图上采样的形状：', np.shape(upMS))
#     return Output


'''归一化图片'''


def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image


def last_pool(im_shp, p_shp, p_strd):
    rval = int(np.ceil(float(im_shp - p_shp) / p_strd))
    return rval

def probabilistic_pooling(Input, pool_size=(2, 2)):
    # input: 输入的张量，形状为(batch_size, channels, height, width)
    # pool_size: 池化窗口的大小，形状为(pool_height, pool_width)
    epsilon = 1e-8
    batch_size, channels, height, width = Input.shape
    pool_height, pool_width = pool_size

    device = Input.device  # 获取当前输入张量所在设备

    output = torch.zeros((batch_size, channels, height // pool_height, width // pool_width),
                         dtype=Input.dtype, device=device)

    # 将输入张量先按池化窗口大小展开成一个四维张量
    unfolded = Input.unfold(2, pool_height, pool_height).unfold(3, pool_width, pool_width)
    # 展开的形状为(batch_size, channels, h_n, w_n, pool_height, pool_width)
    # 其中h_n和w_n是分别是height和width按池化窗口大小展开后的长度

    # 将展开后的张量展平到二维张量
    flattened = unfolded.contiguous().view(batch_size, channels, -1, pool_height * pool_width)
    # 形状为(batch_size, channels, h_n*w_n, pool_height*pool_width)

    flattened = torch.where(torch.logical_or(torch.isinf(flattened), torch.isnan(flattened)) | torch.lt(flattened, 0),
                            torch.tensor(epsilon, device=device), flattened)

    # 计算每个池化区域中的概率分布并进行采样
    prob = torch.softmax(flattened, dim=-1)

    idx = torch.multinomial(prob.view(-1, pool_height * pool_width), num_samples=1)
    # idx = prob.view(-1, pool_height * pool_width).multinomial(num_samples=1)
    idx = idx.view(batch_size, channels, -1, 1)  # 扩展最后一个纬度

    # 根据采样结果更新输出张量
    output = flattened.gather(-1, idx).view(batch_size, channels, height // pool_height, width // pool_width)

    return output

class ca_layer(nn.Module):
    def __init__(self, in_channel=12, out_channel=24):
        super(ca_layer, self).__init__()
        self.in_channels = in_channel
        self.out_channels = out_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.lPPool2d = nn.LPPool2d(norm_type=6, kernel_size=3, stride=2, ceil_mode=False)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        # n_out=(n_in+2p-k)/s+1
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels // 2, kernel_size=1, stride=2, padding=4, bias=False)
        self.conv1_1 = nn.Conv2d(self.in_channels, self.out_channels // 2, kernel_size=1, stride=2, padding=4,
                                 bias=False)
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_1 = nn.Conv2d(self.in_channels, self.out_channels // 2, kernel_size=3, stride=1, padding=1,
                                 bias=False)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels // 2, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3_1 = nn.Conv2d(self.in_channels, self.out_channels // 2, kernel_size=5, stride=1, padding=2,
                                 bias=False)
        self.conv4 = nn.Conv2d(out_channel, out_channel, 3, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        A1 = self.conv1(x)
        F1 = self.conv1_1(x)
        f1_0 = B_LPPool2d(F1, 4, 2)
        f1_0 = self.relu(f1_0)
        # f1_0 = self.lPPool2d(F1)
        f1_1 = self.avg_pool(f1_0)
        f1_2 = self.conv(f1_1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # f1_3 = self.sigmoid(f1_2)
        f1_3 = self.softmax(f1_2)

        A2 = self.conv2(x)
        F2 = self.conv2_1(x)
        f2_0 = B_LPPool2d(F2, 6, 2)
        f2_0 = self.relu(f2_0)
        # f2_0 = self.lPPool2d(F2)
        f2_1 = self.avg_pool(f2_0)
        f2_2 = self.conv(f2_1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # f2_3 = self.sigmoid(f2_2)
        f2_3 = self.softmax(f2_2)

        A3 = self.conv3(x)
        F3 = self.conv3_1(x)
        f3_0 = B_LPPool2d(F3, 4, 2)
        f3_0 = self.relu(f3_0)
        # f3_0 = self.lPPool2d(F3)
        f3_1 = self.avg_pool(f3_0)
        f3_2 = self.conv(f3_1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # f3_3 = self.sigmoid(f3_2)
        f3_3 = self.softmax(f3_2)
        # f3_4 = A3 * f3_3.expand_as(A3)

        Flag = Battle(f1_3, f2_3, f3_3)
        if Flag == 1 or Flag == 3:
            f1_4 = A1 * f1_3.expand_as(A1)
            f2_4 = A2 * f2_3.expand_as(A2)
            # f1_4 = self.relu(f1_4)
            # f2_4 = self.relu(f2_4)
            f = torch.cat([f1_4, f2_4], dim=1)
        elif Flag == 2:
            f1_4 = A1 * f1_3.expand_as(A1)
            f3_4 = A3 * f3_3.expand_as(A3)
            # f1_4 = self.relu(f1_4)
            # f3_4 = self.relu(f3_4)
            f = torch.cat([f1_4, f3_4], dim=1)
        else:
            f2_4 = A2 * f2_3.expand_as(A2)
            f3_4 = A3 * f3_3.expand_as(A3)
            # f2_4 = self.relu(f2_4)
            # f3_4 = self.relu(f3_4)
            f = torch.cat([f2_4, f3_4], dim=1)

        # f = self.down_conv(f)
        # f = x + f
        out = self.relu(f)
        return out


class sa_layer(nn.Module):
    def __init__(self, k_size=3):
        super(sa_layer, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        # self.randomPool2d = RandomPool2d(2)
        self.avg = nn.AdaptiveAvgPool2d((4, 4))
        self.max = nn.AdaptiveMaxPool2d((4, 4))
        self.BI = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1_1 = nn.Conv2d(1, 1, kernel_size=k_size, stride=1, padding=1, bias=False)
        self.conv2_1 = nn.Conv2d(1, 1, kernel_size=k_size, stride=1, padding=1, bias=False)
        self.conv1_2 = nn.Conv2d(2, 1, kernel_size=k_size, stride=1, padding=1, bias=False)
        self.conv2_2 = nn.Conv2d(2, 1, kernel_size=k_size, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(2, 1, kernel_size=k_size, stride=1, padding=1, bias=False)
        # self.sto_pool = probabilistic_pooling(Input, pool_size=(2, 2))
        # self.sto_pool = probabilistic_pooling(pool_size=(2, 2))
        self.conv5 = nn.Conv2d(2, 16 // 4, 3, 1, 1)
        self.conv6 = nn.Conv2d(16 // 4, 1, 3, 1, 1)
        self.BN = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        # 随机、最大池化支路
        # w1 = 0.2
        max_out1, _ = torch.max(x, dim=1, keepdim=True)  # torch.Size([2, 1, 8, 8])
        max_out2 = probabilistic_pooling(max_out1, (2, 2))  # torch.Size([2, 1, 4, 4])
        # max_out2 = self.max(max_out1)
        # max_out3 = torch.max_pool2d(max_out1, 2)
        max_out3 = self.max(max_out1)  # torch.Size([2, 1, 4, 4])
        avg_out4 = self.avg(max_out1)  # torch.Size([2, 1, 4, 4])

        out2 = 0.1 * max_out2 + 0.6 * max_out3 + 0.3 * avg_out4
        out2 = self.conv1_1(out2)
        out2 = self.BI(out2)
        # max_out3 = torch.max_pool2d(max_out2, 2)
        # max_out3 = self.conv1_1(max_out3)
        # max_out3 = BI(max_out3)
        out1 = torch.cat([max_out1, out2], dim=1)
        out = self.sigmoid(self.conv6(F.relu(self.conv5(out1))))
        # out = self.conv1_2(out1)

        # out = self.sigmoid(out)
        out = torch.mul(x, out.expand_as(x))
        # out = x * out.expand_as(x)
        # out = self.relu(out)
        # out = x + out
        out = self.relu(out)

        return out
        # return x * out.expand_as(x)


class Waveletatt(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution=224, in_planes=3, norm_layer=nn.LayerNorm):
        super().__init__()
        wavename = 'haar'
        self.input_resolution = input_resolution

        # self.dim = dim
        # self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # self.norm = norm_layer(4 * dim)
        # self.low_dim = nn.Conv2d(4 * in_planes, in_planes,kernel_size=3, stride=1,padding=1)
        self.downsamplewavelet = nn.Sequential(*[nn.Upsample(scale_factor=2), Downsamplewave1(wavename=wavename)])
        # self.downsamplewavelet = Downsamplewave(wavename=wavename)
        # self.conv1 = nn.Conv2d()
        # self.ac = nn.Sigmoid()
        # self.bn = nn.BatchNorm2d(in_planes)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // 2, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        xori = x
        B, C, H, W = x.shape
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 2, 1)
        # x0,x1,x2,x3 = Downsamplewave(x)
        ##x0,x1,x2,x3= self.downsamplewavelet(x)
        y = self.downsamplewavelet(x)
        y = self.fc(y).view(B, C, 1, 1)  # torch.Size([64, 256])-->torch.Size([64, 256, 1, 1])
        y = xori * y.expand_as(xori)
        return y


class use_l(nn.Module):

    def __init__(self, input_resolution=224, in_planes=4):
        super().__init__()
        wavename = 'haar'
        self.input_resolution = input_resolution
        self.downsamplewavelet = nn.Sequential(*[nn.Upsample(scale_factor=2), Downsamplewave2(wavename=wavename)])
        # self.downsamplewavelet = Downsamplewave(wavename=wavename)
        # self.conv1 = nn.Conv2d()
        self.sigmoid = nn.Sigmoid()
        # self.bn = nn.BatchNorm2d(in_planes)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // 2, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(in_planes // 2, in_planes, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        # self.change = nn.Sequential(Conv3x3BNReLU(in_channels=in_planes, out_channels=out_planes, stride=1, groups=1), )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        xori = x
        B, C, H, W = x.shape
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 2, 1)
        y = self.downsamplewavelet(x)
        y = self.fc(y).view(B, C, 1, 1)  # torch.Size([64, 256])-->torch.Size([64, 256, 1, 1])
        y = xori * y.expand_as(xori)
        return y


class use_h(nn.Module):

    def __init__(self, input_resolution=224, in_planes=1):
        super().__init__()
        wavename = 'haar'
        self.input_resolution = input_resolution
        self.downsamplewavelet = nn.Sequential(Downsamplewave3(wavename=wavename))
        # self.downsamplewavelet = Downsamplewave(wavename=wavename)
        # self.conv1 = nn.Conv2d()
        # self.ac = nn.Sigmoid()
        # self.bn = nn.BatchNorm2d(in_planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: B, H*W, C
        """
        xori = x
        B, C, H, W = x.shape
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 2, 1)
        y1, y2 = self.downsamplewavelet(x)
        y1 = self.sigmoid(torch.tensor(y1, device='cuda'))
        y2 = self.sigmoid(torch.tensor(y2, device='cuda'))
        return y1, y2

class use_hy(nn.Module):

    def __init__(self, input_resolution=224, in_planes=4):
        super().__init__()
        wavename = 'haar'
        self.input_resolution = input_resolution
        self.downsamplewavelet = nn.Sequential(*[nn.Upsample(scale_factor=2), Downsamplewave4(wavename=wavename)])
        # self.downsamplewavelet = Downsamplewave(wavename=wavename)
        # self.conv1 = nn.Conv2d()
        self.sigmoid = nn.Sigmoid()
        # self.bn = nn.BatchNorm2d(in_planes)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // 2, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(in_planes // 2, in_planes, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        # self.change = nn.Sequential(Conv3x3BNReLU(in_channels=in_planes, out_channels=out_planes, stride=1, groups=1), )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        xori = x
        B, C, H, W = x.shape
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 2, 1)
        y = self.downsamplewavelet(x)
        y = self.fc(y).view(B, C, 1, 1)  # torch.Size([64, 256])-->torch.Size([64, 256, 1, 1])
        y = xori * y.expand_as(xori)
        return y

class SequeezeExcite(nn.Module):
    def __init__(self,
                 input_c,  # 输入到MBConv模块的特征图的通道数
                 expand_c,  # 输入到SE模块的特征图的通道数
                 se_ratio=0.25,  # 第一个全连接下降的通道数的倍率
                 ):
        super(SequeezeExcite, self).__init__()

        # 第一个全连接下降的通道数
        sequeeze_c = int(input_c * se_ratio)
        # 1*1卷积代替全连接下降通道数
        self.conv_reduce = nn.Conv2d(expand_c, sequeeze_c, kernel_size=1, stride=1)
        self.act1 = nn.SiLU()
        # 1*1卷积上升通道数
        self.conv_expand = nn.Conv2d(sequeeze_c, expand_c, kernel_size=1, stride=1)
        self.act2 = nn.Sigmoid()

    # 前向传播
    def forward(self, x):
        # 全局平均池化[b,c,h,w]==>[b,c,1,1]
        scale = x.mean((2, 3), keepdim=True)
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        # 对输入特征图x的每个通道加权
        return scale * x


# -------------------------------------- #
# 卷积 + BN + 激活
# -------------------------------------- #
class ConvBnAct(nn.Module):
    def __init__(self,
                 in_planes,  # 输入特征图通道数
                 out_planes,  # 输出特征图通道数
                 kernel_size=3,  # 卷积核大小
                 stride=1,  # 滑动窗口步长
                 groups=1,  # 卷积时通道数分组的个数
                 norm_layer=None,  # 标准化方法
                 activation_layer=None,  # 激活函数
                 ):
        super(ConvBnAct, self).__init__()

        # 计算不同卷积核需要的0填充个数
        padding = (kernel_size - 1) // 2
        # 若不指定标准化和激活函数，就用默认的
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            # activation_layer = nn.SiLU
            activation_layer = nn.ReLU

        # 卷积
        self.conv = nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False,
                              )
        # BN
        self.bn = norm_layer(out_planes)
        # silu
        self.act = activation_layer(inplace=True)

    # 前向传播
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# -------------------------------------- #
# Stochastic Depth dropout 方法，随机丢弃输出层
# -------------------------------------- #
def drop_path(x, drop_prob: float = 0., training: bool = False):  # drop_prob代表丢弃概率
    # （1）测试时不做 drop path 方法
    if drop_prob == 0. or training is False:
        return x
    # （2）训练时使用
    keep_prob = 1 - drop_prob  # 网络每个特征层的输出结果的保留概率
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        out = drop_path(x, self.drop_prob, self.training)
        return out


# -------------------------------------- #
# MBConv卷积块，FusedMBConv卷积块，ResBlk残差模块
# -------------------------------------- #
class MBConv(nn.Module):
    def __init__(self,
                 input_c,
                 output_c,
                 kernel_size,  # DW卷积的卷积核size
                 expand_ratio,  # 第一个1*1卷积上升通道数的倍率
                 stride,  # DW卷积的步长
                 se_ratio,  # SE模块的第一个全连接层下降通道数的倍率
                 drop_rate,  # 随机丢弃输出层的概率
                 norm_layer,
                 ):
        super(MBConv, self).__init__()

        # 下采样模块不存在残差边；基本模块stride=1时且输入==输出特征图通道数，才用到残差边
        # self.has_shortcut = (stride == 1 and input_c == output_c)
        self.has_shortcut = (kernel_size == 3)
        # 激活函数
        activation_layer = nn.SiLU
        # 第一个1*1卷积上升的输出通道数
        expanded_c = input_c * expand_ratio
        self.extra = nn.Sequential()

        if output_c != input_c or stride != 1:
            self.extra = nn.Sequential(
                nn.Conv2d(input_c, output_c, kernel_size=1, stride=stride),
                nn.BatchNorm2d(output_c)
            )
        # 1*1升维卷积
        self.expand_conv = ConvBnAct(in_planes=input_c,  # 输入通道数
                                     out_planes=expanded_c,  # 上升的通道数
                                     kernel_size=1,
                                     stride=1,
                                     groups=1,
                                     norm_layer=norm_layer,
                                     activation_layer=activation_layer,
                                     )
        # DW卷积
        self.dwconv = ConvBnAct(in_planes=expanded_c,
                                out_planes=expanded_c,  # DW卷积输入和输出通道数相同
                                kernel_size=kernel_size,
                                stride=stride,
                                groups=expanded_c,  # 对特征图的每个通道做卷积
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                )
        # SE注意力
        # 如果se_ratio>0就使用SE模块，否则线性连接、
        if se_ratio > 0:
            self.se = SequeezeExcite(input_c=input_c,  # MBConv模块的输入通道数
                                     expand_c=expanded_c,  # SE模块的输出通道数
                                     se_ratio=se_ratio,  # 第一个全连接的通道数下降倍率
                                     )
        else:
            self.se = nn.Identity()

        # 1*1逐点卷积降维
        self.project_conv = ConvBnAct(in_planes=expanded_c,
                                      out_planes=output_c,
                                      kernel_size=1,
                                      stride=1,
                                      groups=1,
                                      norm_layer=norm_layer,
                                      activation_layer=nn.Identity,  # 不使用激活函数，恒等映射
                                      )
        # droppath方法
        self.drop_rate = drop_rate
        # 只在基本模块使用droppath丢弃输出层
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_prob=drop_rate)

    # 前向传播
    def forward(self, x):
        out = self.expand_conv(x)  # 升维
        out = self.dwconv(out)  # DW
        out = self.se(out)  # 通道注意力
        out = self.project_conv(out)  # 降维

        # 残差边
        if self.has_shortcut:
            if self.drop_rate > 0:
                out = self.dropout(out)  # drop_path方法
            out += self.extra(x)  # 输入和输出短接
        out = F.relu(out)
        return out


class FusedMBConv(nn.Module):
    def __init__(self,
                 input_c,
                 output_c,
                 kernel_size,  # DW卷积核的size
                 expand_ratio,  # 第一个1*1卷积上升的通道数
                 stride,  # DW卷积的步长
                 se_ratio,  # SE模块第一个全连接下降通道数的倍率
                 drop_rate,  # drop—path丢弃输出层的概率
                 norm_layer,
                 ):
        super(FusedMBConv, self).__init__()

        # 残差边只用于基本模块
        # self.has_shortcut = (stride == 1 and input_c == output_c)
        self.has_shortcut = (kernel_size == 3 and stride == 1)
        # 随机丢弃输出层的概率
        self.drop_rate = drop_rate
        # 第一个卷积是否上升通道数
        self.has_expansion = (expand_ratio != 1)
        # 激活函数
        activation_layer = nn.SiLU
        self.extra = nn.Sequential()

        if output_c != input_c:
            self.extra = nn.Sequential(
                nn.Conv2d(input_c, output_c, kernel_size=1, stride=stride),
                nn.BatchNorm2d(output_c)
            )
        # 卷积上升的通道数
        expanded_c = input_c * expand_ratio

        # 只有expand_ratio>1时才使用升维卷积
        if self.has_expansion:
            self.expand_conv = ConvBnAct(in_planes=input_c,
                                         out_planes=expanded_c,  # 升维的输出通道数
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer,
                                         )
            # 1*1降维卷积
            self.project_conv = ConvBnAct(in_planes=expanded_c,
                                          out_planes=output_c,
                                          kernel_size=1,
                                          stride=1,
                                          norm_layer=norm_layer,
                                          activation_layer=nn.Identity,
                                          )
        # 如果expand_ratio=1，即第一个卷积不上升通道
        else:
            self.project_conv = ConvBnAct(in_planes=input_c,
                                          out_planes=output_c,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          norm_layer=norm_layer,
                                          activation_layer=activation_layer,
                                          )

        # 只有在基本模块中才使用shortcut，只有存在shortcut时才能用drop_path
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    # 前向传播
    def forward(self, x):
        # 第一个卷积块上升通道数倍率>1
        if self.has_expansion:
            out = self.expand_conv(x)
            out = self.project_conv(out)
        # 不上升通道数
        else:
            out = self.project_conv(x)

        # 基本模块中使用残差边
        if self.has_shortcut:
            if self.drop_rate > 0:
                out = self.dropout(out)
            x = self.extra(x)
            out += x
        out = F.relu(out)
        return out


class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()

        # in_channels：输入张量的channels数，out_channels：输出通道数，kernel_size：卷积核大小，stride：步长大小，Padding：即所谓的图像填充
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        # 防止数据在进行Relu之前因为数据过大而导致网络性能的不稳定
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()

        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.extra(x) + out
        out = F.relu(out)

        return out


# shufflenetv2
# 3x3DW卷积(含激活函数)
def Conv3x3BNReLU(in_channels, out_channels, stride, groups):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                  groups=groups),
        nn.BatchNorm2d(out_channels),
        # nn.ReLU6(inplace=True)
        nn.ReLU(inplace=False)
    )


# 3x3DW卷积(不激活函数)
def Conv3x3BN(in_channels, out_channels, stride, groups):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                  groups=groups),
        nn.BatchNorm2d(out_channels)
    )


# 1x1PW卷积(含激活函数)
def Conv1x1BNReLU(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


# 1x1PW卷积(不含激活函数)
def Conv1x1BN(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels)
    )


# 划分channels: dim默认为0，但是由于channnels位置在1，所以传参为1
class HalfSplit(nn.Module):
    def __init__(self, dim=0, first_half=True):
        super(HalfSplit, self).__init__()
        self.first_half = first_half
        self.dim = dim

    def forward(self, input):
        # 对input的channesl进行分半操作
        splits = torch.chunk(input, 2, dim=self.dim)  # 由于shape=[b, c, h, w],对于dim=1，针对channels
        return splits[0] if self.first_half else splits[1]  # 返回其中的一半


# channels shuffle增加组间交流
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


# ShuffleNet的基本单元
class ShuffleNetUnits(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups):
        super(ShuffleNetUnits, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # 如果stride = 2，由于主分支需要加上从分支的channels，为了两者加起来等于planes，所以需要先减一下
        if self.in_channels != self.out_channels:
            mid_channels = out_channels - in_channels
        # 如果stride = 2，mid_channels是一半，直接除以2即可
        else:
            mid_channels = out_channels // 2
            in_channels = mid_channels
            # 进行两次切分，一次接受一半，一次接受另外一半
            self.first_half = HalfSplit(dim=1, first_half=True)  # 对channels进行切半操作, 第一次分: first_half=True
            self.second_split = HalfSplit(dim=1, first_half=False)  # 返回输入的另外一半channesl，两次合起来才是完整的一份channels

        # 两个结构的主分支都是一样的，只是3x3DW卷积中的stride不一样，所以可以调用同样的self.bottleneck，stride会自动改变
        self.bottleneck = nn.Sequential(
            Conv1x1BNReLU(in_channels, in_channels),  # 没有改变channels
            Conv3x3BN(in_channels, mid_channels, stride, groups),  # 升维
            Conv1x1BNReLU(mid_channels, mid_channels)  # 没有改变channels
        )

        # 结构(d)的从分支，3x3的DW卷积——>1x1卷积
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Sequential(
                Conv3x3BN(in_channels=in_channels, out_channels=in_channels, stride=stride, groups=groups),
                Conv1x1BNReLU(in_channels, in_channels)
            )

        self.channel_shuffle = ChannelShuffle(groups)

    def forward(self, x):
        # stride = 2: 对于结构(d)
        if self.in_channels != self.out_channels:
            # if self.stride > 1:
            x1 = self.bottleneck(x)  # torch.Size([1, 220, 28, 28])
            x2 = self.shortcut(x)  # torch.Size([1, 24, 28, 28])
        # 两个分支作concat操作之后, 输出的channels便为224，与planes[0]值相等
        # out输出为: torch.Size([1, 244, 28, 28])

        # stride = 1: 对于结构(c)
        else:
            x1 = self.first_half(x)  # 一开始直接将channels等分两半，x1称为主分支的一半，此时的x1: channels = 112
            x2 = self.second_split(x)  # x2称为输入的另外一半channels: 此时x2:: channels = 112
            x1 = self.bottleneck(x1)  # 结构(c)的主分支处理
        # 两个分支作concat操作之后, 输出的channels便为224，与planes[0]值相等
        # out输出为: torch.Size([1, 244, 28, 28])

        out = torch.cat([x1, x2], dim=1)  # torch.Size([1, 244, 28, 28])
        out = self.channel_shuffle(out)  # ShuffleNet的精髓
        return out


def stack_same_dim(x):
    """Stack a list/dict of 4D tensors of same img dimension together."""
    # Collect tensor with same dimension into a dict of list
    output = {}

    # Input is list
    if isinstance(x, list):
        for i in range(len(x)):
            if isinstance(x[i], list):
                for j in range(len(x[i])):
                    shape = tuple(x[i][j].shape)
                    if shape in output.keys():
                        output[shape].append(x[i][j])
                    else:
                        output[shape] = [x[i][j]]
            else:
                shape = tuple(x[i].shape)
                if shape in output.keys():
                    output[shape].append(x[i])
                else:
                    output[shape] = [x[i]]
    else:
        for k in x.keys():
            shape = tuple(x[k].shape[2:4])
            if shape in output.keys():
                output[shape].append(x[k])
            else:
                output[shape] = [x[k]]

    # Concat the list of tensors into single tensor
    for k in output.keys():
        output[k] = torch.cat(output[k], dim=1)

    return output
