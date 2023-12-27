from thop import clever_format
from thop import profile
from Super import *
# from torchvision import datasets, transforms
# from torchvision.transforms.functional import to_grayscale
# import pywt
import numpy as np
# import cv2


class ShuffleNetV2(nn.Module):
    def __init__(self, planes, layers, groups, is_shuffle2_0, num_classes=7):
        super(ShuffleNetV2, self).__init__()
        # self.groups = 1
        self.groups = groups

        # input: torch.Size([1, 4, 16, 16])
        self.stage1_1 = nn.Sequential(
            # 结构图中，对于conv1与MaxPool的stride均为2
            Conv3x3BNReLU(in_channels=1, out_channels=24, stride=2, groups=1),  # torch.Size([1, 24, 32, 32])
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # torch.Size([2, 24, 16, 16])
        )
        self.stage2_1 = nn.Sequential(
            # 结构图中，对于conv1与MaxPool的stride均为2
            Conv3x3BNReLU(in_channels=4, out_channels=24, stride=1, groups=1),  # torch.Size([1, 24, 16, 16])
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # torch.Size([2, 24, 8, 8])

        )
        self.stage1_2 = self._make_layer(24, planes[0], layers[0], True, 2)  # torch.Size([1, 244, 8, 8])
        self.stage2_2 = self._make_layer(24, planes[0], layers[0], True, 1)  # torch.Size([1, 244, 8, 8])
        self.stage3_1 = self._make_layer(planes[0]*2, planes[1], layers[1], True, 1)  # torch.Size([1, 488, 8, 8])
        self.stage3_2 = self._make_layer(planes[1], planes[1], layers[1], True, 1)  # torch.Size([1, 488, 8, 8])
        self.stage4 = self._make_layer(planes[1], planes[2], layers[2], True, 1)  # torch.Size([1, 976, 8, 8])

        # 0.5x / 1x / 1.5x 输出为1024, 2x 输出为 2048
        self.conv5 = nn.Conv2d(in_channels=planes[2], out_channels=256 * is_shuffle2_0, kernel_size=1, stride=1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # torch.Size([1, 976, 1, 1])
        self.dropout = nn.Dropout(p=0.2)  # 丢失概率为0.2

        # 0.5x / 1x / 1.5x 输入为1024, 2x 输入为 2048
        self.linear = nn.Linear(in_features=256 * is_shuffle2_0, out_features=num_classes)

        self.init_params()

        self.SoftPlus = nn.Softplus()
        self.relu = nn.ReLU(inplace=False)

        self.ppa_ca2_1 = nn.Sequential(ca_layer(in_channel=planes[0], out_channel=planes[0]),
                                       # nn.BatchNorm2d(planes[0]),
                                       # nn.ReLU(inplace=False)
                                       )
        self.ppa_ca2_2 = ca_layer(in_channel=planes[1], out_channel=planes[1])

        self.sse_sa1_1 = nn.Sequential(sa_layer(k_size=3),
                                       )
        self.sse_sa1_2 = sa_layer(k_size=3)
        # self.sse_sa1_3 = sa_layer(k_size=3)
        # 此处的is_stage2作用不大，以为均采用3x3的DW卷积，也就是group=1的组卷积

    def _make_layer(self, in_channels, out_channels, block_num, is_stage2, stride):
        layers = []
        layers.append(ShuffleNetUnits(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                      groups=1 if is_stage2 else self.groups))

        # 对于stride = 1的情况，对应结构(c): 一开始就切分channel，主分支经过1x1——>3x3——>1x1再与shortcut进行concat操作
        for idx in range(1, block_num):
            layers.append(
                ShuffleNetUnits(in_channels=out_channels, out_channels=out_channels, stride=1, groups=self.groups))
        return nn.Sequential(*layers)

    # 何凯明的方法初始化权重
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # input: torch.Size([1, 3, 224, 224])
    def forward(self, Ms, Pan):
        Pan = self.stage1_1(Pan)  # torch.Size([1, 24, 16, 16])
        Pan = self.stage1_2(Pan)  # torch.Size([1, 32, 8, 8])
        Pan = self.sse_sa1_1(Pan) + Pan

        Ms = self.stage2_1(Ms)  # torch.Size([1, 24, 8, 8])
        Ms = self.stage2_2(Ms)  # torch.Size([1, 32, 8, 8])
        Ms = self.ppa_ca2_1(Ms) + Ms

        fusion = torch.cat([Ms, Pan], 1)  # torch.Size([1, 64, 8, 8])
        fusion = self.stage3_1(fusion)
        fusion = self.sse_sa1_2(fusion) + fusion
        fusion = self.stage3_2(fusion)  # torch.Size([1, 488, 14, 14])
        fusion = self.ppa_ca2_2(fusion) + fusion
        fusion = self.stage4(fusion)  # torch.Size([1, 976, 7, 7])

        fusion = self.conv5(fusion)  # torch.Size([1, 2048, 7, 7])
        fusion = self.global_pool(fusion)  # torch.Size([1, 2048, 1, 1])
        fusion = fusion.view(fusion.size(0), -1)  # torch.Size([1, 2048])
        fusion = self.dropout(fusion)
        out = self.linear(fusion)  # torch.Size([1, 5])
        out = self.SoftPlus(out)
        # out = self.relu(out)
        return out


class ShuffleNetV2_ms(nn.Module):
    def __init__(self, planes, layers, groups, is_shuffle2_0, num_classes=7):
        super(ShuffleNetV2_ms, self).__init__()
        # self.groups = 1
        self.device = None
        self.groups = groups

        self.stage1_1 = nn.Sequential(
            Conv3x3BNReLU(in_channels=4, out_channels=24, stride=1, groups=1))  # torch.Size([1, 24, 16, 16])
        self.stage1_2 = self._make_layer(24, planes[0], layers[0], True, 2)
        self.stage1_3 = self._make_layer(planes[0], planes[1], layers[1], False, 1)

        self.stage2_1 = nn.Sequential(
            Conv3x3BNReLU(in_channels=4, out_channels=24, stride=1, groups=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # torch.Size([1, 24, 8, 8])
        )
        self.stage2_2 = self._make_layer(24, planes[0], layers[0], True, 1)
        self.stage2_3 = self._make_layer(planes[0] + 24, planes[1], layers[1], False, 1)
        self.down1_1 = nn.Sequential(
            # 结构图中，对于conv1与MaxPool的stride均为2
            Conv3x3BNReLU(in_channels=24, out_channels=24, stride=2, groups=1),
            # torch.Size([1, 24, 8, 8])
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # torch.Size([1, 24, 8, 8])
        )
        self.down2_1 = nn.Sequential(
            # 结构图中，对于conv1与MaxPool的stride均为2
            Conv3x3BNReLU(in_channels=24, out_channels=24, stride=2, groups=1),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # torch.Size([1, 24, 8, 8])
        )
        self.stage4 = self._make_layer(planes[1], planes[2], layers[2], False, 1)  # torch.Size([1, 976, 8, 8])

        # 0.5x / 1x / 1.5x 输出为1024, 2x 输出为 2048
        self.conv5 = nn.Conv2d(in_channels=planes[2], out_channels=256 * is_shuffle2_0, kernel_size=1, stride=1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # torch.Size([1, 976, 1, 1])
        self.dropout = nn.Dropout(p=0.2)  # 丢失概率为0.2

        # 0.5x / 1x / 1.5x 输入为1024, 2x 输入为 2048
        self.linear = nn.Linear(in_features=256 * is_shuffle2_0, out_features=num_classes)

        self.init_params()

        self.SoftPlus = nn.Softplus()
        self.relu = nn.ReLU(inplace=False)

        self.downsamplewavelet = nn.Sequential(*[nn.Upsample(scale_factor=2), Downsamplewave1(wavename='haar')])

        self.fc = nn.Sequential(
            nn.Linear(4, 4 // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(4 // 2, 4, bias=False),
            nn.Sigmoid()
        )

        # self.coefs1_1 = use_l(in_planes=8)
        self.coefs1_1 = nn.Sequential(use_l(in_planes=24), nn.BatchNorm2d(24), nn.ReLU(inplace=False))
        self.coefs2_1 = nn.Sequential(use_l(in_planes=24), nn.BatchNorm2d(24), nn.ReLU(inplace=False))
        self.coefs3_1 = nn.Sequential(use_l(in_planes=24+planes[0]), nn.BatchNorm2d(24+planes[0]), nn.ReLU(inplace=False))
        self.coefs4_1 = nn.Sequential(use_l(in_planes=planes[1]), nn.BatchNorm2d(planes[1]), nn.ReLU(inplace=False))
        self.coefs5_1 = nn.Sequential(use_l(in_planes=planes[2]), nn.BatchNorm2d(planes[2]), nn.ReLU(inplace=False))

        self.up = nn.Sequential(Conv3x3BNReLU(in_channels=4, out_channels=1, stride=1, groups=1))

    # 此处的is_stage2作用不大，以为均采用3x3的DW卷积，也就是group=1的组卷积

    def _make_layer(self, in_channels, out_channels, block_num, is_stage2, stride):
        layers = []
        # 在ShuffleNetV2中，每个stage的第一个结构的stride均为2；此stage的其余结构的stride均为1.
        # 对于stride =2 的情况，对应结构(d): 一开始无切分操作，主分支经过1x1——>3x3——>1x1，从分支经过3x3——>1x1，两个分支作concat操作
        layers.append(ShuffleNetUnits(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                      groups=1 if is_stage2 else self.groups))

        # 对于stride = 1的情况，对应结构(c): 一开始就切分channel，主分支经过1x1——>3x3——>1x1再与shortcut进行concat操作
        for idx in range(1, block_num):
            layers.append(
                ShuffleNetUnits(in_channels=out_channels, out_channels=out_channels, stride=1, groups=self.groups))
        return nn.Sequential(*layers)

    # 何凯明的方法初始化权重
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x1 = self.stage1_1(x)  # torch.Size([1, 24, 16, 16])
        x1 = self.coefs1_1(x1)
        x1 = self.down1_1(x1)

        x2_1 = self.stage2_1(x)
        x2_1 = self.coefs2_1(x2_1)
        x2_1 = self.stage2_2(x2_1)

        x2_2 = torch.cat((x2_1, x1), 1)
        x2_2 = self.stage2_3(x2_2)
        x2_2 = self.coefs4_1(x2_2)
        x2_2 = self.stage4(x2_2)  # torch.Size([1, 976, 7, 7])
        x2_2 = self.coefs5_1(x2_2)

        x2_2 = self.conv5(x2_2)  # torch.Size([1, 2048, 7, 7])
        x2_2 = self.global_pool(x2_2)  # torch.Size([1, 2048, 1, 1])
        x2_2 = x2_2.view(x2_2.size(0), -1)  # torch.Size([1, 2048])
        x2_2 = self.dropout(x2_2)
        out = self.linear(x2_2)  # torch.Size([1, 5])
        out = self.SoftPlus(out)
        return out


class ShuffleNetV2_pan(nn.Module):
    # shufflenet_v2_x2_0: planes = [244, 488, 976]  layers = [4, 8, 4]
    # shufflenet_v2_x1_5: planes = [176, 352, 704]  layers = [4, 8, 4]
    def __init__(self, planes, layers, groups, is_shuffle2_0, num_classes=7, n_levs=[0, 3, 3, 3], variant="SSF",
                 spec_type="all"):
        super(ShuffleNetV2_pan, self).__init__()
        # self.groups = 1
        self.device = None
        self.groups = groups
        self.n_levs = n_levs
        self.variant = variant
        self.spec_type = spec_type

        self.stage2_1 = nn.Sequential(Conv3x3BNReLU(in_channels=1, out_channels=24, stride=1, groups=1), )
        self.stage3_1 = nn.Sequential(Conv3x3BNReLU(in_channels=1, out_channels=24, stride=1, groups=1), )
        self.down1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # torch.Size([2, 24, 16, 16])
        self.down2 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # torch.Size([2, 24, 16, 16])

        self.stage3_2 = self._make_layer(24, planes[0], layers[0], True, 1)  # torch.Size([2, 244, 8, 8])
        # self.stage2 = self._make_layer((24+8), planes[0], layers[0], True, 1)  # torch.Size([2, 244, 8, 8])
        self.stage3_3 = self._make_layer((planes[0] + 24), planes[1], layers[1], False, 1)  # torch.Size([2, 488, 8, 8])
        # # self.stage3 = self._make_layer((planes[0]+32), planes[1], layers[1], False, 1)  # torch.Size([2, 488, 8, 8])
        self.stage3_4 = self._make_layer(planes[1], planes[2], layers[2], False, 2)  # torch.Size([2, 976, 8, 8])

        # 0.5x / 1x / 1.5x 输出为1024, 2x 输出为 2048
        self.conv5 = nn.Conv2d(in_channels=planes[2], out_channels=256 * is_shuffle2_0, kernel_size=1, stride=1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # torch.Size([1, 976, 1, 1])
        self.dropout = nn.Dropout(p=0.2)  # 丢失概率为0.2

        # 0.5x / 1x / 1.5x 输入为1024, 2x 输入为 2048
        self.linear = nn.Linear(in_features=256 * is_shuffle2_0, out_features=num_classes)

        self.init_params()
        self.SoftPlus = nn.Softplus()
        self.relu = nn.ReLU(inplace=False)
        self.Sig = nn.Sigmoid()
        # self.bn = nn.BatchNorm2d(1)

        self.coefs1 = use_h(in_planes=1)
        self.coefs2 = use_h(in_planes=1)

        self.BI = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    # 此处的is_stage2作用不大，以为均采用3x3的DW卷积，也就是group=1的组卷积
    def _make_layer(self, in_channels, out_channels, block_num, is_stage2, stride):
        layers = []
        # 在ShuffleNetV2中，每个stage的第一个结构的stride均为2；此stage的其余结构的stride均为1.
        # 对于stride =2 的情况，对应结构(d): 一开始无切分操作，主分支经过1x1——>3x3——>1x1，从分支经过3x3——>1x1，两个分支作concat操作
        layers.append(ShuffleNetUnits(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                      groups=1 if is_stage2 else self.groups))

        # 对于stride = 1的情况，对应结构(c): 一开始就切分channel，主分支经过1x1——>3x3——>1x1再与shortcut进行concat操作
        for idx in range(1, block_num):
            layers.append(
                ShuffleNetUnits(in_channels=out_channels, out_channels=out_channels, stride=1, groups=self.groups))
        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):

        x1 = self.BI(x)
        coefs1_1, coefs1_2 = self.coefs1(x1)
        nn.BatchNorm2d(1)
        coefs2_1, coefs2_2 = self.coefs2(x)

        x2 = self.stage2_1(x)  # torch.Size([1, 24, 56, 56])
        x2 = x2 * coefs1_1
        x2 = self.down1(x2)

        x3_1 = x[:, :, ::2, ::2]
        x3_1 = self.stage3_1(x3_1)
        x3_1 = x3_1 * coefs2_1
        x3_1 = self.stage3_2(x3_1)  # torch.Size([1, 244, 28, 28])
        x3_2 = torch.cat((x2, x3_1), 1)
        x3_2 = self.stage3_3(x3_2)
        x3_2 = x3_2 * coefs1_2
        x3_2 = self.down2(x3_2)
        x3_2 = x3_2 * coefs2_2
        x3_2 = self.stage3_4(x3_2)

        x3_2 = self.conv5(x3_2)  # torch.Size([1, 2048, 7, 7])
        x3_2 = self.global_pool(x3_2)  # torch.Size([1, 2048, 1, 1])
        x3_2 = x3_2.view(x3_2.size(0), -1)  # torch.Size([1, 2048])
        x3_2 = self.dropout(x3_2)
        out = self.linear(x3_2)  # torch.Size([1, 5])
        out = self.SoftPlus(out)
        return out


def shufflenet_v2_x2_0_ms(num_classes):
    planes = [64, 160, 384]
    layers = [1, 1, 1]
    model = ShuffleNetV2_ms(planes, layers, 1, 3, num_classes=num_classes)
    return model


def shufflenet_v2_x2_0_pan(num_classes):
    planes = [64, 160, 384]
    layers = [1, 1, 1]
    model = ShuffleNetV2_pan(planes, layers, 1, 3, num_classes=num_classes)
    return model


def shufflenet_v2(num_classes):
    planes = [64, 160, 384]
    layers = [1, 1, 1]
    model = ShuffleNetV2(planes, layers, 1, 3, num_classes=num_classes)
    return model


class TMC(nn.Module):

    def __init__(self, classes, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of views
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(TMC, self).__init__()
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        self.Classifiers = shufflenet_v2(classes)
        self.Classifiersshuffle_ms = shufflenet_v2_x2_0_ms(classes)
        self.Classifiersshuffle_pan = shufflenet_v2_x2_0_pan(classes)

    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        global alpha_a

        def Battle_three(alpha1, alpha2, alpha3):
            Alpha = dict()
            Alpha[0], Alpha[1], Alpha[2] = alpha1, alpha2, alpha3
            S, E, b, u = dict(), dict(), dict(), dict()
            for v in range(len(Alpha)):
                S[v] = torch.sum(Alpha[v], dim=1, keepdim=True)
                u[v] = self.classes / S[v]

            battle2 = torch.lt(u[0], u[2])
            battle3 = torch.lt(u[1], u[2])
            # 统计每个张量中最小元素的总数
            count2 = torch.sum(battle2).item()
            count3 = torch.sum(battle3).item()
            rate_BN = int(0.6 * len(u[0]))

            # 找出最小元素总数最多的张量
            if count2 < rate_BN and count3 < rate_BN:
                out = alpha3
            else:
                u_list = [u[0], u[1]]
                mask = u_list[0] < u_list[1]
                out = torch.where(mask, Alpha[0], Alpha[1])
            return out

        alpha_a = Battle_three(alpha[0], alpha[1], alpha[2])
        return alpha_a

    def forward(self, Ms, Pan):
        evidence_Ms = self.Classifiersshuffle_ms(Ms)
        evidence_Pan = self.Classifiersshuffle_pan(Pan)
        evidence_fusion = self.Classifiers(Ms, Pan)

        alpha = dict()
        alpha[0] = evidence_Ms + 1
        alpha[1] = evidence_Pan + 1
        alpha[2] = evidence_fusion + 1

        Alpha_a = self.DS_Combin(alpha)
        evidence_a = Alpha_a - 1

        return evidence_a


if __name__ == "__main__":
    pan = torch.randn(2, 1, 64, 64)
    ms = torch.randn(2, 4, 16, 16)
    TMC = TMC(11)
    # global_step为迭代次数epoch
    out_result = TMC(ms, pan)
    # 保存tensor文件
    flops, params = profile(TMC, inputs=(ms, pan,))
    flops, params = clever_format([flops, params], '%.3f')
    print('模型参数：', params)
    print('每一个样本浮点运算量：', flops)

    print(out_result)
    print(type(out_result))
    print(out_result.shape)

