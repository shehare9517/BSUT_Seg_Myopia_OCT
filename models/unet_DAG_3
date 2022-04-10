import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(

            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),

            nn.BatchNorm2d(mid_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(

            nn.MaxPool2d(2),

            DoubleConv(in_channels, out_channels)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):

        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels

        if bilinear:

            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

        else:

            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):

        x1 = self.up(x1)

        # input is CHW

        diffY = x2.size()[2] - x1.size()[2]

        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,

                        diffY // 2, diffY - diffY // 2])

        # if you have padding issues, see

        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a

        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(inplanes),
                self.relu,
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        residue = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += self.shortcut(residue)

        return out


class BottleneckBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes // 4, stride=1)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes // 4, planes // 4, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes // 4)

        self.conv3 = conv1x1(planes // 4, planes, stride=1)
        self.bn3 = nn.BatchNorm2d(planes // 4)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(inplanes),
                self.relu,
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        residue = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += self.shortcut(residue)

        return out


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, bottleneck=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if bottleneck:
            self.conv2 = BottleneckBlock(out_ch, out_ch)
        else:
            self.conv2 = BasicBlock(out_ch, out_ch)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out


class down_block(nn.Module):
    def __init__(self, in_ch, out_ch, scale, num_block, bottleneck=False, pool=True):
        super().__init__()

        block_list = []

        if bottleneck:
            block = BottleneckBlock
        else:
            block = BasicBlock

        if pool:
            block_list.append(nn.MaxPool2d(scale))
            block_list.append(block(in_ch, out_ch))
        else:
            block_list.append(block(in_ch, out_ch, stride=2))

        for i in range(num_block - 1):
            block_list.append(block(out_ch, out_ch, stride=1))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x):
        return self.conv(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# class DAG(nn.Module):
#     """
#     CCA Block
#     """
#
#     def __init__(self, F_g, F_x):
#         super().__init__()
#         self.mlp_x = nn.Sequential(
#             Flatten(),
#             nn.Linear(F_x, F_x))
#         self.mlp_g = nn.Sequential(
#             Flatten(),
#             nn.Linear(F_g, F_x))
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, g, x):
#         # channel-wise attention
#         avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#         channel_att_x = self.mlp_x(avg_pool_x)
#         avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
#         channel_att_g = self.mlp_g(avg_pool_g)
#         channel_att_sum = (channel_att_x + channel_att_g) / 2.0
#         scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
#         x_after_channel = x * scale
#         g_after_channel = g * scale
#         xg_after_channel = x_after_channel + g_after_channel
#         out = self.relu(xg_after_channel)
#         return out

class DAG(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(DAG, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.conv1 = nn.Conv2d(2, 1, (7, 7), padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        avg_out_g = self.fc2(self.relu1(self.fc1(self.avg_pool(g))))
        max_out_g = self.fc2(self.relu1(self.fc1(self.max_pool(g))))
        out_g = avg_out_g + max_out_g
        out_g = out_g * g

        avg_out_x = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out_x = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out_x = avg_out_x + max_out_x
        out_x = out_x * x

        avg_sp_g = torch.mean(out_g, dim=1, keepdim=True)
        max_sp_g, _ = torch.max(out_g, dim=1, keepdim=True)
        sp_g = torch.cat([avg_sp_g, max_sp_g], dim=1)
        sp_g = self.conv1(sp_g)
        sp_g = sp_g * out_g

        avg_sp_x = torch.mean(out_x, dim=1, keepdim=True)
        max_sp_x, _ = torch.max(out_x, dim=1, keepdim=True)
        sp_x = torch.cat([avg_sp_x, max_sp_x], dim=1)
        sp_x = self.conv1(sp_x)
        sp_x = sp_x * out_x
        out = sp_g + sp_x

        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, scale=(2, 2), bottleneck=False):
        super().__init__()
        self.scale = scale

        self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        if bottleneck:
            block = BottleneckBlock
        else:
            block = BasicBlock

        block_list = []
        block_list.append(block(2 * out_ch, out_ch))

        for i in range(num_block - 1):
            block_list.append(block(out_ch, out_ch))

        self.conv = nn.Sequential(*block_list)
        # if in_ch == 64:
        #     in_ch = 32
        #     self.dag = DAG(in_ch // 2, out_ch // 2)
        # else:
        self.dag = DAG(in_ch // 2, in_ch // 2)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x1 = self.conv_ch(x1)

        x2 = self.dag(g=x1, x=x2)

        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)

        return out



