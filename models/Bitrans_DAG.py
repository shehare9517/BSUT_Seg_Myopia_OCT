import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from model.unet_DAG_3 import up_block
from model.transunet import ResNetV2
from model.conv_trans_utils import block_trans
import pdb
from model.transformer import ViT
from model.DualAtt import DualAtt
from model.ResNet import resnet34
from model.PVT import pvt_v2_b2


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        x = x * y.expand_as(x)
        return y


class Bi_TransNet(nn.Module):
    def __init__(self, in_ch, num_class, reduce_size=8, block_list='0123', num_blocks=[3, 4, 6, 3], projection='interp',
                 num_heads=[1, 2, 4, 8], attn_drop=0., proj_drop=0., rel_pos=True, block_units=(3, 4, 9),
                 width_factor=1):

        super().__init__()

        self.resnet = resnet34(pretrained=True)
        if '0' in block_list:
            self.trans_0 = block_trans(64, num_blocks[-4], 64 // num_heads[-4], attn_drop=attn_drop,
                                       proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                       rel_pos=rel_pos)
        else:
            self.trans_0 = nn.Identity()

        if '1' in block_list:
            self.trans_1 = block_trans(128, num_blocks[-3], 128 // num_heads[-3], attn_drop=attn_drop,
                                       proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                       rel_pos=rel_pos)

        else:
            self.trans_1 = nn.Identity()

        if '2' in block_list:
            self.trans_2 = block_trans(256, num_blocks[-2], 256 // num_heads[-2], attn_drop=attn_drop,
                                       proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                       rel_pos=rel_pos)

        else:
            self.trans_2 = nn.Identity()

        if '3' in block_list:
            self.trans_3 = block_trans(512, num_blocks[-1], 512 // num_heads[-1], attn_drop=attn_drop,
                                       proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                       rel_pos=rel_pos)

        else:
            self.trans_3 = nn.Identity()

        self.up1 = up_block(512, 256, scale=(2, 2), num_block=1)
        self.up2 = up_block(256, 128, scale=(2, 2), num_block=1)
        self.up3 = up_block(128, 64, scale=(2, 2), num_block=1)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)

        self.output = nn.Conv2d(64, num_class, kernel_size=3, padding=1, bias=True)
        self.dualatt = DualAtt(512)
        self.dualatt_2 = DualAtt(256)
        self.dualatt_3 = DualAtt(128)
        self.dualatt_4 = DualAtt(64)

        # self.transformer = ViT(in_channels=3,
        #                        patch_size=16,
        #                        emb_size=512,
        #                        img_size=256,
        #                        depth=4,
        #                        n_regions=(256 // 16) ** 2)
        self.transformer = pvt_v2_b2()

        self.boundary = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, stride=1),
                                      nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                      nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False),
                                      nn.Sigmoid())
        self.se = SE_Block(c=64)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        trans_feas = self.transformer(x)

        x, features = self.resnet(x)

        # x = self.dualatt(x)
        out3 = self.trans_3(x)  # ([4, 512, 8, 8])
        out3 = self.dualatt(out3, trans_feas[3])
        # out3 = self.dualatt(out3)
        out2 = self.trans_2(features[0])  # ([4, 256, 16, 16]))
        out2 = self.dualatt_2(out2, trans_feas[2])
        # out2 = self.dualatt_2(out2)
        out1 = self.trans_1(features[1])  # ([4, 128, 32, 32])
        out1 = self.dualatt_3(out1, trans_feas[1])
        out0 = self.trans_0(features[2])  # ([4, 64, 64, 64])
        out0 = self.dualatt_4(out0, trans_feas[0])


        out = self.up1(out3, out2)  # ([4, 256, 16, 16])
        out = self.up2(out, out1)  # ([4, 128, 32, 32])
        out = self.up3(out, out0)  # ([4, 64, 64, 64])
        out = self.up4(out)  # ([4, 64, 256, 256])

        Bound = self.boundary(out)  # ([4, 1, 256, 256])
        B = Bound.repeat_interleave(int(out.shape[1]), dim=1)   #([4, 64, 256, 256])
        out = self.se(out)   # ([4, 64, 1, 1])
        # out = torch.cat((out, B), dim=1)
        out = out * B

        # att = regional_distribution.repeat_interleave(int(out.shape[1]), dim=1)
        # att = trans_feas[3]
        # out = out * att
        # out = torch.cat((out, global_contexual), dim=1)

        out = self.output(out)
        out = torch.sigmoid(out)

        return out, Bound


if __name__ == '__main__':
    img = torch.randn(4, 3, 256, 256)
    img = img.cuda()
    Net = Bi_TransNet(in_ch=3, num_class=1)
    Net = Net.cuda()
    out = Net(img)
    print(out[0].shape)

