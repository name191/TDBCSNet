import torch.nn as nn

from .DSCNet import TopologyawareBranch
from .crossnet import CrossNet
from engine.logger import get_logger
from models.encoders.vmamba import Context_branch, SemanticBranch, Front_Net, Permute
from ..rep import FFM, AFFM

logger = get_logger()
class my_backbone_detail(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 norm_layer=nn.LayerNorm,
                 depths=[2, 2, 27, 2],  # [2,2,27,2] for vmamba small
                 dims=96,
                 pretrained=None,
                 mlp_ratio=4.0,
                 downsample_version='v1',
                 ape=False,
                 img_size=[480, 640],
                 patch_size=4,
                 drop_path_rate=0.2,
                 **kwargs):
        super().__init__()

        self.ape = ape
        self.numlayer = len(depths)
        self.vssm = Context_branch(
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )
        self.crosspath=CrossNet(layers=len(depths)+2,dim=dims//4,norm='gn')
        self.fuses=nn.ModuleList(
            [
            FFM(dims*(2**i))
            for i in range(self.numlayer)]
        )
    def forward_features(self, x_rgb):

        global x_fuse
        B = x_rgb.shape[0]

        outs_rgb, en0, en1 = self.vssm(x_rgb)  # B x C x H x W
        outs_cross=self.crosspath(x_rgb)

        outs_fused = [en0, en1]

        for i in range(self.numlayer):
            if self.ape:
                # this has been discarded
                out_rgb = self.absolute_pos_embed[i].to(outs_rgb[i].device) + outs_rgb[i]
            else:
                out_rgb = outs_rgb[i]
            # x_fuse = out_rgb+outs_cross[i+2]
            if i<self.numlayer-1:
                x_fuse = self.fuses[i](out_rgb,outs_cross[i+2])
            else:
                x_fuse=out_rgb
            outs_fused.append(x_fuse)
        return outs_fused

    def forward(self, x_rgb):
        out = self.forward_features(x_rgb)
        return out

class my_backbone(my_backbone_detail):
    def __init__(self, dims, depths, pretrainpath, fuse_cfg=None, **kwargs):
        super(my_backbone, self).__init__(
            depths=depths,
            dims=dims,
            pretrained=pretrainpath,
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.2,
        )

class TDBCSNet_encoder_base(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 norm_layer=nn.LayerNorm,
                 depths=[2, 2, 9, 2],  # [2,2,27,2] for vmamba small
                 dims=96,
                 pretrained=None,
                 mlp_ratio=4.0,
                 downsample_version='v1',
                 ape=False,
                 img_size=[480, 640],
                 patch_size=4,
                 drop_path_rate=0.2,
                 **kwargs):
        super().__init__()

        self.ape = ape
        self.numlayer = len(depths)
        self.front=Front_Net(resultdim=dims//2,norm='gn')
        self.tomamba=nn.Sequential(
            nn.Conv2d(dims // 2, dims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims),
            Permute(0, 2, 3, 1),
        )
        self.tocnn=nn.GroupNorm(dims//2//4,dims//2)
        self.vssm = SemanticBranch(
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )
        self.CNN=TopologyawareBranch(layers=len(depths)+1,dim=dims//4)
        self.fuses=nn.ModuleList(
            [
            AFFM(dims*(2**i))
            for i in range(self.numlayer)]
        )
        self.pool=nn.MaxPool2d(2,2)
    def forward_features(self, x_rgb):

        global x_fuse
        B = x_rgb.shape[0]

        # outs_cnn=self.CNN(x_rgb)
        # outs_mamba=self.vssm(outs_cnn[1])
        en0,en1=self.front(x_rgb)
        en2 = self.pool(en1)
        x_mamba = self.tomamba(en2)
        x_cnn = self.tocnn(en2)

        outs_mamba = self.vssm(x_mamba)
        outs_cnn=self.CNN(x_cnn)

        outs_fused = [en0, en1]
        for i in range(self.numlayer):
            if self.ape:
                # this has been discarded
                out_m = self.absolute_pos_embed[i].to(outs_mamba[i].device) + outs_mamba[i]
            else:
                out_m = outs_mamba[i]
            if i<self.numlayer-1:
                x_fuse = self.fuses[i](out_m,outs_cnn[i])
            else:
                x_fuse=out_m
            outs_fused.append(x_fuse)
        return outs_fused

    def forward(self, x_rgb):
        out = self.forward_features(x_rgb)
        return out

class TDBCSNet_encoder(TDBCSNet_encoder_base):
    def __init__(self, dims, depths, pretrainpath,fuse_cfg=None, **kwargs):
        super(TDBCSNet_encoder, self).__init__(
            depths=depths,
            dims=dims,
            pretrained=pretrainpath,
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.2
        )
