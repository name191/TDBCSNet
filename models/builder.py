import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight
from utils.load_utils import load_pretrain
from functools import partial

from engine.logger import get_logger

logger = get_logger()


class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255),
                 norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        # import backbone and decoder
        if cfg.backbone == 'my_backbone':
            logger.info('Using backbone: my_backbone')
            self.channels = [64, 128, 256, 512]
            from .encoders.dual_vmamba import my_backbone as backbone
            self.backbone = backbone(dims=self.channels[0], depths=[2, 2, 9, 2], pretrainpath=cfg.pretrained_model)
        elif cfg.backbone == 'TDBCSNet_encoder':
            logger.info('Using backbone: TDBCSNet_encoder')
            self.channels = [64, 128, 256, 512]
            from .encoders.dual_vmamba import TDBCSNet_encoder as backbone
            self.backbone = backbone(dims=self.channels[0], depths=[2, 2, 9, 2], pretrainpath=cfg.pretrained_model)

        self.aux_head = None

        ##############################################################################################
        if cfg.decoder == 'myMambaDecoder':
            logger.info('Using myMambaDecoder Decoder')
            from .decoders.MambaDecoder import myMambaDecoder
            self.deep_supervision = False
            self.decode_head = myMambaDecoder(img_size=[cfg.image_height, cfg.image_width], depths=[4, 4, 4, 4],
                                                in_channels=self.channels, num_classes=cfg.num_classes,
                                                embed_dim=self.channels[0], deep_supervision=self.deep_supervision)
        elif cfg.decoder == 'TDBCSNet_decoder':
            logger.info('Using TDBCSNet_decoder Decoder')
            from .decoders.MambaDecoder import TDBCSNet_decoder
            self.deep_supervision = False
            self.decode_head = TDBCSNet_decoder(img_size=[cfg.image_height, cfg.image_width], depths=[4, 4, 4, 4],
                                                in_channels=self.channels, num_classes=cfg.num_classes,
                                                embed_dim=self.channels[0], deep_supervision=self.deep_supervision)
        self.criterion = criterion
        # if self.criterion:
        # self.init_weights(cfg, pretrained=cfg.pretrained_model)

    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            if cfg.backbone != 'vmamba':
                logger.info('Loading pretrained model: {}'.format(pretrained))
                self.backbone.init_weights(pretrained=pretrained)
        logger.info('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                    self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                    mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                        self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                        mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb):
        if not self.deep_supervision:
            orisize = rgb.shape
            x = self.backbone(rgb)
            out = self.decode_head.forward(x)
            # out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
            if self.aux_head:
                aux_fm = self.aux_head(x[self.aux_index])
                aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
                return out, aux_fm
            return out
        else:
            x = self.backbone(rgb)
            x_last, x_output_0, x_output_1, x_output_2 = self.decode_head.forward(x)
            return x_last, x_output_0, x_output_1, x_output_2

    def forward(self, rgb, label=None):
        if not self.deep_supervision:
            if self.aux_head:
                out, aux_fm = self.encode_decode(rgb)
            else:
                out = self.encode_decode(rgb)
            if label is not None:
                loss = self.criterion(out, label)
                if self.aux_head:
                    loss += self.aux_rate * self.criterion(aux_fm, label.long())
                return loss
            return out
        else:
            x_last, x_output_0, x_output_1, x_output_2 = self.encode_decode(rgb)
            if label is not None:
                loss = self.criterion(x_last, label.long())
                loss += self.criterion(x_output_0, label.long())
                loss += self.criterion(x_output_1, label.long())
                loss += self.criterion(x_output_2, label.long())
                return loss
            return x_last

    def flops(self, shape=(3, 512, 512)):
        from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
        import copy

        '''
        code from
        https://github.com/MzeroMiko/VMamba/blob/main/classification/models/vmamba.py#L4
        '''

        # shape = self.__input_shape__[1:]
        supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = (torch.randn((1, *shape), device=next(model.parameters()).device))
        print(len(input))
        for i in input:
            print(i.shape)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=input, supported_ops=supported_ops)

        del model, input
        # return sum(Gflops.values())
        return f"params {params} GFLOPs {sum(Gflops.values())}"


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops