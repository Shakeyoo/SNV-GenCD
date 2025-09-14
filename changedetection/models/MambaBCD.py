import torch
import torch.nn.functional as F
import torch.nn as nn
from changedetection.models.Mamba_backbone import Backbone_VSSM  # 导入主干网络（编码器）
from classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute  # 导入相关模块
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
from changedetection.models.ChangeDecoder import ChangeDecoder  # 导入变化检测解码器
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat  # 用于高效张量操作
from timm.models.layers import DropPath, trunc_normal_  # timm库中的高级层
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count  # 计算 FLOPs 和参数量


class STMambaBCD(nn.Module):
    def __init__(self, pretrained, **kwargs):
        super(STMambaBCD, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        
        _NORMLAYERS = dict(ln=nn.LayerNorm, ln2d=LayerNorm2d, bn=nn.BatchNorm2d)
        _ACTLAYERS = dict(silu=nn.SiLU, gelu=nn.GELU, relu=nn.ReLU, sigmoid=nn.Sigmoid)

        norm_layer = _NORMLAYERS.get(kwargs['norm_layer'].lower())
        ssm_act_layer = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower())
        mlp_act_layer = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower())

        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        self.decoder = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.main_clf = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)


    def _upsample_add(self, x, y):
        """
        上采样并相加，用于特征融合
        `x` 进行双线性插值到 `y` 的尺寸，然后两者相加
        """
        _, _, H, W = y.size()  # 获取目标特征图的高度和宽度
        return F.interpolate(x, size=(H, W), mode='bilinear') + y  # 插值并相加
        
    def forward(self, pre_data, post_data):
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        output = self.decoder(pre_features, post_features)
        output = self.main_clf(output)
        output = F.interpolate(output, size=pre_data.size()[-2:], mode='bilinear')

        # 返回输出和用于PCL的特征（取最深层的pre和post特征）
        return output, pre_features[-1], post_features[-1]


