from classification.models.vmamba import VSSM, LayerNorm2d  # 导入 VSSM 模型和 LayerNorm2d 归一化层

import torch
import torch.nn as nn  # 导入 PyTorch 及神经网络模块
#


class Backbone_VSSM(VSSM):  # 继承 VSSM，定义一个新的主干网络 Backbone_VSSM
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer='ln2d', **kwargs):
        """
        :param out_indices: 指定哪些层的输出作为最终特征输出
        :param pretrained: 预训练模型路径
        :param norm_layer: 归一化层类型（'ln2d', 'bn', 'ln'）
        :param kwargs: 额外参数，传递给 VSSM 模型
        """
        kwargs.update(norm_layer=norm_layer)  # 将 norm_layer 参数添加到 kwargs 字典中
        super().__init__(**kwargs)  # 调用 VSSM 的构造函数，初始化主干网络

        # 判断是否使用批归一化（bn）或二维层归一化（ln2d），如果是，则通道优先存储
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])

        # 定义归一化层类型的映射表
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,  # 层归一化（LayerNorm）
            ln2d=LayerNorm2d,  # 2D 版本的层归一化（LayerNorm2d）
            bn=nn.BatchNorm2d,  # 批归一化（BatchNorm2d）
        )

        # 获取用户指定的归一化层，如果找不到则返回 None
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        # 记录需要输出的层索引
        self.out_indices = out_indices

        # 为 out_indices 指定的每个层，创建一个归一化层，并添加到模型中
        for i in out_indices:
            layer = norm_layer(self.dims[i])  # 获取对应通道数的归一化层
            layer_name = f'outnorm{i}'  # 归一化层的名称，例如 outnorm0, outnorm1, ...
            self.add_module(layer_name, layer)  # 将归一化层注册到模型中

        # 删除分类器部分，因为这个主干网络主要用于特征提取
        del self.classifier

        # 如果提供了预训练模型路径，则加载预训练权重
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        """
        加载预训练模型权重
        :param ckpt: 预训练权重文件路径
        :param key: 预训练模型的键（默认是 "model"）
        """
        if ckpt is None:  # 如果没有提供 ckpt，则直接返回
            return

        try:
            # 读取预训练权重，映射到 CPU
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"), weights_only=True)
            print(f"Successfully loaded checkpoint {ckpt}")  # 打印加载成功信息

            # 加载模型参数，允许部分不匹配
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)  # 打印不匹配的键值信息

        except Exception as e:
            print(f"Failed to load checkpoint from {ckpt}: {e}")  # 发生异常时打印错误信息

    def forward(self, x):
        """
        前向传播，提取输入的多尺度特征
        :param x: 输入图像数据 (B, C, H, W)
        :return: 选定层的特征列表（outs）
        """

        def layer_forward(l, x):
            """
            处理单个层的前向传播
            :param l: 当前处理的层
            :param x: 进入该层的特征
            :return: 处理后的特征图 o 以及下采样特征图 y
            """
            x = l.blocks(x)  # 经过该层的 Transformer 块
            y = l.downsample(x)  # 进行下采样
            return x, y

        # 通过 patch embedding 将输入图像转换为嵌入特征
        x = self.patch_embed(x)

        outs = []  # 用于存储输出的特征图

        # 遍历网络中的每一层
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x)  # 获取当前层的特征 o 以及下采样后的特征 x

            # 如果当前层索引 i 在 out_indices 里，则需要输出该层的特征
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')  # 获取该层对应的归一化层
                out = norm_layer(o)  # 归一化特征图 o

                # 如果通道维度不是优先的，则进行格式变换 (B, H, W, C) -> (B, C, H, W)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2).contiguous()

                outs.append(out)  # 存储归一化后的特征

        # 如果 out_indices 为空，则返回最后一层的特征
        if len(self.out_indices) == 0:
            return x

        return outs  # 返回所有指定层的特征输出
