# TumorModel.py (修改后)

import math
import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

import random
from typing import Dict, Optional, Tuple


# ==================================================================================
# 新增：CBAM 注意力模块 (ChannelAttention, SpatialAttention, CBAM)
# ==================================================================================
# 在 TumorModel.py 中，找到并替换整个 ChannelAttention 类

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # 核心修复：确保中间通道数（hidden_planes）永远不会小于1
        hidden_planes = max(1, in_planes // ratio)

        self.fc = nn.Sequential(
            # 使用修复后的 hidden_planes
            nn.Conv3d(in_planes, hidden_planes, 1, bias=False),
            nn.ReLU(),
            # 使用修复后的 hidden_planes
            nn.Conv3d(hidden_planes, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


# ==================================================================================
# 优化的数据增强函数 (来自您提供的原始代码，功能完善)
# ==================================================================================
def augment3d(
        inp: torch.Tensor,
        augmentation_dict: Optional[Dict] = None,
        device: Optional[torch.device] = None
) -> torch.Tensor:
    # (此函数来自您提供的原始代码，保持不变)
    aug_config = augmentation_dict or {
        'flip': True, 'offset': True, 'offset_range': 0.1, 'rotate': True, 'scale': False,
        'scale_range': (0.8, 1.2), 'noise': False, 'noise_level': 0.01, 'elastic': False,
        'elastic_strength': 0.02, 'contrast': True, 'contrast_range': (0.8, 1.2),
        'brightness_range': (-0.1, 0.1), 'random_crop': False, 'crop_size': (32, 32, 32)
    }
    device = device or inp.device
    inp = inp.to(device)
    if aug_config.get('random_crop', False):
        crop_size = aug_config.get('crop_size', (32, 32, 32))
        assert all(c <= s for c, s in zip(crop_size, inp.shape[2:])), "裁剪尺寸不能大于输入数据的对应维度"
        start_d = random.randint(0, inp.size(2) - crop_size[0])
        start_h = random.randint(0, inp.size(3) - crop_size[1])
        start_w = random.randint(0, inp.size(4) - crop_size[2])
        inp = inp[:, :, start_d:start_d + crop_size[0], start_h:start_h + crop_size[1], start_w:start_w + crop_size[2]]
    transform_t = torch.eye(4, dtype=torch.float32, device=device)
    if aug_config.get('flip', True):
        for i in range(3):
            if random.random() > 0.5:
                transform_t[i, i] *= -1
    if aug_config.get('scale', False):
        scale_range = aug_config.get('scale_range', (0.8, 1.2))
        scale_factor = random.uniform(*scale_range)
        for i in range(3):
            transform_t[i, i] *= scale_factor
    if aug_config.get('offset', True):
        offset_range = aug_config.get('offset_range', 0.1)
        for i in range(3):
            random_offset = (random.random() * 2 - 1) * offset_range
            transform_t[3, i] = random_offset
    if aug_config.get('rotate', True):
        angle_rad = random.random() * np.pi * 2
        s, c = np.sin(angle_rad), np.cos(angle_rad)
        rotation_t = torch.tensor([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32,
                                  device=device)
        transform_t = transform_t @ rotation_t
    batch_size = inp.size(0)
    affine_matrix = transform_t[:3].unsqueeze(0).expand(batch_size, -1, -1)
    affine_grid = F.affine_grid(affine_matrix, inp.shape, align_corners=False)
    if aug_config.get('elastic', False):
        elastic_strength = aug_config.get('elastic_strength', 0.02)
        displacement = torch.randn_like(affine_grid) * elastic_strength
        affine_grid = affine_grid + displacement
    augmented = F.grid_sample(inp, affine_grid, padding_mode='border', align_corners=False)
    if aug_config.get('contrast', False):
        contrast_range = aug_config.get('contrast_range', (0.8, 1.2))
        brightness_range = aug_config.get('brightness_range', (-0.1, 0.1))
        contrast_factor = random.uniform(*contrast_range)
        brightness_shift = random.uniform(*brightness_range)
        augmented = augmented * contrast_factor + brightness_shift
        min_val, max_val = inp.min(), inp.max()
        augmented = torch.clamp(augmented, min_val, max_val)
    if aug_config.get('noise', False):
        noise_level = aug_config.get('noise_level', 0.01)
        noise = torch.randn_like(augmented) * noise_level
        augmented = augmented + noise
    return augmented


class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batchnorm = nn.BatchNorm3d(1)

        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)

        # !!! 注意 !!!
        # 由于模型结构改变，这里的 1152 可能不再正确。
        # 需要运行一次模型，打印出 conv_flat.size() 来确定新的正确维度。
        # 例如：
        # with torch.no_grad():
        #     model = LunaModel()
        #     out = model(torch.randn(1, 1, 32, 48, 48)))
        #     # 观察打印出的 conv_flat 的形状
        self.head_linear = nn.Linear(1152, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear, nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_batch):
        bn_output = self.tail_batchnorm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)

        conv_flat = block_out.view(
            block_out.size(0),
            -1,
        )
        # print("DEBUG: conv_flat shape:", conv_flat.shape) # 取消注释以找到正确的维度
        linear_output = self.head_linear(conv_flat)

        return linear_output, self.head_softmax(linear_output)


# ==================================================================================
# 修改后的 LunaBlock：加入了残差连接、批量归一化和CBAM注意力
# ==================================================================================
class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        # 结构: Conv -> BN -> ReLU
        self.conv1 = nn.Conv3d(in_channels, conv_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(conv_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(conv_channels, conv_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(conv_channels)

        self.relu2 = nn.ReLU(inplace=True)

        # 残差连接的下采样层，用于匹配通道数
        if in_channels != conv_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, conv_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(conv_channels),
            )
        else:
            self.downsample = None

        # 注意力模块
        self.cbam = CBAM(conv_channels)

        self.maxpool = nn.MaxPool3d(2, 2)

    def forward(self, input_batch):
        residual = input_batch

        # 主路径
        out = self.conv1(input_batch)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 添加残差连接
        if self.downsample:
            residual = self.downsample(residual)
        out += residual

        # 在残差连接后应用激活函数
        out = self.relu2(out)

        # 应用注意力机制
        out = self.cbam(out)

        return self.maxpool(out)