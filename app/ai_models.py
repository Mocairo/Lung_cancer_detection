import torch
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from scipy import ndimage
from collections import namedtuple
import math

# --- 1. 从 util/unet.py 粘贴的代码 ---

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False,
                 batch_norm=False, up_mode='upconv'):
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        return self.last(x)

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out

# --- 2. 从您的代码中整合的模型和工具函数 ---

class UNetWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_batchnorm = nn.BatchNorm2d(kwargs['in_channels'])
        self.unet = UNet(**kwargs)
        self.final = nn.Sigmoid()

    def forward(self, input_batch):
        bn_output = self.input_batchnorm(input_batch)
        un_output = self.unet(bn_output)
        fn_output = self.final(un_output)
        return fn_output

class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, conv_channels, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(conv_channels, conv_channels, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(2, 2)
    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)
        return self.maxpool(block_out)

class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()
        self.tail_batchnorm = nn.BatchNorm3d(1)
        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)
        self.head_linear = nn.Linear(1152, 2)
        self.head_softmax = nn.Softmax(dim=1)

    def forward(self, input_batch):
        bn_output = self.tail_batchnorm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.head_linear(conv_flat)
        return linear_output, self.head_softmax(linear_output)

# --- 3. 模型加载 ---

def load_model(model_path, ModelClass, **kwargs):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在 {device} 设备上加载模型: {model_path.name}")

    model = ModelClass(**kwargs).to(device)

    state = torch.load(model_path, map_location=device)

    # 处理 DataParallel 包装的模型
    model_state = state['model_state']
    # 如果权重字典的键以 'module.' 开头，则去掉这个前缀
    if next(iter(model_state)).startswith('module.'):
        model_state = {k[7:]: v for k, v in model_state.items()}

    model.load_state_dict(model_state)
    model.eval()
    return model

# 应用启动时加载所有模型
try:
    seg_model = load_model(
        'rawdata/seg/models/seg/seg_2025-07-23_13.37.22_augment.best.state',
        UNetWrapper,
        in_channels=7, n_classes=1, depth=3, wf=4, padding=True, batch_norm=True, up_mode='upconv'
    )
    nodule_cls_model = load_model(
        'rawdata/tumor/models/nodule_cls/tumor_2025-07-21_08.37.08_100.best.state',
        LunaModel
    )
    tumor_cls_model = load_model(
        'rawdata/tumor/models/nodule_cls/tumor_2025-07-22_13.20.15_finetune_depth_2.best.state',
        LunaModel
    )
    print("所有AI模型加载成功。")
except Exception as e:
    print(f"模型加载时发生错误: {e}")
    seg_model, nodule_cls_model, tumor_cls_model = None, None, None


# --- 4. AI 分析流程 ---

def resample_image(itk_image, new_spacing=[1.0, 1.0, 1.0]):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    new_size = [
        int(round(osz * osp / nsp))
        for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(itk_image.GetDirection())
    resampler.SetOutputOrigin(itk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(itk_image)

def find_nodules(mask_array, ct_array, resampled_itk):
    labeled_mask, num_features = ndimage.label(mask_array)
    if num_features == 0:
        return []
    nodules = []
    lesion_slices = ndimage.find_objects(labeled_mask)
    voxel_volume = np.prod(resampled_itk.GetSpacing())
    for i, sl in enumerate(lesion_slices):
        lesion_id = i + 1
        volume_voxels = np.sum(labeled_mask == lesion_id)
        volume_mm3 = volume_voxels * voxel_volume
        
        # --- 关键修改：基于体积估算平均直径，并使用直径作为阈值 ---
        # V = 4/3 * pi * r^3  =>  r = (3V / 4pi)^(1/3)
        # d = 2r
        radius_mm = ( (3 * volume_mm3) / (4 * math.pi) )**(1/3)
        diameter_mm = radius_mm * 2

        if diameter_mm < 3:
            continue
            
        # 关键修正：使用CT的HU值作为权重计算重心，而不是用掩码
        ct_intensity = ct_array[sl]
        center_voxel = ndimage.center_of_mass(ct_intensity, labeled_mask[sl], lesion_id)
        
        # 将局部坐标转换回全局坐标
        center_voxel = np.array(center_voxel) + np.array([s.start for s in sl])

        center_physical = resampled_itk.TransformContinuousIndexToPhysicalPoint(center_voxel[::-1])
        patch_slices = [
            slice(max(0, s.start - 5), min(d, s.stop + 5))
            for s, d in zip(sl, mask_array.shape)
        ]
        nodules.append({
            'id': lesion_id,
            'diameter_mm': round(diameter_mm, 2),
            'center_physical': [round(c, 2) for c in center_physical],
            'volume_mm3': round(volume_mm3, 2),
            'patch_slices': tuple(patch_slices),
        })
    return nodules

def analyze_ct_scan(image_path):
    if not all([seg_model, nodule_cls_model, tumor_cls_model]):
        return {"status": "error", "message": "一个或多个AI模型未能成功加载，无法执行分析。"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        itk_image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
        resampled_itk = resample_image(itk_image)
        ct_array = sitk.GetArrayFromImage(resampled_itk)

        # 裁剪HU值以备后用
        ct_array_clipped = np.clip(ct_array, -1000, 1000)

        # 分割模型输入不再进行归一化，直接使用裁剪后的HU值
        full_mask = np.zeros_like(ct_array, dtype=np.float32)
        context_slices = 3
        for slice_idx in range(ct_array.shape[0]):
            slice_chunk = np.zeros((context_slices * 2 + 1, ct_array.shape[1], ct_array.shape[2]), dtype=np.float32)
            for i, context_idx in enumerate(range(slice_idx - context_slices, slice_idx + context_slices + 1)):
                safe_idx = max(0, min(context_idx, ct_array.shape[0] - 1))
                slice_chunk[i] = ct_array_clipped[safe_idx]
            slice_tensor = torch.from_numpy(slice_chunk).unsqueeze(0).to(device)
            with torch.no_grad():
                mask_tensor = seg_model(slice_tensor)
            
            output_mask_np = mask_tensor.cpu().numpy().squeeze()

            # --- FIX: Pad the output mask to match the input slice shape ---
            input_shape = slice_tensor.shape[2:]
            output_shape = output_mask_np.shape
            
            pad_h = input_shape[0] - output_shape[0]
            pad_w = input_shape[1] - output_shape[1]
            
            if pad_h >= 0 and pad_w >= 0:
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                
                padded_mask = np.pad(
                    output_mask_np, 
                    ((pad_top, pad_bottom), (pad_left, pad_right)), 
                    mode='constant', 
                    constant_values=0
                )
                full_mask[slice_idx] = padded_mask
            else:
                # Fallback in case the output is larger, though unlikely with this UNet
                # We simply crop the output to fit
                h_start = (output_shape[0] - input_shape[0]) // 2
                w_start = (output_shape[1] - input_shape[1]) // 2
                full_mask[slice_idx] = output_mask_np[
                    h_start : h_start + input_shape[0],
                    w_start : w_start + input_shape[1]
                ]

        # --- 后处理：应用阈值并进行形态学腐蚀以去除噪声 ---
        mask_a = (full_mask > 0.4) # 将阈值调整回一个更平衡的水平
        mask_a = ndimage.binary_erosion(mask_a, iterations=1) # 恢复腐蚀操作
        binary_mask = mask_a.astype(np.uint8)

        nodules = find_nodules(binary_mask, ct_array_clipped, resampled_itk)
        
        # 即使没有找到结节，我们依然希望能够展示原始CT图像，所以返回基础数据
        if not nodules:
            return {
                "status": "success", 
                "nodules_found": 0, 
                "nodules": [],
                "resampled_ct_array": ct_array_clipped,
                "segmentation_mask": binary_mask,
                "raw_segmentation_mask": full_mask, # 增加返回原始概率图
                "resampled_itk_image": resampled_itk
            }

        analysis_results = []

        for nodule in nodules:
            patch_array = ct_array_clipped[nodule['patch_slices']]
            patch_itk = sitk.GetImageFromArray(patch_array)
            patch_itk.SetSpacing(resampled_itk.GetSpacing())

            # --- 缩放到分类模型需要的尺寸 (32, 48, 48) ---
            output_size = [32, 48, 48]
            reference_image = sitk.Image(output_size, patch_itk.GetPixelIDValue())
            reference_image.SetOrigin(patch_itk.GetOrigin())
            reference_image.SetDirection(patch_itk.GetDirection())
            reference_image.SetSpacing([
                patch_itk.GetSpacing()[i] * (patch_itk.GetSize()[i] / output_size[i])
                for i in range(3)
            ])

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(reference_image)
            resampler.SetInterpolator(sitk.sitkLinear)

            resized_patch_itk = resampler.Execute(patch_itk)
            # --- 缩放结束 ---

            resized_patch_array = sitk.GetArrayFromImage(resized_patch_itk)
            patch_tensor = torch.from_numpy(resized_patch_array).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                nodule_logits, nodule_prob_softmax = nodule_cls_model(patch_tensor)
                tumor_logits, tumor_prob_softmax = tumor_cls_model(patch_tensor)

            nodule_prob = nodule_prob_softmax[0, 1].item()
            tumor_prob = tumor_prob_softmax[0, 1].item()

            analysis_results.append({
                'lesion_id': nodule['id'],
                'diameter_mm': nodule['diameter_mm'],
                'position': nodule['center_physical'],
                'size_mm3': nodule['volume_mm3'],
                'nodule_malignancy_prob': nodule_prob,
                'tumor_malignancy_prob': tumor_prob,
            })

        return {
            "status": "success",
            "nodules_found": len(analysis_results),
            "nodules": analysis_results,
            "resampled_ct_array": ct_array_clipped,
            "segmentation_mask": binary_mask,
            "raw_segmentation_mask": full_mask, # 增加返回原始概率图
            "resampled_itk_image": resampled_itk
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"分析时出现内部错误: {e}"} 