import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import vedo
import os
from pathlib import Path

# 配置 Matplotlib 支持中文
def setup_matplotlib_for_chinese():
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        print("警告: 未找到'SimHei'字体。标题中的中文可能无法正确显示。")

def resample_image_to_isotropic(itk_image):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    target_spacing_val = min(original_spacing)
    target_spacing = [target_spacing_val] * 3
    target_size = [
        int(round(osz * osp / nsp))
        for osz, osp, nsp in zip(original_size, original_spacing, target_spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputDirection(itk_image.GetDirection())
    resampler.SetOutputOrigin(itk_image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(itk_image)

def generate_visualizations(ct_array, mask_array, raw_mask_array, itk_image, base_filename):
    """
    接收预处理好的CT、最终掩码和原始概率掩码，生成并保存带有标注的2D切片和3D视图。
    """
    output_dir = Path('static/reports')
    output_dir.mkdir(parents=True, exist_ok=True)

    filenames = {
        "2d_slice_top": f"{base_filename}_2d_slice_top.png",
        "2d_slice_center": f"{base_filename}_2d_slice_center.png",
        "2d_slice_bottom": f"{base_filename}_2d_slice_bottom.png",
        "3d_default": f"{base_filename}_3d_default.png",
        "3d_mip": f"{base_filename}_3d_mip.png",
        "3d_iso": f"{base_filename}_3d_iso.png",
    }
    
    try:
        hu_array = ct_array # 数组已经由 ai_models 传递过来
        spacing_xyz = itk_image.GetSpacing()

        # --- 1. 生成并保存 3 张带分割掩码的 2D 切片图 ---
        setup_matplotlib_for_chinese()
        total_slices = hu_array.shape[0]
        slice_indices = {
            "top": int(total_slices * 0.25),
            "center": int(total_slices * 0.50),
            "bottom": int(total_slices * 0.75),
        }

        for position, slice_idx in slice_indices.items():
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # 显示CT图像
            ax.imshow(hu_array[slice_idx, :, :], cmap=plt.cm.gray)
            
            # 叠加原始概率热力图
            raw_mask_slice = raw_mask_array[slice_idx, :, :]
            if np.any(raw_mask_slice):
                # 创建一个从透明到红色的颜色图，只在高概率区域显示
                from matplotlib.colors import ListedColormap
                cmap_red = plt.cm.get_cmap('Reds')
                cmap_red._init()
                # 让低概率值接近透明
                alphas = np.linspace(0, 0.6, cmap_red.N + 3)
                cmap_red._lut[:, -1] = alphas
                
                ax.imshow(raw_mask_slice, cmap=cmap_red, alpha=0.5, interpolation='bilinear')


            ax.set_title(f'CT扫描 - {position.capitalize()} 切片 (AI标注)')
            ax.axis('off')
            
            plt.savefig(output_dir / filenames[f"2d_slice_{position}"])
            plt.close(fig)

        # --- 2. 使用 Vedo 生成并保存 3D 视图 ---
        flipped_hu_array = np.flip(hu_array, axis=0)
        vol = vedo.Volume(flipped_hu_array, spacing=spacing_xyz)
        
        # 创建一个离屏的 plotter
        plt_vedo = vedo.Plotter(offscreen=True, size=(800, 800))

        # 保存默认风格
        plt_vedo.add(vol)
        plt_vedo.show(axes=1, viewup='y').screenshot(output_dir / filenames["3d_default"])
        plt_vedo.clear()

        # 保存 MIP 风格
        mip_vol = vol.clone().mode(1)
        plt_vedo.add(mip_vol)
        plt_vedo.show(axes=1, viewup='y').screenshot(output_dir / filenames["3d_mip"])
        plt_vedo.clear()

        # 保存等值面风格
        bone_surface = vol.isosurface(200).color('ivory').opacity(0.8)
        lung_surface = vol.isosurface(-500).color('red').opacity(0.3)
        plt_vedo.add(bone_surface, lung_surface)
        plt_vedo.show(axes=1, viewup='y').screenshot(output_dir / filenames["3d_iso"])
        plt_vedo.close()

        return {"status": "success", "filenames": filenames}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)} 