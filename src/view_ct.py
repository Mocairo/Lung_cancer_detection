import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import vedo
import os


def setup_matplotlib_for_chinese():
    """配置Matplotlib以支持中文字符显示"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        print("Matplotlib中文环境配置成功。")
    except Exception as e:
        print(f"Matplotlib中文环境配置失败，可能需要您手动安装'SimHei'字体。错误: {e}")


def resample_image_to_isotropic(itk_image):
    """
    将SimpleITK图像重采样为各向同性（所有轴的间距相同）。
    """
    print("\n正在对图像进行各向同性重采样...")

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    target_spacing_val = min(original_spacing)
    target_spacing = [target_spacing_val, target_spacing_val, target_spacing_val]

    target_size = [
        int(round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / target_spacing[2])))
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputDirection(itk_image.GetDirection())
    resampler.SetOutputOrigin(itk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resampler.SetInterpolator(sitk.sitkLinear)

    resampled_itk_image = resampler.Execute(itk_image)

    print("重采样完成。")
    return resampled_itk_image


def visualize_ct_scan(mhd_file_path):
    """
    读取、重采样并以多种风格可视化CT扫描。
    """
    if not os.path.exists(mhd_file_path):
        print(f"错误：文件未找到于 '{mhd_file_path}'")
        return

    print(f"正在读取文件: {mhd_file_path}...")
    itk_image = sitk.ReadImage(mhd_file_path)
    resampled_itk_image = resample_image_to_isotropic(itk_image)

    hu_array = sitk.GetArrayFromImage(resampled_itk_image)
    spacing_xyz = resampled_itk_image.GetSpacing()

    print(f"\n重采样后图像尺寸 (Depth, Height, Width): {hu_array.shape}")
    print(f"重采样后体素间距 (Spacing in mm): {spacing_xyz}")

    # 配置中文显示
    setup_matplotlib_for_chinese()

    # *** 这是已优化修改的部分：从中心区域选取切片 ***
    print("\n将逐个显示5张独立的2D切片...")

    total_slices = hu_array.shape[0]

    # 增加一个边距，避免选取到最开始和最末尾的全黑切片
    # 例如，我们只在10%到90%的范围内选取
    margin = int(total_slices * 0.10)
    slice_indices = np.linspace(margin, total_slices - 1 - margin, 5, dtype=int)

    # 循环遍历并为每张切片创建一个独立的窗口
    for i, slice_index in enumerate(slice_indices):
        print(f"显示第 {i + 1}/5 张切片 (索引: {slice_index})... 关闭当前窗口以查看下一张。")
        plt.figure(figsize=(8, 8))  # 为每张图创建一个新的figure
        plt.imshow(hu_array[slice_index, :, :], cmap=plt.cm.gray)
        plt.title(f'重采样后的CT扫描 - 切片索引: {slice_index}')
        plt.axis('off')  # 关闭坐标轴显示
        plt.show()  # 显示当前窗口，程序会在此暂停直到窗口关闭

    flipped_hu_array = np.flip(hu_array, axis=0)
    vol = vedo.Volume(flipped_hu_array, spacing=spacing_xyz)

    print("\n[模式1/3] 正在准备默认体积渲染...")
    print("显示3D体积。拖动旋转，滚轮缩放。按 'q' 键关闭窗口。")
    vedo.show(vol, "3D体积渲染 (默认风格)", axes=1, viewup='y').close()

    print("\n[模式2/3] 正在准备最大密度投影 (MIP) 风格的渲染...")
    mip_vol = vol.clone().mode(1)
    print("显示MIP风格的3D渲染。按 'q' 键关闭窗口。")
    vedo.show(mip_vol, "最大密度投影 (MIP) 渲染", axes=1, viewup='y').close()

    print("\n[模式3/3] 正在准备3D等值面重建...")

    bone_surface = vol.isosurface(200).color('ivory').opacity(0.8)
    lung_surface = vol.isosurface(-500).color('red').opacity(0.3)

    print("显示3D表面模型（骨骼-白色，肺部-红色）。按 'q' 键关闭窗口。")
    vedo.show(bone_surface, lung_surface, "3D等值面重建", axes=1, viewup='y').close()


if __name__ == '__main__':
    path_to_mhd_file = '../rawdata/data/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd'

    if not os.path.exists(path_to_mhd_file):
        print("=" * 60)
        print("请在脚本中修改 'path_to_mhd_file' 变量，使其指向一个有效的.mhd文件。")
        print("=" * 60)
    else:
        visualize_ct_scan(path_to_mhd_file)
