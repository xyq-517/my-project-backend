import os
import numpy as np
from PIL import Image

# 替换为你的数据集路径
label_dir = r'F:\VOCdevkit_lidc\VOC2007\SegmentationClass'

if not os.path.exists(label_dir):
    print(f"路径不存在: {label_dir}")
else:
    files = os.listdir(label_dir)
    if len(files) == 0:
        print("文件夹为空")
    else:
        print(f"正在检查 {label_dir} 中的标签...")
        unique_values_all = set()
        # 检查前10张图片
        for i in range(min(10, len(files))):
            img_path = os.path.join(label_dir, files[i])
            img = Image.open(img_path)
            img_array = np.array(img)
            unique_values = np.unique(img_array)
            unique_values_all.update(unique_values)
            print(f"图片 {files[i]} 的像素值有: {unique_values}")
        
        print("-" * 30)
        print(f"所有检查图片中的总像素值集合: {sorted(list(unique_values_all))}")
        
        if 255 in unique_values_all:
            print("\n警告: 发现像素值 255！这通常意味着标签格式不正确。")
            print("背景应该是 0，目标应该是 1, 2, 3...")
        elif max(unique_values_all) > 1 and len(unique_values_all) == 2:
             print(f"\n警告: 发现像素值 {max(unique_values_all)}，但你可能只需要 0 和 1。")
