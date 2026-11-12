# import os
# import glob
# import random
#
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def rect_to_polar(img: np.ndarray) -> np.ndarray:
#     """
#     将 (H, W) 的矩形极坐标图 (theta × r) 映射成圆形空间图。
#     默认 H = 90, W = 360.
#     """
#     if img.ndim != 2:
#         raise ValueError("rect_to_polar 目前假设输入为灰度图 (H, W)")
#
#     h, w = img.shape          # h=90, w=360
#     R = h                     # 圆的半径
#     out_size = 2 * R          # 输出图大小 (2R, 2R)
#     cx = cy = R               # 圆心
#
#     # 目标图每个像素 (x, y) 的坐标网格
#     y, x = np.indices((out_size, out_size), dtype=np.float32)
#     x_c = x - cx
#     y_c = y - cy
#
#     # 极坐标
#     r = np.sqrt(x_c ** 2 + y_c ** 2)
#     theta = np.arctan2(y_c, x_c)          # [-pi, pi]
#     theta[theta < 0] += 2 * np.pi         # [0, 2pi)
#
#     # 映射到源图坐标 (u,v)
#     # 宽度 w 对应 0~2pi， 高度 h 对应 0~R
#     u = (theta / (2 * np.pi)) * (w - 1)   # 列索引
#     v = (r / R) * (h - 1)                 # 行索引
#
#     # 圆外的区域设置为无效坐标，后面会用常数填充
#     mask = r > R
#     u[mask] = -1
#     v[mask] = -1
#
#     map_x = u.astype(np.float32)
#     map_y = v.astype(np.float32)
#
#     # remap 生成圆形图
#     polar = cv2.remap(
#         img.astype(np.float32),
#         map_x,
#         map_y,
#         interpolation=cv2.INTER_LINEAR,
#         borderMode=cv2.BORDER_CONSTANT,
#         borderValue=0,        # 圆外背景为 0（蓝色）
#     )
#     return polar
#
#
# if __name__ == "__main__":
#     # 1. 随机选一张图片
#     root_dir = r".\datasets\RFID\s23-dataset\spectrum"
#     png_list = sorted(glob.glob(os.path.join(root_dir, "*.png")))
#     assert png_list, f"在 {root_dir} 下没有找到 png 文件"
#
#     img_path = random.choice(png_list)
#     print("Selected:", img_path)
#
#     # 2. 读取为灰度图
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise RuntimeError(f"无法读取图像: {img_path}")
#
#     # 3. 变换为圆形 spatial spectrum
#     polar_img = rect_to_polar(img)
#
#     # 4. 可视化：左边矩形，右边圆形
#     plt.figure(figsize=(8, 4))
#
#     plt.subplot(1, 2, 1)
#     plt.imshow(img, cmap="jet", aspect="auto", origin="lower")
#     plt.title("Rectangular (360×90)")
#     plt.axis("off")
#
#     plt.subplot(1, 2, 2)
#     plt.imshow(polar_img, cmap="jet", origin="lower")
#     plt.title("Circular Spatial Spectrum")
#     plt.axis("off")
#
#     plt.tight_layout()
#     plt.show()
#
#     # 5. 保存圆形图片
#     base_name = os.path.splitext(os.path.basename(img_path))[0]
#     out_path = os.path.join(root_dir, f"{base_name}_polar.png")
#     plt.imsave(out_path, polar_img, cmap="jet", origin="lower")
#     print("Saved circular spectrum to:", out_path)



import os
import glob
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt


def rect_to_polar_with_alpha(img: np.ndarray) -> np.ndarray:
    """
    将 (H, W) 的矩形极坐标图映射成圆形空间图，并输出 RGBA，圆外为透明。
    """
    if img.ndim != 2:
        raise ValueError("rect_to_polar 目前假设输入为灰度图 (H, W)")

    h, w = img.shape
    R = h
    out_size = 2 * R
    cx = cy = R

    # 坐标网格
    y, x = np.indices((out_size, out_size), dtype=np.float32)
    x_c = x - cx
    y_c = y - cy

    # 极坐标
    r = np.sqrt(x_c**2 + y_c**2)
    theta = np.arctan2(y_c, x_c)
    theta[theta < 0] += 2 * np.pi

    # 映射
    u = (theta / (2 * np.pi)) * (w - 1)
    v = (r / R) * (h - 1)

    # 圆外 mask
    mask_outside = r > R
    u[mask_outside] = -1
    v[mask_outside] = -1

    polar = cv2.remap(
        img.astype(np.float32),
        u.astype(np.float32),
        v.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Normalize for colormap
    polar_norm = (polar - polar.min()) / (polar.max() - polar.min() + 1e-8)

    # Apply colormap (jet)
    polar_rgb = cv2.applyColorMap((polar_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    polar_rgb = cv2.cvtColor(polar_rgb, cv2.COLOR_BGR2RGB)

    # Create RGBA image, A=255 inside circle, A=0 outside
    alpha = np.where(mask_outside, 0, 255).astype(np.uint8)
    rgba = np.dstack([polar_rgb, alpha])

    return rgba


if __name__ == "__main__":
    root_dir = r".\datasets\RFID\s23-dataset\spectrum"
    png_list = sorted(glob.glob(os.path.join(root_dir, "*.png")))
    assert png_list, f"在 {root_dir} 下没有找到 png 文件"

    img_path = random.choice(png_list)
    print("Selected:", img_path)

    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"无法读取图像: {img_path}")

    # 生成透明背景圆形图
    polar_rgba = rect_to_polar_with_alpha(img)

    # 保存PNG (带透明通道)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(root_dir, f"{base_name}_polar_transparent.png")

    plt.imsave(out_path, polar_rgba)
    print("Saved transparent circular spectrum:", out_path)
