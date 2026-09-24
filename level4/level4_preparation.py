"""
数据预处理脚本（含输入图与对照图的配准对齐）。

【本次修改】因为数据没有手写掩码（两图差异法失效），移除了掩码生成，
只输出 input / clean 两个子目录，供训练直接使用。

保留：
  - IMAGE_EXT 支持 .jpg
  - 保持长宽比缩放再裁剪（避免整页图压变形）
  - ORB+RANSAC 配准（--align，需要 opencv）
  - train/val 划分

用法：
    python level4_preparation.py --size 256 --aug
    python level4_preparation.py --size 256 --align --aug   # 若 input/clean 存在旋转错位
"""

import argparse
import os
import random

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

# ===== 数据路径 =====
# level4（代码）与 level4_dataset（数据）同级，自动推导；也可用命令行参数覆盖。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, "level4_dataset")
RAW_INPUT = os.path.join(DATA_ROOT, "input")
RAW_OUTPUT = os.path.join(DATA_ROOT, "output")
PROCESSED = os.path.join(DATA_ROOT, "data")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_image(name):
    return os.path.splitext(name)[1].lower() in IMAGE_EXT


def find_pairs(input_dir, output_dir):
    inp = {os.path.splitext(f)[0]: f for f in os.listdir(input_dir) if is_image(f)}
    out = {os.path.splitext(f)[0]: f for f in os.listdir(output_dir) if is_image(f)}
    common = sorted(set(inp) & set(out))
    return [
        (os.path.join(input_dir, inp[k]), os.path.join(output_dir, out[k]), k)
        for k in common
    ]


def align_to_reference(src, ref, max_features=5000, ratio=0.75, ransac_reproj=5.0):
    """把 src（原始输入图）配准到 ref（已处理的干净图）的坐标空间。失败返回 None。"""
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(src, None)
    kp2, des2 = orb.detectAndCompute(ref, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    if len(good) < 10:
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj)
    if H is None or (inliers is not None and int(inliers.sum()) < 8):
        return None

    h, w = ref.shape
    return cv2.warpPerspective(src, H, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def _resize_short_side(img, size):
    w, h = img.size
    if w < h:
        nw, nh = size, max(size, int(round(h * size / w)))
    else:
        nh, nw = size, max(size, int(round(w * size / h)))
    return img.resize((nw, nh), Image.BILINEAR)


def _center_crop(img, size):
    w, h = img.size
    x0 = (w - size) // 2
    y0 = (h - size) // 2
    return img.crop((x0, y0, x0 + size, y0 + size))


def augment(inp, clean, size, angle, flip):
    if angle > 0:
        a = random.uniform(-angle, angle)
        inp = inp.rotate(a, resample=Image.BILINEAR, expand=False)
        clean = clean.rotate(a, resample=Image.BILINEAR, expand=False)

    w, h = inp.size
    x0 = random.randint(0, w - size)
    y0 = random.randint(0, h - size)
    inp = inp.crop((x0, y0, x0 + size, y0 + size))
    clean = clean.crop((x0, y0, x0 + size, y0 + size))

    if flip and random.random() < 0.5:
        inp = inp.transpose(Image.FLIP_LEFT_RIGHT)
        clean = clean.transpose(Image.FLIP_LEFT_RIGHT)
    return inp, clean


def process_one(inp_path, clean_path, size, angle, flip, aug, align):
    inp = Image.open(inp_path).convert("L")
    clean = Image.open(clean_path).convert("L")

    if align:
        aligned = align_to_reference(np.asarray(inp), np.asarray(clean))
        if aligned is None:
            return None
        inp = Image.fromarray(aligned)

    if inp.size != clean.size:
        clean = clean.resize(inp.size, Image.BILINEAR)

    inp = _resize_short_side(inp, size)
    clean = _resize_short_side(clean, size)

    if aug:
        inp, clean = augment(inp, clean, size, angle, flip)
    else:
        inp = _center_crop(inp, size)
        clean = _center_crop(clean, size)

    return inp, clean


def save_split(pairs, save_dir, split, size, angle, flip, aug, align):
    skipped = 0
    for inp_path, clean_path, stem in pairs:
        r = process_one(inp_path, clean_path, size, angle, flip, aug, align)
        if r is None:
            skipped += 1
            continue
        inp, clean = r
        for sub, img in (("input", inp), ("clean", clean)):
            d = os.path.join(save_dir, split, sub)
            os.makedirs(d, exist_ok=True)
            img.save(os.path.join(d, stem + ".png"))
    return skipped


def main():
    p = argparse.ArgumentParser(description="预处理原始数据为训练数据集（可配准对齐）")
    p.add_argument("--input-dir", default=RAW_INPUT)
    p.add_argument("--output-dir", default=RAW_OUTPUT)
    p.add_argument("--save-dir", default=PROCESSED)
    p.add_argument("--size", type=int, default=256, help="训练图边长（保持长宽比缩放后裁剪）")
    p.add_argument("--angle", type=float, default=10.0)
    p.add_argument("--flip", action="store_true")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--aug", action="store_true")
    p.add_argument("--align", action="store_true", help="配准输入图到干净图（需要 opencv）")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.align and cv2 is None:
        print("错误：--align 需要安装 opencv-python（pip install opencv-python）。")
        return

    random.seed(args.seed)
    pairs = find_pairs(args.input_dir, args.output_dir)
    if not pairs:
        print("未找到 input/output 中文件名匹配的图对。")
        return
    print(f"找到 {len(pairs)} 对匹配图片")

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_ratio))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]

    sk1 = save_split(train_pairs, args.save_dir, "train", args.size, args.angle,
                     args.flip, args.aug, args.align)
    sk2 = save_split(val_pairs, args.save_dir, "val", args.size, args.angle,
                     args.flip, args.aug, args.align)

    print(f"完成：train={len(train_pairs) - sk1}，val={len(val_pairs) - sk2}，"
          f"跳过 {sk1 + sk2} 张（配准失败）")


if __name__ == "__main__":
    main()
