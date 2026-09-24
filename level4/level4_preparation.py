"""
数据预处理脚本（含输入图与对照图的配准对齐）。

背景：
    你的「干净对照图」已经做过旋转、裁切等预处理，而「带手写输入图」没有，
    两者并非逐像素对齐。若不处理，两图差异会因错位而失真（掩码不可用），
    模型也会学到错误的「像素对像素」映射。

本脚本的做法：
    用 ORB 特征点 + RANSAC 单应矩阵,把输入图配准,warp,到干净图的坐标空间,
    使两者对齐后再生成掩码、增广、划分数据集。

依赖（仅 --align 时必需）：
    pip install opencv-python

用法：
    python preparation.py --input-dir ./input --output-dir ./output \
        --save-dir ./data --size 256 --align --aug

说明：
    - 配准需要两图有足够的印刷文字特征；特征过少的图会跳过并在结束时报告数量。
    - 配准失败或未开启 --align 时，生成的掩码不可信，建议训练时加 --w-mask 0。
"""

import argparse
import os
import random

import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2
except ImportError:
    cv2 = None

IMAGE_EXT = {".png"}


def is_image(name):
    return os.path.splitext(name)[1].lower() in IMAGE_EXT


def find_pairs(input_dir, output_dir):
    """按文件名匹配"""
    inp = {os.path.splitext(f)[0]: f for f in os.listdir(input_dir) if is_image(f)}
    out = {os.path.splitext(f)[0]: f for f in os.listdir(output_dir) if is_image(f)}
    common = sorted(set(inp) & set(out))
    return [
        (os.path.join(input_dir, inp[k]), os.path.join(output_dir, out[k]), k)
        for k in common
    ]


def align_to_reference(src, ref, max_features=5000, ratio=0.75, ransac_reproj=5.0):
    """把 src原始输入图配准到 ref已处理的干净图的坐标空间。

    返回与 ref 同尺寸的灰度图；特征点不足或配准失败返回 None。
    """
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
    # 空白处用白色(255)填充，避免黑边被误当成墨迹
    return cv2.warpPerspective(src, H, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def augment(inp, clean, size, angle, flip, crop_scale):
    """对 input 和 clean 施加完全相同的随机几何变换，保持逐像素对齐。"""
    if angle > 0:
        a = random.uniform(-angle, angle)
        inp = inp.rotate(a, resample=Image.BILINEAR, expand=False)
        clean = clean.rotate(a, resample=Image.BILINEAR, expand=False)

    w, h = inp.size
    cw = max(1, int(w * random.uniform(crop_scale, 1.0)))
    ch = max(1, int(h * random.uniform(crop_scale, 1.0)))
    x0 = random.randint(0, w - cw)
    y0 = random.randint(0, h - ch)
    inp = inp.crop((x0, y0, x0 + cw, y0 + ch))
    clean = clean.crop((x0, y0, x0 + cw, y0 + ch))

    if flip and random.random() < 0.5:
        inp = inp.transpose(Image.FLIP_LEFT_RIGHT)
        clean = clean.transpose(Image.FLIP_LEFT_RIGHT)

    inp = inp.resize((size, size), Image.BILINEAR)
    clean = clean.resize((size, size), Image.BILINEAR)
    return inp, clean


def make_mask(inp, clean, threshold=0.08, dilate=1):
    """由两图灰度差生成手写掩码：差异大 -> 手写像素255，其余 0。"""
    a = np.asarray(inp, dtype=np.float32) / 255.0
    b = np.asarray(clean, dtype=np.float32) / 255.0
    diff = np.abs(a - b)
    diff_img = Image.fromarray((diff * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0))
    diff_arr = np.asarray(diff_img, dtype=np.float32) / 255.0
    mask = (diff_arr > threshold).astype(np.uint8) * 255
    if dilate > 0:
        mask = np.asarray(Image.fromarray(mask).filter(ImageFilter.MaxFilter(dilate * 2 + 1)))
    return mask


def process_one(inp_path, clean_path, size, angle, flip, crop_scale, threshold, dilate, aug, align):
    inp = Image.open(inp_path).convert("L")
    clean = Image.open(clean_path).convert("L")

    if align:
        aligned = align_to_reference(np.asarray(inp), np.asarray(clean))
        if aligned is None:
            return None   # 配准失败，跳过该图
        inp = Image.fromarray(aligned)

    # 对齐后两者尺寸应一致；仍不一致则强制统一
    if inp.size != clean.size:
        clean = clean.resize(inp.size, Image.BILINEAR)

    if aug:
        inp, clean = augment(inp, clean, size, angle, flip, crop_scale)
    else:
        inp = inp.resize((size, size), Image.BILINEAR)
        clean = clean.resize((size, size), Image.BILINEAR)

    mask = make_mask(inp, clean, threshold, dilate)
    return inp, clean, mask


def save_split(pairs, save_dir, split, size, angle, flip, crop_scale, threshold, dilate, aug, align):
    skipped = 0
    for inp_path, clean_path, stem in pairs:
        r = process_one(inp_path, clean_path, size, angle, flip, crop_scale,
                        threshold, dilate, aug, align)
        if r is None:
            skipped += 1
            continue
        inp, clean, mask = r
        for sub, img in (("input", inp), ("clean", clean), ("mask", Image.fromarray(mask))):
            d = os.path.join(save_dir, split, sub)
            os.makedirs(d, exist_ok=True)
            img.save(os.path.join(d, stem + ".png"))
    return skipped


def main():
    p = argparse.ArgumentParser(description="预处理原始数据为训练数据集（可配准对齐）")
    p.add_argument("--input-dir", required=True, help="带手写的原始图目录")
    p.add_argument("--output-dir", required=True, help="干净对照图目录")
    p.add_argument("--save-dir", default="data")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--angle", type=float, default=10.0)
    p.add_argument("--flip", action="store_true")
    p.add_argument("--crop-scale", type=float, default=0.8)
    p.add_argument("--threshold", type=float, default=0.08)
    p.add_argument("--dilate", type=int, default=1)
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
        print("未找到 input/output 中文件名匹配的图对，请检查目录与文件名。")
        return
    print(f"找到 {len(pairs)} 对匹配图片")

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_ratio))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]

    sk1 = save_split(train_pairs, args.save_dir, "train", args.size, args.angle,
                     args.flip, args.crop_scale, args.threshold, args.dilate, args.aug, args.align)
    sk2 = save_split(val_pairs, args.save_dir, "val", args.size, args.angle,
                     args.flip, args.crop_scale, args.threshold, args.dilate, args.aug, args.align)

    print(f"完成：train={len(train_pairs) - sk1}，val={len(val_pairs) - sk2}，"
          f"跳过 {sk1 + sk2} 张（配准失败）")


if __name__ == "__main__":
    main()
