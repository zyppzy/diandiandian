"""
加载 .pth 模型
    python level4_predict.py --ckpt model.pth --image scan.jpg --out-dir result
    python level4_predict.py --ckpt model.pth --input-dir scans --out-dir result --deskew
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from level4_model import UNet

try:
    import cv2
except ImportError:
    cv2 = None

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state, cfg = ckpt["state_dict"], ckpt.get("config", {})
    else:
        state, cfg = ckpt, {}
    model = UNet(
        in_channals=cfg.get("in_channals", 1),
        out_channals=cfg.get("out_channals", 1),
        base_channals=cfg.get("base_channals", 64),
        bilinear=cfg.get("bilinear", True),
        use_attention=cfg.get("use_attention", True),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def deskew(gray):
    """基于文字内容外接矩形做粗略纠偏（去旋转）。"""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 100:
        return gray, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) > 20:
        angle = 0.0
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return rotated, angle


def predict_sliding(model, img, device, tile=256, overlap=32):
    """滑动窗口推理，支持任意大图；重叠区取平均消除拼缝。"""
    stride = tile - overlap
    H, W = img.shape
    Hp = tile if H <= tile else ((H - tile) // stride + 1) * stride + tile
    Wp = tile if W <= tile else ((W - tile) // stride + 1) * stride + tile
    img_pad = np.pad(img, ((0, Hp - H), (0, Wp - W)), mode="reflect")

    clean_sum = np.zeros((Hp, Wp), dtype=np.float32)
    cnt = np.zeros((Hp, Wp), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, Hp - tile + 1, stride):
            for x in range(0, Wp - tile + 1, stride):
                t = img_pad[y:y + tile, x:x + tile]
                t = torch.from_numpy(t).unsqueeze(0).unsqueeze(0).to(device)
                clean_sum[y:y + tile, x:x + tile] += model(t)[:, 0:1].squeeze().cpu().numpy()
                cnt[y:y + tile, x:x + tile] += 1.0

    return (clean_sum / cnt)[:H, :W]


def process_one(model, path, out_dir, device, tile, overlap, deskew_flag):
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if deskew_flag:
        gray, angle = deskew(gray)
        print(f"deskew {os.path.basename(path)}: {angle:.2f} deg")
    clean = predict_sliding(model, gray.astype(np.float32) / 255.0, device, tile, overlap)
    clean = np.clip(clean, 0.0, 1.0)
    stem = os.path.splitext(os.path.basename(path))[0]
    Image.fromarray((clean * 255).astype(np.uint8)).save(os.path.join(out_dir, stem + "_clean.png"))
    print(f"done: {path}")


def main():
    p = argparse.ArgumentParser(description="文档增强 + 去手写推理")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image", default=None)
    p.add_argument("--input-dir", default=None)
    p.add_argument("--out-dir", default="result")
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--deskew", action="store_true")
    args = p.parse_args()

    if not args.image and not args.input_dir:
        p.error("必须指定 --image 或 --input-dir")
    if args.deskew and cv2 is None:
        p.error("--deskew 需要安装 opencv-python")
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    if args.image:
        process_one(model, args.image, args.out_dir, device, args.tile, args.overlap, args.deskew)
    else:
        files = [f for f in sorted(os.listdir(args.input_dir))
                 if os.path.splitext(f)[1].lower() in IMAGE_EXT]
        for f in files:
            process_one(model, os.path.join(args.input_dir, f), args.out_dir, device,
                        args.tile, args.overlap, args.deskew)
        print(f"批量处理完成 {len(files)} 张 -> {args.out_dir}")


if __name__ == "__main__":
    main()
