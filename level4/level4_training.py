"""
训练脚本
训练直接读 level4_dataset/input 和 level4_dataset/output
不再依赖预先跑 preparation 生成的 data/train。预处理（保持长宽比缩放 + 裁剪
+ train/val 划分）在 DataLoader 里在线完成
  - 每 metric_interval 步记录 loss / PSNR / SSIM（相对 clean 真值）。
  - 每个 epoch 结束、训练结束时保存 training_curves.png 和 metrics.csv。
  - --max-steps 限制总迭代步数。
    python level4_training.py --epochs 20 --batch-size 8 --save model_test.pth
    python level4_training.py --epochs 100 --use-perceptual --use-ssim --metric-interval 20
"""

import argparse
import csv
import math
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from level4_model import UNet
from level4_losses import CombinedLoss, ssim
from level4_preparation import (
    RAW_INPUT, RAW_OUTPUT, find_pairs, _resize_short_side, _center_crop,
)


class PairedDataset(Dataset):
    """在线读原始 input/output，做缩放+裁剪+train/val 划分。"""

    def __init__(self, input_dir, output_dir, size=256, split="train",
                 val_ratio=0.15, seed=0, flip=False):
        self.size = size
        self.split = split
        self.flip = flip
        pairs = find_pairs(input_dir, output_dir)
        rng = random.Random(seed)
        rng.shuffle(pairs)
        n_val = max(1, int(len(pairs) * val_ratio))
        self.pairs = pairs[n_val:] if split == "train" else pairs[:n_val]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp_path, clean_path, _ = self.pairs[idx]
        inp = Image.open(inp_path).convert("L")
        clean = Image.open(clean_path).convert("L")
        if inp.size != clean.size:
            clean = clean.resize(inp.size, Image.BILINEAR)

        inp = _resize_short_side(inp, self.size)
        clean = _resize_short_side(clean, self.size)

        if self.split == "train":
            w, h = inp.size
            x0 = random.randint(0, w - self.size)
            y0 = random.randint(0, h - self.size)
            inp = inp.crop((x0, y0, x0 + self.size, y0 + self.size))
            clean = clean.crop((x0, y0, x0 + self.size, y0 + self.size))
            if self.flip and random.random() < 0.5:
                inp = inp.transpose(Image.FLIP_LEFT_RIGHT)
                clean = clean.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            inp = _center_crop(inp, self.size)
            clean = _center_crop(clean, self.size)

        return self._t(inp), self._t(clean)

    @staticmethod
    def _t(img):
        return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).unsqueeze(0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr(pred, target, max_val=1.0):
    mse = ((pred - target) ** 2).mean().item()
    if mse == 0:
        return 100.0
    return min(100.0, 20 * math.log10(max_val / math.sqrt(mse)))


def evaluate(model, loader, device):
    model.eval()
    ps, ss = [], []
    with torch.no_grad():
        for inp, clean in loader:
            inp, clean = inp.to(device), clean.to(device)
            pred = model(inp)
            ps.append(psnr(pred, clean))
            ss.append(ssim(pred, clean).item())
    model.train()
    return float(np.mean(ps)), float(np.mean(ss))


def save_csv(records, save_dir):
    path = os.path.join(save_dir, "metrics.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss", "psnr", "ssim"])
        for k in range(len(records["step"])):
            w.writerow([records["step"][k], records["loss"][k],
                        records["psnr"][k], records["ssim"][k]])


def plot_curves(records, save_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图（metrics.csv 已保存）。")
        return

    steps = records["step"]
    if len(steps) == 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot(steps, records["loss"], color="tab:red")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[1].plot(steps, records["psnr"], color="tab:blue")
    axes[1].set_title("PSNR (dB)")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("dB")
    axes[2].plot(steps, records["ssim"], color="tab:green")
    axes[2].set_title("SSIM")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("SSIM")
    fig.tight_layout()
    out = os.path.join(save_dir, "training_curves.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"已保存曲线 -> {out}")


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds = PairedDataset(args.input_dir, args.output_dir, size=args.size,
                             split="train", val_ratio=args.val_ratio,
                             seed=args.seed, flip=args.flip)
    val_ds = PairedDataset(args.input_dir, args.output_dir, size=args.size,
                           split="val", val_ratio=args.val_ratio,
                           seed=args.seed, flip=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    model = UNet(in_channals=1, out_channals=1, base_channals=args.base,
                 bilinear=args.bilinear, use_attention=args.use_attention).to(device)
    criterion = CombinedLoss(
        w_recon=args.w_recon,
        w_perceptual=args.w_perceptual,
        w_ssim=args.w_ssim,
        use_perceptual=args.use_perceptual,
        use_ssim=args.use_ssim,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    save_dir = os.path.dirname(os.path.abspath(args.save))
    os.makedirs(save_dir, exist_ok=True)
    best_psnr = -float("inf")
    ckpt = None

    records = {"step": [], "loss": [], "psnr": [], "ssim": []}
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        stop = False

        for inp, clean in train_loader:
            inp, clean = inp.to(device), clean.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                pred = model(inp)
                loss, metrics = criterion(pred, clean)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            running += metrics["total"]

            if global_step % args.metric_interval == 0:
                with torch.no_grad():
                    pred_f = pred.detach().float()
                    p = psnr(pred_f, clean)
                    s = ssim(pred_f, clean).item()
                records["step"].append(global_step)
                records["loss"].append(loss.item())
                records["psnr"].append(p)
                records["ssim"].append(s)
                print(f"[step {global_step}] loss={loss.item():.4f} "
                      f"psnr={p:.2f} ssim={s:.4f}")

            if args.max_steps > 0 and global_step >= args.max_steps:
                stop = True
                break

        scheduler.step()

        val_psnr, val_ssim = evaluate(model, val_loader, device)
        print(f"== Epoch {epoch} | train_loss={running / len(train_loader):.4f} "
              f"| val_psnr={val_psnr:.3f} | val_ssim={val_ssim:.4f}")

        ckpt = {
            "state_dict": model.state_dict(),
            "config": {
                "in_channals": 1, "out_channals": 1, "base_channals": args.base,
                "bilinear": args.bilinear, "use_attention": args.use_attention,
            },
        }
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(ckpt, args.save)
            print(f"  -> saved best -> {args.save} (psnr={best_psnr:.3f})")

        save_csv(records, save_dir)
        plot_curves(records, save_dir)

        if stop:
            break

    torch.save(ckpt, os.path.join(save_dir, "last.pth"))
    save_csv(records, save_dir)
    plot_curves(records, save_dir)
    print("training done.")


def main():
    p = argparse.ArgumentParser(description="U-Net 文档增强 + 去手写训练")
    p.add_argument("--input-dir", default=RAW_INPUT, help="带手写原始图目录")
    p.add_argument("--output-dir", default=RAW_OUTPUT, help="干净对照图目录")
    p.add_argument("--save", default="model.pth")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=0, help="限制总迭代步数，0 表示不限制")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-ratio", type=float, default=0.15, help="验证集比例")
    p.add_argument("--flip", action="store_true", help="训练时随机水平翻转")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--metric-interval", type=int, default=20, help="每多少步记录一次 loss/PSNR/SSIM")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--bilinear", action="store_true")
    p.add_argument("--use-attention", action="store_true")
    p.add_argument("--w-recon", type=float, default=1.0)
    p.add_argument("--w-perceptual", type=float, default=0.1)
    p.add_argument("--w-ssim", type=float, default=0.2)
    p.add_argument("--use-perceptual", action="store_true", help="启用感知损失（需下载 VGG）")
    p.add_argument("--use-ssim", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
