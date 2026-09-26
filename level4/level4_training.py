"""
训练脚本（文档增强 + 去除手写，单输出，直接读原始 input/output）。

【本次新增：对抗损失 GAN】
  加了 PatchGAN 判别器 + LSGAN 对抗损失（pix2pix 风格）：
  - 生成器损失 = 重建 + 感知 + SSIM + w_adv * 对抗损失
  - 判别器损失 = 0.5*(真图判真 + 假图判假) 的 MSE
  判别器能专门揪出「残留墨迹/直线」这类不干净的地方，逼生成器擦得更彻底。
  用 --use-gan 开启，--w-adv 控制对抗损失权重，--lr-d 控制判别器学习率。

【之前已保留】
  - 在线读原始 input/output，保持长宽比缩放 + 裁剪 + train/val 划分。
  - 每 metric_interval 步记录 loss/PSNR/SSIM，保存 training_curves.png 和 metrics.csv。
  - --max-steps 限制总步数，--amp 混合精度。

用法：
    python level4_training.py --epochs 100 --batch-size 16 --amp \
        --use-perceptual --use-ssim --use-gan --w-adv 0.1 --save model.pth
"""

import argparse
import csv
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载轻微截断的图片（数据上传不完整时不崩溃）
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from level4_model import UNet, PatchDiscriminator
from level4_losses import CombinedLoss, ssim
from level4_preparation import (
    RAW_INPUT, RAW_OUTPUT, find_pairs, _center_crop,
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
        selected = pairs[n_val:] if split == "train" else pairs[:n_val]
        # 过滤掉仍无法完整读取的损坏图片，避免训练中途崩溃
        self.pairs = []
        for p in selected:
            try:
                Image.open(p[0]).load()
                Image.open(p[1]).load()
                self.pairs.append(p)
            except Exception:
                continue

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp_path, clean_path, _ = self.pairs[idx]
        inp = Image.open(inp_path).convert("L")
        clean = Image.open(clean_path).convert("L")
        if inp.size != clean.size:
            clean = clean.resize(inp.size, Image.BILINEAR)

        # 【重要修复】直接在原分辨率上随机裁剪，不要再把整页缩成小图。
        # 否则 5712x4284 的整页会被压到短边 256，手写笔画被压没，
        # 模型学的是缩略图，推理却在原图分块，两者尺度对不上。
        w, h = inp.size
        if w < self.size or h < self.size:
            # 极端保护：万一图比 patch 还小，先放大到 size
            inp = inp.resize((self.size, self.size), Image.BILINEAR)
            clean = clean.resize((self.size, self.size), Image.BILINEAR)
            w = h = self.size

        if self.split == "train":
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
    if math.isnan(mse) or math.isinf(mse):
        return 0.0  # 模型输出 NaN/Inf 时返回 0，避免被误判成高 PSNR
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

    # 生成器
    model = UNet(in_channals=1, out_channals=1, base_channals=args.base,
                 bilinear=args.bilinear, use_attention=args.use_attention).to(device)
    criterion = CombinedLoss(
        w_recon=args.w_recon,
        w_perceptual=args.w_perceptual,
        w_ssim=args.w_ssim,
        use_perceptual=args.use_perceptual,
        use_ssim=args.use_ssim,
    ).to(device)
    optimizer_G = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer_G, T_max=args.epochs)
    scaler_G = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # 判别器
    discriminator = None
    optimizer_D = None
    scaler_D = None
    if args.use_gan:
        discriminator = PatchDiscriminator(in_channels=2, base=64).to(device)
        lr_d = args.lr_d if args.lr_d > 0 else args.lr
        optimizer_D = AdamW(discriminator.parameters(), lr=lr_d, weight_decay=args.weight_decay)
        scaler_D = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    save_dir = os.path.dirname(os.path.abspath(args.save))
    os.makedirs(save_dir, exist_ok=True)
    best_psnr = -float("inf")
    ckpt = None

    records = {"step": [], "loss": [], "psnr": [], "ssim": []}
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        gan_active = args.use_gan and epoch > args.gan_warmup
        model.train()
        if discriminator is not None:
            discriminator.train()
        running = 0.0
        stop = False

        for inp, clean in train_loader:
            inp, clean = inp.to(device), clean.to(device)

            # ---------- 生成器更新 ----------
            optimizer_G.zero_grad()
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                pred = model(inp)
                recon_loss, metrics = criterion(pred, clean)
                g_loss = recon_loss
                adv_loss = None
                if gan_active:
                    fake_score = discriminator(inp, pred)
                    adv_loss = F.mse_loss(fake_score, torch.ones_like(fake_score))
                    g_loss = g_loss + args.w_adv * adv_loss
            scaler_G.scale(g_loss).backward()
            scaler_G.unscale_(optimizer_G)
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler_G.step(optimizer_G)
            scaler_G.update()

            # ---------- 判别器更新 ----------
            d_loss = None
            if gan_active:
                optimizer_D.zero_grad()
                with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                    real_score = discriminator(inp, clean)
                    fake_score = discriminator(inp, pred.detach())
                    real_label = 0.9 * torch.ones_like(real_score)  # 标签平滑，防判别器过强
                    fake_label = 0.1 * torch.ones_like(fake_score)
                    d_loss = 0.5 * (F.mse_loss(real_score, real_label) +
                                    F.mse_loss(fake_score, fake_label))
                scaler_D.scale(d_loss).backward()
                scaler_D.unscale_(optimizer_D)
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.clip_grad)
                scaler_D.step(optimizer_D)
                scaler_D.update()

            global_step += 1
            running += metrics["total"]

            if global_step % args.metric_interval == 0:
                with torch.no_grad():
                    pred_f = pred.detach().float()
                    p = psnr(pred_f, clean)
                    s = ssim(pred_f, clean).item()
                records["step"].append(global_step)
                records["loss"].append(metrics["total"])
                records["psnr"].append(p)
                records["ssim"].append(s)
                if gan_active:
                    print(f"[step {global_step}] loss={metrics['total']:.4f} "
                          f"psnr={p:.2f} ssim={s:.4f} "
                          f"adv={adv_loss.item():.4f} d={d_loss.item():.4f}")
                else:
                    print(f"[step {global_step}] loss={metrics['total']:.4f} "
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
    p = argparse.ArgumentParser(description="U-Net")
    p.add_argument("--input-dir", default=RAW_INPUT, help="带手写原始图目录")
    p.add_argument("--output-dir", default=RAW_OUTPUT, help="干净对照图目录")
    p.add_argument("--save", default="model.pth")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=0, help="限制总迭代步数，0 表示不限制")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--flip", action="store_true")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--metric-interval", type=int, default=20)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--bilinear", action="store_true")
    p.add_argument("--use-attention", action="store_true")
    p.add_argument("--w-recon", type=float, default=1.0)
    p.add_argument("--w-perceptual", type=float, default=0.1)
    p.add_argument("--w-ssim", type=float, default=0.2)
    p.add_argument("--use-perceptual", action="store_true", help="启用感知损失（需下载 VGG）")
    p.add_argument("--use-ssim", action="store_true")
    # 对抗损失
    p.add_argument("--use-gan", action="store_true", help="启用 PatchGAN 对抗损失")
    p.add_argument("--w-adv", type=float, default=0.1, help="对抗损失权重")
    p.add_argument("--lr-d", type=float, default=0.0, help="判别器学习率，0 表示与生成器一致")
    p.add_argument("--clip-grad", type=float, default=5.0, help="梯度裁剪阈值，0 表示不裁剪")
    p.add_argument("--gan-warmup", type=int, default=0, help="前 N 个 epoch 只训重建、不启用对抗（稳定 GAN）")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
