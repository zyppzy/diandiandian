"""
    重建(Charbonnier) + 感知(VGG) + SSIM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def charbonnier_loss(pred, target, eps=1e-6):
    """像素重建损失（L1 的平滑近似，抗模糊、保边缘）。"""
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


def _gaussian_1d(window_size, sigma):
    coords = torch.arange(window_size, dtype=torch.float32)
    g = torch.exp(-((coords - window_size // 2) ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _create_window(window_size, channels):
    w1 = _gaussian_1d(window_size, 1.5)
    w2 = (w1.unsqueeze(1) @ w1.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    return w2.expand(channels, 1, window_size, window_size).contiguous()


def ssim(img1, img2, window_size=11, size_average=True):
    """可微 SSIM：按局部窗口计算，把注意力放在有结构的文字区域。"""
    channels = img1.size(1)
    window = _create_window(window_size, channels).to(img1.device).type_as(img1)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean() if size_average else ssim_map.mean(dim=(1, 2, 3))


def ssim_loss(pred, target):
    return 1.0 - ssim(pred, target)


class VGGPerceptualLoss(nn.Module):
    """基于预训练 VGG16 的感知损失（内容损失），保文字结构。"""

    def __init__(self, layers=(3, 8, 15)):
        super().__init__()
        try:
            vgg = torchvision.models.vgg16(
                weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
            )
        except Exception:
            vgg = torchvision.models.vgg16(pretrained=True)
        self.features = vgg.features[: max(layers) + 1].eval()
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.layers = layers
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, pred, target):
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = 0.0
        x, y = pred, target
        for i, layer in enumerate(self.features):
            x = layer(x)
            y = layer(y)
            if i in self.layers:
                loss = loss + F.l1_loss(x, y)
        return loss / len(self.layers)


class CombinedLoss(nn.Module):
    """组合损失：重建(Charbonnier) + 感知 + SSIM。"""

    def __init__(self, w_recon=1.0, w_perceptual=0.1, w_ssim=0.2,
                 use_perceptual=True, use_ssim=True):
        super().__init__()
        self.w_recon = w_recon
        self.w_perceptual = w_perceptual
        self.w_ssim = w_ssim
        self.use_perceptual = use_perceptual
        self.use_ssim = use_ssim
        self.perceptual = VGGPerceptualLoss() if use_perceptual else None

    def forward(self, pred, clean_gt):
        recon = charbonnier_loss(pred, clean_gt)
        loss = self.w_recon * recon

        if self.use_perceptual and self.perceptual is not None:
            loss = loss + self.w_perceptual * self.perceptual(pred, clean_gt)

        if self.use_ssim:
            loss = loss + self.w_ssim * ssim_loss(pred, clean_gt)

        return loss, {"total": loss.item(), "recon": recon.item()}
