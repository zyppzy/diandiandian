import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

#像素重建损失
def charbonnier_loss(pred,target,eps=1e-6):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))

#掩码损失，识别是否为手写部分
def bce_with_logits_loss(pred_logits,target):
    #二分类交叉熵，输入未过sigmoid的logits,target 可为0/1或[0,1]软标签
    return F.binary_cross_entropy_with_logits(pred_logits, target)

def dice_loss(pred_logits,target,smooth=1.0):
    #惩罚全0预测
    p = torch.sigmoid(pred_logits)
    inter = (p * target).sum(dim=(2, 3))
    union = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * inter + smooth) / (union + smooth)
    return 1.0 - dice.mean()

def focal_loss(pred_logits, target, gamma=2.0, alpha=0.25):
    #给背景降低权重
    p = torch.sigmoid(pred_logits)
    ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)   #每个像素被正确分类的概率
    return (alpha * (1 - p_t) ** gamma * ce).mean()

#SSIM结构相似度
def _gaussian_1d(window_size, sigma):
    coords = torch.arange(window_size, dtype=torch.float32)
    g = torch.exp(-((coords - window_size // 2) ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _create_window(window_size, channels):
    w1 = _gaussian_1d(window_size, 1.5)
    w2 = (w1.unsqueeze(1) @ w1.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    return w2.expand(channels, 1, window_size, window_size).contiguous()

def ssim(img1, img2, window_size=11, size_average=True):
    #可微的 SSIM。按局部窗口计算，把注意力放在有结构的文字区域而非大片白背景
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
#利用VGG计算感知损失
    """layers 对应 VGG16 features 中的 ReLU 层索引：
        3  -> relu1_2边缘、纹理
        8  -> relu2_2笔画组合
        15 -> relu3_3字符级形状
    选偏浅层是为避免深层特征诱导网络幻觉出不存在的文字
    """

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
            p.requires_grad_(False)   # VGG 只当标尺，不参与训练
        self.layers = layers
        # ImageNet 统计量（VGG 训练时的归一化）
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, pred, target):
        # 灰度图复制成 3 通道，适配 VGG 的 RGB 输入
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

#组合损失
class CombinedLoss(nn.Module):
    """把重建、感知、SSIM、掩码损失按权重组合起来。"""

    def __init__(
        self,
        w_recon=1.0,
        w_perceptual=0.1,
        w_ssim=0.2,
        w_mask=0.5,
        use_perceptual=True,
        use_ssim=True,
    ):
        super().__init__()
        self.w_recon = w_recon
        self.w_perceptual = w_perceptual
        self.w_ssim = w_ssim
        self.w_mask = w_mask
        self.use_perceptual = use_perceptual
        self.use_ssim = use_ssim
        self.perceptual = VGGPerceptualLoss() if use_perceptual else None

    def forward(self, pred, clean_gt, mask_gt):
        """
        Args:
            pred:     网络输出 (B, 2, H, W)，[0] 干净图、[1] 掩码 logits
            clean_gt: 干净印刷图真值 (B, 1, H, W)
            mask_gt:  手写掩码真值 (B, 1, H, W)
        """
        clean_pred = pred[:, 0:1]
        mask_pred = pred[:, 1:2]

        # 像素重建
        recon = charbonnier_loss(clean_pred, clean_gt)
        loss = self.w_recon * recon

        # 感知损失保文字结构
        if self.use_perceptual and self.perceptual is not None:
            loss = loss + self.w_perceptual * self.perceptual(clean_pred, clean_gt)

        # SSIM保局部结构
        if self.use_ssim:
            loss = loss + self.w_ssim * ssim_loss(clean_pred, clean_gt)

        # 掩码损失：BCE（稳定逐像素梯度）+ Dice（抗不平衡、防全 0）
        mask = bce_with_logits_loss(mask_pred, mask_gt) + dice_loss(mask_pred, mask_gt)
        loss = loss + self.w_mask * mask

        return loss, {
            "total": loss.item(),
            "recon": recon.item(),
            "mask": mask.item(),
        }