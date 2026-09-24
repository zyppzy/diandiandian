"""
流程验证
    python level4_check.py --num 8 --out-dir preview
    python level4_check.py --num 8 --align --out-dir preview
"""

import argparse
import os
import random

from PIL import Image, ImageDraw, ImageFont

from level4_preparation import RAW_INPUT, RAW_OUTPUT, find_pairs, process_one


def _load_font(size=20):
    for path in (
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _blend(a, b):
    return Image.blend(a.convert("RGB"), b.convert("RGB"), 0.5)


def _hconcat(images, gap=4):
    w = sum(im.width for im in images) + gap * (len(images) - 1)
    h = max(im.height for im in images)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for im in images:
        canvas.paste(im.convert("RGB"), (x, 0))
        x += im.width + gap
    return canvas


def _header(labels, panel_w, gap, font):
    label_h = 34
    strip_w = len(labels) * panel_w + (len(labels) - 1) * gap
    img = Image.new("RGB", (strip_w, label_h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for i, lab in enumerate(labels):
        cx = i * (panel_w + gap) + panel_w // 2
        bbox = d.textbbox((0, 0), lab, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        d.text((cx - tw // 2, (label_h - th) // 2), lab, font=font, fill=(0, 0, 0))
    return img


def main():
    p = argparse.ArgumentParser(description="验证 input/clean 是否对齐")
    p.add_argument("--num", type=int, default=8)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--angle", type=float, default=10.0)
    p.add_argument("--flip", action="store_true")
    p.add_argument("--align", action="store_true")
    p.add_argument("--aug", action="store_true")
    p.add_argument("--input-dir", default=RAW_INPUT)
    p.add_argument("--output-dir", default=RAW_OUTPUT)
    p.add_argument("--out-dir", default="preview")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    pairs = find_pairs(args.input_dir, args.output_dir)
    if not pairs:
        print("未找到匹配图对。")
        return
    random.shuffle(pairs)
    pairs = pairs[: args.num]

    os.makedirs(args.out_dir, exist_ok=True)
    font = _load_font()
    labels = ["input", "clean", "blend(50%)"]
    gap = 4
    strips = []
    skipped = 0

    for inp_path, clean_path, stem in pairs:
        r = process_one(inp_path, clean_path, args.size, args.angle, args.flip,
                        args.aug, args.align)
        if r is None:
            skipped += 1
            continue
        inp, clean = r
        panels = [inp, clean, _blend(inp, clean)]
        strip = _hconcat(panels, gap)
        strip.save(os.path.join(args.out_dir, stem + ".png"))
        strips.append(strip)

    if strips:
        header = _header(labels, args.size, gap, font)
        vgap = 4
        total_h = header.height + sum(s.height for s in strips) + vgap * len(strips)
        sheet = Image.new("RGB", (header.width, total_h), (255, 255, 255))
        sheet.paste(header, (0, 0))
        y = header.height
        for s in strips:
            sheet.paste(s, (0, y))
            y += s.height + vgap
        sheet.save(os.path.join(args.out_dir, "contact_sheet.png"))

    print(f"处理 {len(strips)} 张，跳过 {skipped} 张（配准失败）")
    print(f"结果保存在 {args.out_dir}/（每张单独一份 + contact_sheet.png 总览）")
    print("三列：input | clean | blend；看第三列有没有重影来判断是否对齐")


if __name__ == "__main__":
    main()
