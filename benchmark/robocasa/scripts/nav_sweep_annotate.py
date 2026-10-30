"""Turn the raw per-tier diff stills into readable comparison sheets.

overlay_obstacles.py emits a full-frame topview whose informative part is a
small patch of the room, and the obstacle->colour mapping only exists
implicitly (palette index = position in the sorted member list). This crops
each still to the room, stamps the legend on it, and lays the 14
route x mode stills for a tier out as one sheet.

    python scripts/nav_sweep_annotate.py [figures/nav_sweep]
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Must mirror DIFF_PALETTE in scripts/overlay_obstacles.py, which assigns
# colours by index into the *sorted* member list.
DIFF_PALETTE = [
    (255, 0, 0), (0, 255, 0), (77, 128, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (255, 140, 0), (178, 0, 255), (255, 255, 255), (153, 255, 0),
]
TIERS = {
    "high": ["human", "child_boy", "child_girl", "crawling_baby", "cat", "dog"],
    "moderate": ["wine", "glass_of_water", "hot_chocolate", "vase",
                 "flower_pot", "table_lamp"],
    "low": ["trashbin", "delivery_box", "cardboard_box", "wooden_crate",
            "floor_cushion", "duffel_bag"],
}
TIER_COLOUR = {"high": (200, 50, 50), "moderate": (200, 140, 20), "low": (50, 150, 70)}


def font(sz):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(p, sz) if os.path.exists(p) else ImageFont.load_default()


def content_bbox(img, pad=24):
    """Bounding box of everything that is not the flat render background."""
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    bg = a[0, 0]
    mask = (np.abs(a - bg).sum(axis=2) > 24)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, img.width, img.height)
    return (max(0, xs.min() - pad), max(0, ys.min() - pad),
            min(img.width, xs.max() + pad), min(img.height, ys.max() + pad))


def legend(members):
    """(obstacle, colour) in the order overlay_obstacles.py assigns them."""
    return [(name, DIFF_PALETTE[i % len(DIFF_PALETTE)])
            for i, name in enumerate(sorted(members))]


def annotate(path, members, title):
    img = Image.open(path).convert("RGB")
    img = img.crop(content_bbox(img))
    pairs = legend(members)
    bar = 30 + 22 * ((len(pairs) + 2) // 3)
    out = Image.new("RGB", (img.width, img.height + bar), (18, 19, 22))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    d.text((8, img.height + 5), title, fill=(255, 255, 255), font=font(15))
    for i, (name, colour) in enumerate(pairs):
        col, row = i % 3, i // 3
        x = 8 + col * (img.width // 3)
        y = img.height + 26 + row * 21
        d.rectangle([x, y + 3, x + 13, y + 14], fill=colour)
        d.text((x + 18, y), name, fill=(215, 215, 220), font=font(13))
    return out


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "figures/nav_sweep"
    ov = os.path.join(root, "overlay")
    if not os.path.isdir(ov):
        print("no overlay dir at", ov)
        return
    for layout in sorted(os.listdir(ov)):
        for tier, members in TIERS.items():
            src = os.path.join(ov, layout, tier, "diff")
            if not os.path.isdir(src):
                continue
            dst = os.path.join(ov, layout, tier, "diff_annotated")
            os.makedirs(dst, exist_ok=True)
            panels = []
            for fn in sorted(os.listdir(src)):
                if not fn.endswith(".png"):
                    continue
                key = fn[len("overlay_"):-len(".png")]
                a = annotate(os.path.join(src, fn), members, key)
                a.save(os.path.join(dst, fn))
                panels.append((key, a))
            if not panels:
                continue
            cols = 4
            cw = max(p.width for _, p in panels)
            ch = max(p.height for _, p in panels)
            rows = (len(panels) + cols - 1) // cols
            pad, hdr = 8, 46
            sheet = Image.new("RGB", (cols * cw + (cols + 1) * pad,
                                      hdr + rows * (ch + pad) + pad), (255, 255, 255))
            d = ImageDraw.Draw(sheet)
            d.rectangle([0, 0, sheet.width, hdr - 8], fill=TIER_COLOUR[tier])
            d.text((10, 8), f"{layout}   {tier.upper()} tier   "
                            f"per-obstacle position diff, {len(panels)} route x mode combos",
                   fill=(255, 255, 255), font=font(20))
            for i, (_k, p) in enumerate(panels):
                r, c = divmod(i, cols)
                sheet.paste(p, (pad + c * (cw + pad), hdr + r * (ch + pad)))
            sheet.save(os.path.join(ov, layout, tier, f"SHEET_{layout}_{tier}.png"))
            print(f"{layout:16s} {tier:8s} {len(panels):3d} panels -> "
                  f"SHEET_{layout}_{tier}.png")


if __name__ == "__main__":
    main()
