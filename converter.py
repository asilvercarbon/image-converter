import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SIZE = (172, 320)  
RESAMPLE = Image.Resampling.LANCZOS
BACKGROUND = (0, 0, 0)  # shown behind transparent pixels and pad bars


def load_reduced(path, size):
    with Image.open(path) as img:
        img.draft("RGB", size)  # scaled JPEG decode (no-op for PNG); must precede pixel access
        img = ImageOps.exif_transpose(img)
        if img.has_transparency_data:
            # RGB565 has no alpha: blend onto the background, don't just drop it
            img = img.convert("RGBA")
            bg = Image.new("RGBA", img.size, BACKGROUND + (255,))
            img = Image.alpha_composite(bg, img)
        return img.convert("RGB")


def resize(img, size, fit):
    if fit == "crop":
        return ImageOps.fit(img, size, RESAMPLE)
    if fit == "pad":
        return ImageOps.pad(img, size, RESAMPLE, color=BACKGROUND)
    return img.resize(size, RESAMPLE)


def to_rgb565(img, endian):
    # uint16 before shifting, or r << 11 overflows uint8
    rgb = np.asarray(img, dtype=np.uint16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    packed = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
    return packed.astype(endian + "u2").tobytes()


def main():
    parser = argparse.ArgumentParser(description="Convert JPEG/PNG images to raw RGB565.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("out"))
    parser.add_argument("--fit", choices=["crop", "pad", "stretch"], default="crop")
    parser.add_argument("--endian", choices=["big", "little"], default="big")
    args = parser.parse_args()

    endian = ">" if args.endian == "big" else "<"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in args.inputs:
        img = resize(load_reduced(path, SIZE), SIZE, args.fit)
        data = to_rgb565(img, endian)
        out = args.out_dir / (path.stem + ".bin")
        out.write_bytes(data)
        print(f"{path} -> {out} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
