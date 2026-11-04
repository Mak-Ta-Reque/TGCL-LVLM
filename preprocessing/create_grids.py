#!/usr/bin/env python3
"""
Create random image grids by sampling one image from each of N subdirectories.

Example usage:
    python preprocessing/create_grids.py \
        --input_dir "data/colorful/val" \
        --output_dir "data/colorful/val_grids" \
        --n 4 \
        --num_grids 50 \
        --image_size 512 \
        --seed 0
"""

import argparse
import math
import os
import random

from PIL import Image


def create_image_grids(input_dir, output_dir, n=4, num_grids=1, image_size=256):
    """
    Create multiple grid images by randomly sampling 1 image from each
    subdirectory. Output filename encodes the subdirectory class names in grid
    order.
    """
    grid_size = int(math.sqrt(n))
    if grid_size * grid_size != n:
        raise ValueError("n must be a perfect square (4, 16, 36, ...)")

    subdirs = sorted(
        os.path.join(input_dir, d)
        for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    )

    if len(subdirs) < n:
        raise ValueError(
            f"Not enough subdirectories. Found {len(subdirs)}, need {n}."
        )

    os.makedirs(output_dir, exist_ok=True)

    for grid_idx in range(num_grids):
        chosen_subdirs = random.sample(subdirs, n)

        images, class_names = [], []
        for subdir in chosen_subdirs:
            class_name = os.path.basename(subdir)
            class_names.append(class_name)

            files = sorted(
                f
                for f in os.listdir(subdir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
            )
            if not files:
                raise ValueError(f"No images found in {subdir}")

            img_path = os.path.join(subdir, random.choice(files))
            img = (
                Image.open(img_path)
                .convert("RGB")
                .resize((image_size, image_size))
            )
            images.append(img)

        grid_img = Image.new(
            "RGB", (grid_size * image_size, grid_size * image_size)
        )
        for idx, img in enumerate(images):
            row, col = divmod(idx, grid_size)
            grid_img.paste(img, (col * image_size, row * image_size))

        # The class names in the file name are the grid's labels: never truncate
        # them.
        output_name = f"grid_{n}_{grid_idx+1}__{'__'.join(class_names)}.jpg"
        if len(output_name.encode()) > 255:
            raise ValueError(
                f"grid file name longer than 255 bytes: {output_name}"
            )
        output_path = os.path.join(output_dir, output_name)
        grid_img.save(output_path)

        print(f"Saved grid {grid_idx+1}/{num_grids} at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create random image grids.")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Input directory with subdirectories (classes).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where output grids will be saved.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=4,
        help="Number of images per grid (must be a perfect square).",
    )
    parser.add_argument(
        "--num_grids", type=int, default=40, help="How many grids to generate."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Resize each image to this size (square).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed (reproducible grids)",
    )

    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    create_image_grids(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n=args.n,
        num_grids=args.num_grids,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
