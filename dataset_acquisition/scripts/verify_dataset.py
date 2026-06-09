#!/usr/bin/env python3
"""
verify_dataset.py
─────────────────
Display N randomly sampled dataset images with OBB annotations drawn on top,
arranged in a grid for quick visual quality inspection.

Does NOT require roscore — pure Python / OpenCV / matplotlib.

Usage
-----
# Show 50 images from all splits (default)
python3 verify_dataset.py

# Show 100 images, 10 columns, train split only
python3 verify_dataset.py --n 100 --cols 10 --split train

# Save grid to a file instead of opening a window
python3 verify_dataset.py --save verify_grid.png

# Custom dataset path
python3 verify_dataset.py --dataset_dir /path/to/dataset
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from annotator import draw_obb


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_pairs(dataset_dir: str, splits: tuple):
    """Return list of (img_path, lbl_path, split_name) for every sample."""
    pairs = []
    for split in splits:
        img_dir = os.path.join(dataset_dir, "images", split)
        lbl_dir = os.path.join(dataset_dir, "labels", split)
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(fname)[0]
            pairs.append((
                os.path.join(img_dir, fname),
                os.path.join(lbl_dir, stem + ".txt"),
                split,
            ))
    return pairs


def annotate(img_path: str, lbl_path: str) -> tuple:
    """
    Load image, draw OBB(s) from label file.
    Returns (bgr_image, has_label).
    """
    img = cv2.imread(img_path)
    if img is None:
        raise IOError(f"Cannot read: {img_path}")
    h, w = img.shape[:2]

    has_label = False
    if os.path.isfile(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    img = draw_obb(img, line, w, h, color=(0, 255, 0), thickness=2)
                    has_label = True

    if not has_label:
        # Red border flags missing / empty label
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 220), 6)

    return img, has_label


def thumbnail(img_bgr: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Resize to (tw, th) and convert BGR → RGB."""
    t = cv2.resize(img_bgr, (tw, th), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(t, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visual QA — display OBB-annotated dataset images in a grid")
    parser.add_argument("--dataset_dir", default="",
                        help="Dataset root (default: ../dataset relative to this script)")
    parser.add_argument("--n",     type=int, default=50, help="Images to display (default 50)")
    parser.add_argument("--cols",  type=int, default=10, help="Grid columns (default 10)")
    parser.add_argument("--split", default="all",
                        choices=["all", "train", "val", "test"],
                        help="Split to sample from (default: all)")
    parser.add_argument("--seed",  type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--save",  default="",
                        help="Save grid to this file instead of displaying")
    args = parser.parse_args()

    # ── Resolve dataset dir ──────────────────────────────────────────────────
    if args.dataset_dir:
        dataset_dir = os.path.realpath(args.dataset_dir)
    else:
        dataset_dir = os.path.realpath(
            os.path.join(os.path.dirname(__file__), "..", "dataset"))

    if not os.path.isdir(dataset_dir):
        sys.exit(f"[verify] Dataset not found: {dataset_dir}")

    # ── Collect & sample pairs ───────────────────────────────────────────────
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    all_pairs = find_pairs(dataset_dir, splits)

    if not all_pairs:
        sys.exit(f"[verify] No images found in {dataset_dir} for splits {splits}")

    if args.seed is not None:
        random.seed(args.seed)

    n = min(args.n, len(all_pairs))
    sample = random.sample(all_pairs, n)

    print(f"[verify] {len(all_pairs)} total images — displaying {n} samples")

    # ── Layout ───────────────────────────────────────────────────────────────
    cols     = min(args.cols, n)
    rows     = (n + cols - 1) // cols
    fig_w    = 20                               # figure width in inches
    thumb_w  = max(60, int(fig_w / cols * 96)) # px at ~96 dpi
    thumb_h  = int(thumb_w * 690 / 924)        # preserve 924×690 aspect ratio
    cell_h   = fig_w * rows / cols * (thumb_h / thumb_w)

    fig, axes = plt.subplots(rows, cols,
                              figsize=(fig_w, cell_h),
                              squeeze=False)
    fig.subplots_adjust(wspace=0.02, hspace=0.10)
    fig.patch.set_facecolor("#111118")

    split_color = {"train": "#4caf50", "val": "#2196f3", "test": "#ff9800"}
    errors, missing_labels = 0, 0

    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx >= n:
            ax.set_facecolor("#111118")
            continue

        img_path, lbl_path, split = sample[idx]
        try:
            img_bgr, has_label = annotate(img_path, lbl_path)
            rgb = thumbnail(img_bgr, thumb_w, thumb_h)
            ax.imshow(rgb)

            if not has_label:
                missing_labels += 1

            # Split colour badge (top-left corner)
            ax.text(0.03, 0.97, split[0].upper(),
                    transform=ax.transAxes, fontsize=5.5,
                    color="white", fontweight="bold", va="top",
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc=split_color[split], alpha=0.85, lw=0))

            # Filename stem (below cell)
            stem = os.path.splitext(os.path.basename(img_path))[0]
            ax.set_title(stem, fontsize=3.5, color="#888888", pad=1.5)

        except Exception as exc:
            ax.set_facecolor("#2a0000")
            ax.text(0.5, 0.5, f"ERR\n{exc}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=4, color="#ff6666", wrap=True)
            errors += 1

    # ── Title & legend ───────────────────────────────────────────────────────
    split_counts = {s: sum(1 for _, _, sp in sample if sp == s) for s in splits}
    count_str = "  |  ".join(f"{s}: {split_counts.get(s, 0)}" for s in splits)
    fig.suptitle(
        f"Dataset QA  ·  {n} / {len(all_pairs)} images  ·  {count_str}\n"
        f"{dataset_dir}",
        fontsize=8, color="#dddddd", y=1.002)

    legend_handles = [
        mpatches.Patch(color=split_color[s], label=s)
        for s in splits if s in split_color
    ]
    legend_handles.append(
        mpatches.Patch(color="#dd2222", label="missing label"))
    leg = fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
                     fontsize=7, framealpha=0.4,
                     facecolor="#111118", edgecolor="none",
                     bbox_to_anchor=(0.5, -0.01))
    for text in leg.get_texts():
        text.set_color("white")

    # ── Output ───────────────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0.02, 1, 1])

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[verify] Grid saved → {args.save}")
    else:
        plt.show()

    # ── Summary ──────────────────────────────────────────────────────────────
    if errors:
        print(f"[verify] WARNING: {errors} image(s) could not be loaded.")
    if missing_labels:
        print(f"[verify] WARNING: {missing_labels} image(s) had no label "
              f"(red border shown).")
    if not errors and not missing_labels:
        print(f"[verify] All {n} images rendered with valid OBB labels. ✓")


if __name__ == "__main__":
    main()
