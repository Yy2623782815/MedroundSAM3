# filename: /root/autodl-tmp/work/sam3_med_lora/tools/test_med_dataset.py
import os
import sys
from pathlib import Path
import argparse

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.med_labelname_dataset import MedLabelNameDataset  # noqa: E402


def overlay_mask(image_chw: np.ndarray, mask_hw: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    image = np.transpose(image_chw, (1, 2, 0))
    image = (image * 255.0).clip(0, 255).astype(np.uint8)

    color = np.zeros_like(image)
    color[mask_hw > 0] = (0, 255, 0)

    out = cv2.addWeighted(image, 1.0, color, alpha, 0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_jsonl", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ds = MedLabelNameDataset(
        index_jsonl=args.index_jsonl,
        image_size=args.image_size,
        normalize=True,
    )

    print(f"[DATASET] len={len(ds)}")

    n = min(args.num_samples, len(ds))
    for i in range(n):
        item = ds[i]
        img = item["image"].numpy()
        mask = item["mask"].numpy()

        vis = overlay_mask(img, mask)
        text = f'{item["dataset"]} | {item["label_name"]} | ch={item["channel_idx"]}'
        cv2.putText(
            vis, text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
        )

        out_path = Path(args.output_dir) / f"sample_{i:03d}.png"
        cv2.imwrite(str(out_path), vis)
        print(
            f'[OK] idx={i} dataset={item["dataset"]} '
            f'label={item["label_name"]} ch={item["channel_idx"]} '
            f'mask_nonzero={(mask > 0).sum()}'
        )

    print(f"[OUT] {args.output_dir}")


if __name__ == "__main__":
    main()