import inspect
import json
import os
import sys
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = "/root/autodl-tmp/work/sam3_med_lora"
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

for p in [PROJECT_ROOT, DATASETS_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from collate import med_labelname_collate_fn
from med_labelname_dataset import MedLabelNameDataset
from build_sam3_lora import build_sam3_lora_model


def _to_shape(x: Any):
    if torch.is_tensor(x):
        return {
            "type": "tensor",
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
        }
    if isinstance(x, dict):
        return {k: _to_shape(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return {
            "type": type(x).__name__,
            "len": len(x),
            "items": [_to_shape(v) for v in x[:5]],
        }
    return {"type": str(type(x)), "repr": repr(x)[:200]}


def try_call(model, call_name: str, kwargs: Dict[str, Any]):
    print(f"\n===== try: {call_name} =====")
    print("kwargs keys:", list(kwargs.keys()))
    try:
        out = model(**kwargs)
        print("[OK] call success")
        print(json.dumps(_to_shape(out), indent=2, ensure_ascii=False))
        return True, out
    except Exception as e:
        print("[FAIL]", type(e).__name__, str(e))
        return False, None


def main():
    os.environ.pop("OMP_NUM_THREADS", None)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = MedLabelNameDataset(
        index_jsonl="/root/autodl-tmp/work/sam3_med_lora/data_index/chaos_only/train_samples.jsonl",
        image_size=512,
        normalize=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=med_labelname_collate_fn,
    )
    batch = next(iter(loader))
    images = batch["images"].to(device)
    prompt_texts = batch["prompt_texts"]
    masks = batch["masks"].to(device)

    model, extra = build_sam3_lora_model(
        checkpoint_path="/root/autodl-tmp/models/sam3_base/sam3.pt",
        bpe_path="/root/autodl-tmp/repos/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        device=device,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        train_bias=False,
        train_norm=False,
        load_from_hf=False,
        verbose=False,
    )
    model.train()

    print("===== model type =====")
    print(type(model))

    print("\n===== model.forward signature =====")
    print(inspect.signature(model.forward))

    print("\n===== batch summary =====")
    print("images:", images.shape, images.dtype, images.device)
    print("masks:", masks.shape, masks.dtype, masks.device)
    print("prompt_texts:", prompt_texts[:2])

    candidates = [
        {
            "images": images,
            "text": prompt_texts,
        },
        {
            "images": images,
            "text_prompts": prompt_texts,
        },
        {
            "image": images,
            "text": prompt_texts,
        },
        {
            "img_batch": images,
            "text": prompt_texts,
        },
        {
            "images": images,
            "captions": prompt_texts,
        },
        {
            "images": images,
            "prompt": prompt_texts,
        },
        {
            "images": images,
            "prompt_texts": prompt_texts,
        },
        {
            "images": images,
            "targets": masks,
            "text": prompt_texts,
        },
        {
            "images": images,
            "mask_labels": masks,
            "text": prompt_texts,
        },
    ]

    any_ok = False
    for i, kwargs in enumerate(candidates, start=1):
        ok, _ = try_call(model, f"candidate_{i}", kwargs)
        any_ok = any_ok or ok

    if not any_ok:
        print("\nNo candidate call succeeded.")
        print("Next step: inspect the exact expected batch format based on the thrown error messages.")


if __name__ == "__main__":
    main()