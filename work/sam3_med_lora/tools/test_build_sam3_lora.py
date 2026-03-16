import os
import sys

import torch

PROJECT_ROOT = "/root/autodl-tmp/work/sam3_med_lora"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.build_sam3_lora import build_sam3_lora_model


def main():
    os.environ.pop("OMP_NUM_THREADS", None)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        verbose=True,
    )

    print("\n[OK] model built.")
    print(f"device={device}")
    print(f"num_replaced_modules={extra['num_replaced_modules']}")
    print("first 10 replaced modules:")
    for x in extra["replaced_modules"][:10]:
        print("  ", x)

    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    print(f"\ntrainable tensors: {len(trainable)}")
    for n, c in trainable[:20]:
        print(f"  {n}: {c:,}")


if __name__ == "__main__":
    main()