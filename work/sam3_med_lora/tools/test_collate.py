import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
if str(DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(DATASETS_DIR))

from collate import med_labelname_collate_fn
from med_labelname_dataset import MedLabelNameDataset

def main():
    dataset = MedLabelNameDataset(
        index_jsonl=str(PROJECT_ROOT / "data_index" / "chaos_only" / "train_samples.jsonl"),
        image_size=512,
        normalize=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=med_labelname_collate_fn,
    )

    batch = next(iter(loader))

    print("images:", batch["images"].shape)
    print("masks:", batch["masks"].shape)
    print("prompt_texts:", batch["prompt_texts"][:2])
    print("label_names:", batch["label_names"][:2])
    print("channel_idxs:", batch["channel_idxs"][:2])
    print("datasets:", batch["datasets"][:2])


if __name__ == "__main__":
    main()