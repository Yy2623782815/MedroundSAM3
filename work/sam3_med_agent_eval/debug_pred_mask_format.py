# filename: debug_pred_mask_format.py
"""
Debug script: run ONE agent inference round and print the format of final_outputs["pred_masks"][0].

Goal:
- Confirm pred_masks element encoding (COCO RLE dict? string? etc.)
- This will unblock writing the batch evaluator decode logic.

Run (recommended, explicit PYTHONPATH):
PYTHONPATH=/root/autodl-tmp/work/sam3_med_agent_eval:/root/autodl-tmp/repos/sam3 \
python3 /root/autodl-tmp/work/sam3_med_agent_eval/debug_pred_mask_format.py
"""

import os
import json
import inspect
import torch
from functools import partial

# ----------------------------
# Environment / Torch settings (same spirit as notebook)
# ----------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
torch.inference_mode().__enter__()

# ----------------------------
# Build SAM3 model + processor (copied from your notebook, minimal edits)
# ----------------------------
import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

BPE_PATH = "/root/autodl-tmp/repos/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
CKPT_PATH = "/root/autodl-tmp/models/sam3_base/sam3.pt"

sam3_model = build_sam3_image_model(
    bpe_path=BPE_PATH,
    checkpoint_path=CKPT_PATH,
    load_from_HF=False,  # keep as in your notebook; remove if your build function doesn't accept it
    device="cuda",
    eval_mode=True,
)
processor = Sam3Processor(sam3_model, confidence_threshold=0.5)

# ----------------------------
# Bind WORK agent tool functions (IMPORTANT: use work agent, not sam3.agent)
# ----------------------------
from agent.agent_core import agent_inference
from agent.client_llm import send_generate_request as send_generate_request_orig
from agent.client_sam3 import call_sam_service as call_sam_service_orig

LLM_SERVER_URL = "http://0.0.0.0:8001/v1"
LLM_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
LLM_API_KEY = "DUMMY_API_KEY"  # vLLM usually ignores; can also use "EMPTY"

send_generate_request = partial(
    send_generate_request_orig,
    server_url=LLM_SERVER_URL,
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
)
call_sam_service = partial(
    call_sam_service_orig,
    sam3_processor=processor,
)

print("send_generate_request signature:", inspect.signature(send_generate_request))
print("call_sam_service signature:", inspect.signature(call_sam_service))

# ----------------------------
# Pick a single AMOS2022 test sample image (change if you want)
# ----------------------------
DATA_ROOT = "/root/autodl-tmp/data/SAM3_data/AMOS2022"
IMAGE_REL = "image/x/amos_0551_0.png"  # change to any existing file
IMG_PATH = os.path.join(DATA_ROOT, IMAGE_REL)

if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(f"Image not found: {IMG_PATH}")

# Current question: use a real one (from your JSON example)
CURRENT_QUESTION = (
    "Please segment the most prominent large vascular structure in the center of the image, "
    "which enters the abdominal cavity at the diaphragm and descends along the left side of the spine."
)

# History text: plain text block (FULL style). A is your fixed phrase template.
HISTORY_TEXT = (
    "Turn 1:\n"
    "Q: dummy previous question\n"
    "A: 目标aorta 已被成功分割。\n"
)

OUT_DIR = "/root/autodl-tmp/work/sam3_med_agent_eval/outputs/_debug_one"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Run one agent inference
# ----------------------------
agent_history, final_outputs, rendered = agent_inference(
    img_path=IMG_PATH,
    initial_text_prompt=CURRENT_QUESTION,
    history_text=HISTORY_TEXT,
    debug=False,
    send_generate_request=send_generate_request,
    call_sam_service=call_sam_service,
    output_dir=OUT_DIR,
)

# Save full outputs for inspection
save_json_path = os.path.join(OUT_DIR, "final_outputs_debug.json")
with open(save_json_path, "w") as f:
    json.dump(final_outputs, f, indent=2)
print(f"\nSaved final_outputs to: {save_json_path}")

# ----------------------------
# Print pred_masks[0] format
# ----------------------------
print("\n===== final_outputs keys =====")
print(list(final_outputs.keys()))

pred_masks = final_outputs.get("pred_masks", [])
print("\n===== pred_masks length =====")
print(len(pred_masks))

if not pred_masks:
    print("\nNo pred_masks returned (empty).")
    raise SystemExit(0)

pm0 = pred_masks[0]
print("\n===== pred_masks[0] type =====")
print(type(pm0))

print("\n===== pred_masks[0] preview =====")
if isinstance(pm0, dict):
    print("dict keys:", list(pm0.keys()))
    for k in ["size", "counts", "rle", "shape"]:
        if k in pm0:
            v = pm0[k]
            if isinstance(v, str) and len(v) > 200:
                print(f"{k}: (str len={len(v)}) head={v[:120]!r} ...")
            else:
                print(f"{k}: {v!r}")
elif isinstance(pm0, str):
    print(f"str len={len(pm0)} head={pm0[:200]!r} ...")
else:
    # list/tuple/other: avoid huge prints
    try:
        s = json.dumps(pm0)
        print(f"json len={len(s)} head={s[:300]!r} ...")
    except Exception:
        print(repr(pm0))

print("\nDONE.")