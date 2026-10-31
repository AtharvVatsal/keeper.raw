"""
Build keeper.raw's "accidental shot" model from OpenAI's CLIP ViT-B/32 (MIT licence).

Produces, in this folder:
  clip_vit_b32_image.onnx  CLIP's image encoder, weights quantised to int8 (~90 MB)
  clip_prompts.json        the text descriptions below, pre-encoded by CLIP's
                           text encoder (so the app never needs the text model)

How the app uses them: a photo's image embedding is compared with every
description; the share of probability that lands on the ACCIDENTAL
descriptions is the photo's "accidental score" (0..1).

Usage (PowerShell, from the repo root) — uses a throwaway virtual environment
so nothing is installed into your system Python:

    python -m venv target\clip-venv
    target\clip-venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    target\clip-venv\Scripts\python -m pip install open_clip_torch onnx onnxruntime
    target\clip-venv\Scripts\python models\convert_clip.py
"""

import json
import os

import torch
import open_clip
from onnxruntime.quantization import QuantType, quantize_dynamic

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FP32_PATH = os.path.join(OUT_DIR, "clip_vit_b32_image.fp32.onnx")
INT8_PATH = os.path.join(OUT_DIR, "clip_vit_b32_image.onnx")
PROMPTS_PATH = os.path.join(OUT_DIR, "clip_prompts.json")

# OpenAI's original weights use QuickGELU; the plain "ViT-B-32" config would
# silently use the wrong activation.
MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"

# =============================================================================
# Descriptions
#
# Generic on purpose — they don't name the objects in any test shoot. They were
# chosen by trying several independent wordings on a real 118-photo test
# folder: combining them was the most robust. In the app (int8 model, photos
# with a face skipped): 8 of 9 accidental shots flagged, 0 real photos.
# =============================================================================
ACCIDENTAL = [
    "an accidental photo taken by mistake",
    "a test shot of random objects at home",
    "a boring snapshot of clutter",
    "a random photo of the floor",
    "a photo of a computer screen",
    "a snapshot of everyday household items",
    "a photo taken by accident",
    "a meaningless picture with no subject",
    "a quick camera test shot",
    "a dull photo of random stuff",
    "an unintentional snapshot",
    "an accidental photo",
    "a camera test shot",
]
DELIBERATE = [
    "a photo of a person",
    "a photo of a group of people",
    "a concert photo",
    "a portrait photo",
    "a photo of an animal",
    "a landscape photo",
    "a photo of a bird",
    "a street photo",
    "a sports photo",
    "an event photo",
    "a photograph of people",
    "a picture of a performer on stage",
    "a photo of a crowd",
    "a close-up portrait",
    "a wildlife photograph",
    "a scenic photo of nature",
    "a photo of a pet",
    "a city photo",
    "an action shot",
    "a wedding photo",
    "a photo of food",
    "a photo of a building",
]


def main():
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.eval()

    # ---- Image encoder → ONNX (fp32), then int8 weights ------------------------
    # Input: 1×3×224×224, RGB, CLIP-normalised. Output: 1×512 embedding.
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model.visual,
        dummy,
        FP32_PATH,
        input_names=["pixels"],
        output_names=["embedding"],
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(FP32_PATH, INT8_PATH, weight_type=QuantType.QInt8, per_channel=True)
    os.remove(FP32_PATH)
    print(f"wrote {INT8_PATH} ({os.path.getsize(INT8_PATH) / 1e6:.0f} MB)")

    # ---- Descriptions → unit-length text embeddings ----------------------------
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    with torch.no_grad():
        text = model.encode_text(tokenizer(ACCIDENTAL + DELIBERATE))
        text = text / text.norm(dim=-1, keepdim=True)
    emb = [[round(v, 6) for v in row] for row in text.tolist()]
    prompts = {
        "model": f"{MODEL_NAME} ({PRETRAINED})",
        # CLIP's learned temperature (100 for the OpenAI weights).
        "logit_scale": round(model.logit_scale.exp().item(), 4),
        "accidental": [
            {"text": t, "embedding": e} for t, e in zip(ACCIDENTAL, emb[: len(ACCIDENTAL)])
        ],
        "deliberate": [
            {"text": t, "embedding": e} for t, e in zip(DELIBERATE, emb[len(ACCIDENTAL) :])
        ],
    }
    with open(PROMPTS_PATH, "w", encoding="utf-8") as f:
        json.dump(prompts, f)
    print(f"wrote {PROMPTS_PATH} ({len(ACCIDENTAL)} accidental, {len(DELIBERATE)} deliberate)")


if __name__ == "__main__":
    main()
