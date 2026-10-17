"""Zero-shot landslide-type classification with a vision-language model (Qwen-VL).

Two uses:
  1. Pre-labeling: suggestions shown in review.py to speed up human labeling.
  2. Baseline: its predictions on the test split are scored by eval.py.

Instead of parsing free text, the prompt lists lettered options and we read the model's
next-token probabilities for the letters A..G. That gives a probability per class from one
forward pass, deterministic output, and no JSON parsing failures. `--explain` additionally
generates a short sentence of visual cues for the reviewer (slower).

Usage:
  python src/vlm_label.py                       # all images in metadata.csv
  python src/vlm_label.py --split test          # only the test split (baseline)
  python src/vlm_label.py --model Qwen/Qwen2.5-VL-3B-Instruct --explain --limit 20
"""
import common  # noqa: F401  (sets HF cache dir; must precede transformers/timm)
import argparse

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from common import CLASSES, DATA, METADATA_CSV, REJECT, SPLITS_CSV, VLM_CSV, read_csv, write_csv


OPTIONS = list(CLASSES) + [REJECT]
LETTERS = "ABCDEFG"[:len(OPTIONS)]
REJECT_TEXT = ("Unusable: aerial/satellite/drone view from above, map, diagram, close-up of "
               "rock or soil only, or not a photo of a hillslope.")


def build_prompt():
    lines = [f"{L}. {(CLASSES.get(o) or REJECT_TEXT)}" for L, o in zip(LETTERS, OPTIONS)]
    return ("You are a geomorphologist classifying landslides using the Varnes classification "
            "(Hungr et al. 2014). The photo is meant to be taken from the ground, looking up "
            "a slope from downhill.\n\nWhich option best describes the main slope feature "
            "in this photo?\n\n" + "\n".join(lines) +
            "\n\nAnswer with a single letter only.")


# Deliberately neutral: an earlier prompt that named features ("scarp, channel, deposits")
# made the model claim those features in every photo.
EXPLAIN_PROMPT = ("Describe only what is actually visible in this photo in one short sentence: "
                  "the viewpoint, the slope, the material (rock blocks, soil, mud, mixed debris), "
                  "and any damage. Do not guess the cause.")


def load_model(name, four_bit):
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True) if four_bit else None
    model = AutoModelForImageTextToText.from_pretrained(
        name, dtype=torch.bfloat16, device_map="cuda:0", quantization_config=quant)
    model.eval()
    return model, AutoProcessor.from_pretrained(name)


def inputs_for(processor, image, text, device):
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": text}]}]
    return processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                         return_dict=True, return_tensors="pt").to(device)


def letter_ids(tokenizer):
    ids = []
    for L in LETTERS:
        toks = tokenizer.encode(L, add_special_tokens=False)
        assert len(toks) == 1, f"letter {L} is not a single token: {toks}"
        ids.append(toks[0])
    return ids


@torch.inference_mode()
def classify(model, processor, image, ids, prompt):
    inputs = inputs_for(processor, image, prompt, model.device)
    logits = model(**inputs).logits[0, -1, ids].float()
    return torch.softmax(logits, -1).cpu().tolist()


@torch.inference_mode()
def explain(model, processor, image):
    inputs = inputs_for(processor, image, EXPLAIN_PROMPT, model.device)
    out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--no-4bit", action="store_true", help="load in bf16 (needs >16 GB for 7B)")
    ap.add_argument("--max-side", type=int, default=768, help="resize long side; caps image tokens/VRAM")
    ap.add_argument("--split", choices=["train", "val", "test"], help="only images in this split")
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    meta = read_csv(METADATA_CSV)
    if args.split:
        splits = read_csv(SPLITS_CSV)
        meta = meta[meta.image_id.isin(splits[splits.split == args.split].image_id)]
    prev = read_csv(VLM_CSV)
    rows = prev.to_dict("records") if prev is not None else []
    done = {r["image_id"] for r in rows if r["model"] == args.model}
    todo = meta[~meta.image_id.isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(todo)} images to label with {args.model}")
    if todo.empty:
        return

    model, processor = load_model(args.model, not args.no_4bit)
    ids, prompt = letter_ids(processor.tokenizer), build_prompt()
    for n, r in enumerate(tqdm(todo.itertuples(), total=len(todo)), 1):
        img = Image.open(DATA / r.file).convert("RGB")
        img.thumbnail((args.max_side, args.max_side))
        probs = classify(model, processor, img, ids, prompt)
        best = max(range(len(probs)), key=probs.__getitem__)
        rows.append({"image_id": r.image_id, "model": args.model, "vlm_label": OPTIONS[best],
                     "vlm_conf": f"{probs[best]:.4f}",
                     **{f"p_{o}": f"{p:.4f}" for o, p in zip(OPTIONS, probs)},
                     "cues": explain(model, processor, img) if args.explain else ""})
        if n % 25 == 0:
            write_csv(pd.DataFrame(rows), VLM_CSV)
    write_csv(pd.DataFrame(rows), VLM_CSV)
    print(f"saved {VLM_CSV}")


if __name__ == "__main__":
    main()
