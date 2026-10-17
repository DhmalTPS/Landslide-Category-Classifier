"""Fine-tune a pretrained CNN (default ConvNeXt-Tiny) end to end.

Use once there are roughly >150 labeled images per class, or when a dataset is provided
(Scenario A). With less data, probe.py is usually as good or better; compare both in eval.py.

- Pretrained ImageNet weights (timm); full fine-tuning with AdamW + cosine schedule + warmup.
- Augmentation: phone-style (src/phone_aug.py: framing, small tilt, exposure, blur, noise,
  JPEG) + horizontal flip. No vertical flip or large rotations: "downhill looking up"
  orientation carries information.
- Class-balanced sampling; label smoothing; mixed precision (fits easily in 8 GB).
- Best epoch chosen by val macro-F1; predicts the test split with flip TTA, both on the
  original photos and on fixed phone-degraded copies (same copies as probe.py).

Usage:  python src/train_cnn.py [--model convnext_tiny.fb_in22k_ft_in1k --epochs 30 --size 288]
Output: models/cnn_<model>.pt, results/pred_cnn.csv, results/pred_cnn_phonesim.csv,
        results/train_log_cnn.csv
"""
import common  # noqa: F401  (sets HF cache dir; must precede transformers/timm)
import argparse
import math
import random
import time
import zlib

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from common import CLASS_NAMES, DATA, MODELS, RESULTS, labeled_dataset, write_csv
from phone_aug import PhoneLike, phone_like


class ImageSet(Dataset):
    def __init__(self, df, tf, phone_copy=0):
        self.files = [DATA / f for f in df.file]
        self.ids = list(df.image_id)
        self.y = [CLASS_NAMES.index(l) for l in df.label]
        self.tf, self.phone_copy = tf, phone_copy

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        if self.phone_copy:   # fixed phone-degraded copy, seeded like probe.variant()
            img = phone_like(img, random.Random(zlib.crc32(f"{self.ids[i]}#{self.phone_copy}".encode())))
        return self.tf(img), self.y[i]


def transforms(cfg, size):
    norm = T.Normalize(cfg["mean"], cfg["std"])
    # PhoneLike crops, tilts, blurs and recompresses like a field phone photo (src/phone_aug.py)
    train = T.Compose([PhoneLike(), T.Resize(int(size * 1.14)), T.RandomCrop(size),
                       T.ColorJitter(0.1, 0.1, 0.1, 0.02), T.ToTensor(), norm])
    test = T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size), T.ToTensor(), norm])
    return train, test


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    probs = []
    for x, _ in loader:
        x = x.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            p = model(x).softmax(-1) + model(x.flip(-1)).softmax(-1)
        probs.append((p / 2).float().cpu())
    return torch.cat(probs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--size", type=int, default=288)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from the last saved epoch")
    ap.add_argument("--max-minutes", type=float, help="stop cleanly after this long (resume later)")
    ap.add_argument("--predict-only", action="store_true", help="only write test predictions of the saved model")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = labeled_dataset()
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}")
    if va.empty:
        raise SystemExit("val split has no labeled images yet")

    model = timm.create_model(args.model, pretrained=True, num_classes=len(CLASS_NAMES)).to(device)
    train_tf, test_tf = transforms(timm.data.resolve_data_config({}, model=model), args.size)
    counts = np.bincount([CLASS_NAMES.index(l) for l in tr.label], minlength=len(CLASS_NAMES))
    weights = [1.0 / counts[CLASS_NAMES.index(l)] for l in tr.label]
    train_dl = DataLoader(ImageSet(tr, train_tf), batch_size=args.batch, num_workers=args.workers,
                          sampler=WeightedRandomSampler(weights, len(tr), replacement=True),
                          drop_last=len(tr) > args.batch, persistent_workers=args.workers > 0)
    val_dl = DataLoader(ImageSet(va, test_tf), batch_size=args.batch, num_workers=args.workers)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * len(train_dl)
    warm = min(len(train_dl) * 2, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    ckpt = MODELS / f"cnn_{args.model.replace('/', '_')}.pt"
    state_path = MODELS / f"cnn_{args.model.replace('/', '_')}.state.pt"   # full training state for --resume
    best_path = MODELS / f"cnn_{args.model.replace('/', '_')}.best.pt"     # best so far; becomes ckpt at the end
    MODELS.mkdir(exist_ok=True)
    best, log, start = -1.0, [], 1
    if args.resume and not state_path.exists():
        print(f"note: --resume but no {state_path.name}; starting from epoch 1")
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        best, log, start = st["best"], st["log"], st["epoch"] + 1
        torch.set_rng_state(st["rng"].cpu())
        print(f"resumed after epoch {st['epoch']} (best val macro-F1 {best:.3f})")
    t0 = time.time()
    if args.predict_only:
        start = args.epochs + 1
    for epoch in range(start, args.epochs + 1):
        if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
            print(f"time budget reached; continue with --resume")
            return
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(y)
        val_proba = predict(model, val_dl, device)
        val_pred = val_proba.argmax(1)
        f1 = f1_score(val_dl.dataset.y, val_pred, average="macro")
        acc = float(np.mean(val_pred == np.array(val_dl.dataset.y)))
        log.append({"epoch": epoch, "train_loss": total / len(train_dl.dataset), "val_acc": acc, "val_macro_f1": f1})
        print(f"epoch {epoch:3d}  loss {log[-1]['train_loss']:.3f}  val acc {acc:.3f}  macro-F1 {f1:.3f}")
        if f1 > best:
            best = f1
            torch.save({"model": args.model, "size": args.size, "classes": CLASS_NAMES, "epoch": epoch,
                        "val_macro_f1": f1, "train_ids": sorted(tr.image_id), "state_dict": model.state_dict()},
                       best_path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "best": best, "log": log, "epoch": epoch, "rng": torch.get_rng_state()}, state_path)
    if not args.predict_only:
        write_csv(pd.DataFrame(log), RESULTS / "train_log_cnn.csv")
        if best_path.exists():
            best_path.replace(ckpt)   # only a finished run replaces the model the app can load
        state_path.unlink(missing_ok=True)
        print(f"best val macro-F1 {best:.3f} -> {ckpt}")

    if not ckpt.exists():
        raise SystemExit(f"{ckpt} not found; train first (src/train_cnn.py)")
    if len(te):
        model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
        for name, k in (("cnn", 0), ("cnn_phonesim", 1)):
            dl = DataLoader(ImageSet(te, test_tf, phone_copy=k), batch_size=args.batch, num_workers=args.workers)
            proba = predict(model, dl, device)
            out = pd.DataFrame({"image_id": te.image_id.values, "pred": [CLASS_NAMES[i] for i in proba.argmax(1)],
                                "conf": proba.max(1).round(4),
                                **{f"p_{c}": proba[:, i].round(4) for i, c in enumerate(CLASS_NAMES)}})
            write_csv(out, RESULTS / f"pred_{name}.csv")
        print(f"test predictions -> {RESULTS}/pred_cnn*.csv; score with: python src/eval.py")


if __name__ == "__main__":
    main()
