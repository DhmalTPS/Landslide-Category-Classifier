"""Landslide dimensions from one ground-level phone photo, measured after classification.

The approach is hybrid. Deep networks propose (outline, depth, objects, grain instances). Pixel statistics
and robust estimators fuse, check and measure (src/dim_stats.py). No single network sets a number alone.

  1. OUTLINE: three probability sources are fused.
     - CLIPSeg, with class-specific text prompts scored against background prompts;
     - a self-trained outline head on DINOv2 patch features (tools/train_outline_head.py), if trained;
     - SAM, run with 6 jittered prompt sets; their mean is a soft mask that also measures boundary uncertainty.
     A pixel-statistics prior (Excess-Green vegetation, blue/bright/smooth sky) damps non-landslide pixels.
     A graph cut (max-flow) on the fused probability, with colour- AND depth-edge pairwise terms, gives the
     final outline. Variants at probability 0.35 and 0.65 carry the boundary uncertainty forward.
  2. DEPTH: Depth Anything V2 metric (original, mirrored, gamma-jittered), Depth Pro (original, mirrored;
     it also estimates focal length) and ZoeDepth (original, mirrored). Per-pixel inverse-variance fusion:
     a model that is unstable under augmentation at a pixel counts less there. Then a guided filter
     (edge-preserving, guided by the photo) on the log-depth.
  3. SCALE, best first:
     - the distance the user typed;
     - a weighted median of OWLv2 detections of known-height objects (person, car, motorbike, truck,
       bus, door, utility pole, and trees as a weak prior; truncated boxes rejected, 3-MAD outliers removed);
     - field calibration (tools/calibrate_dimensions.py);
     - the depth ensemble alone.
  4. GEOMETRY: back-projection, RANSAC slope plane, across/down-slope axes (gravity from the phone sensor
     when given), 2-98 % extents, area corrected for viewing angle, outline length along the contour.
  5. PER CLASS: measures and volume. Area-volume laws are combined with a reconstructed pre-failure surface
     (thin-plate spline fitted on the intact ring, smoothing chosen by cross-validation), and each source
     is weighted by its uncertainty.
  6. GRAINS: SAM instances at zoom 1 (whole photo) and zoom 2 (upsampled tiles), checked for agreement
     where they overlap. Convex-hull b-axis (occlusion), grid-by-number weighting, truncated log-normal
     MLE with bootstrap CI.
  7. UNCERTAINTY: Monte Carlo (1000 draws) over outline variant, scale error and volume priors. Ranges are
     P5-P95; the size class is given as probabilities.
  8. CHECKS -> TRUST: TTA stability, model agreement, haze cue, texture-frequency slope, plane inlier
     fraction, object-vs-depth agreement, Corominas (1996) mobility residual.
"""
import argparse
import io
import json
from pathlib import Path

import common  # noqa: F401  (sets the HF cache before transformers loads)
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from scipy import ndimage

import dim_stats as S

SEG_MODEL = "CIDAS/clipseg-rd64-refined"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
DEPTH_PRO_MODEL = "apple/DepthPro-hf"
ZOE_MODEL = "Intel/zoedepth-kitti"
SAM_MODEL = "facebook/sam-vit-base"
OWL_MODEL = "google/owlv2-base-patch16-ensemble"
HEAD_BACKBONE = "facebook/dinov2-base"
HEAD_FILE = common.MODELS / "outline_head.joblib"
CALIBRATION = common.MODELS / "dimension_calibration.json"
# global depth-scale correction from OWLv2 objects of tight known height (person, car, motorcycle, bus, door) found in
# evaluation photos (written by tools/dimension_eval.py). Used only when a photo has no typed distance, no object of
# its own and no field calibration. Field data replaces it.
OBJECT_PRIOR = common.MODELS / "dimension_object_prior.json"
DEFAULT_F35 = 26.0                  # mm, 35 mm-equivalent focal length of a typical phone main camera
FOCAL_SD = {"EXIF": 0.02, "Depth Pro estimate": 0.15, "typical phone camera": 0.25}   # log-sd of focal length
DEPTH_SD = 0.25                     # log-sd of zero-shot metric depth (typical AbsRel 0.2-0.3 outdoors)
USE_ZOE = False   # ZoeDepth (KITTI) read 4-6x closer than the other two on 40 landslide photos (median max-pair ratio
                  # 6.25) and against OWLv2 known-size objects (6/6 factors > 1), so it is out of the scale ensemble

BACKGROUND = ["sky", "clouds", "green trees and forest", "grass and vegetation", "asphalt road",
              "a building", "a person", "a car", "water"]
PROMPTS = {
    "fall_topple": ["fallen rocks and boulders", "rock debris and talus", "broken rock blocks on the ground",
                    "freshly exposed rock face"],
    "rotational_slide": ["landslide scar", "slumped earth and displaced soil", "curved landslide scarp",
                         "bare disturbed soil on a slope"],
    "translational_slide": ["landslide scar", "bare exposed soil on a hillside", "sheet of slid soil and debris",
                            "exposed sliding surface on a slope"],
    "debris_flow": ["debris flow track", "mud and boulders in a gully", "debris deposit",
                    "channel filled with rocks and mud"],
    "earth_mud_flow": ["mudflow", "wet mud flowing down a slope", "lobe of wet earth", "muddy landslide deposit"],
}
FAMILY_LABEL = {"flow": "debris_flow", "slide": "translational_slide"}
# known-height objects: (OWLv2 query, height m, log-sd of that height). Trees are a weak prior.
KNOWN_OBJECTS = [("a photo of a person", 1.70, 0.06), ("a photo of a car", 1.50, 0.08),
                 ("a photo of a motorcycle", 1.10, 0.10), ("a photo of a truck", 3.0, 0.15),
                 ("a photo of a bus", 3.2, 0.06), ("a photo of a door", 2.1, 0.06),
                 ("a photo of a utility pole", 9.0, 0.25), ("a photo of a tree", 15.0, 0.5)]
GUZZETTI = (0.074, 1.450, 0.6)      # V = a A^g (Guzzetti et al. 2009); log-sd of the scatter about the law
SIZE_CLASSES = [(500, "extremely small"), (5e3, "very small"), (5e4, "small"), (2.5e5, "medium"),
                (1e6, "large"), (5e6, "very large"), (float("inf"), "extremely large")]   # Fell (1994), m^3
BLOCK_CLASSES = [(0.06, "gravel"), (0.2, "cobble"), (1.0, "small boulder"), (4.0, "large boulder"),
                 (float("inf"), "very large block")]
# deposit thickness priors (median, 95 % interval in m)
THICKNESS = {"debris_flow": (1.0, 0.3, 3.0), "earth_mud_flow": (2.0, 0.5, 5.0), "fall_topple": (0.5, 0.2, 2.0)}


def focal_35mm(src):
    """35 mm-equivalent focal length from EXIF, or None. src: path or bytes."""
    try:
        im = Image.open(io.BytesIO(src) if isinstance(src, bytes) else src)
        v = im.getexif().get_ifd(0x8769).get(0xA405)    # FocalLengthIn35mmFilm
        return float(v) if v and 10 <= float(v) <= 300 else None
    except Exception:
        return None


def _try(name, fn):
    try:
        return fn()
    except Exception as e:          # an optional model missing must not take the measurement down
        print(f"{name} unavailable ({e!r}); continuing without it")
        return None


class Dimensioner:
    """offload: on GPUs under 12 GB the two large depth models (Depth Pro, ZoeDepth) live in CPU RAM and move to the
    GPU only while they run. With the classifier loaded too, all models together exceed 8 GB, and Windows would
    otherwise spill into shared memory (4-5x slower)."""

    def __init__(self, device=None, depth=True, extras=True, offload=None):
        import transformers as T
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dt = torch.float16 if self.device == "cuda" else torch.float32
        if offload is None:
            offload = self.device == "cuda" and torch.cuda.get_device_properties(0).total_memory < 12e9
        self.offload = offload
        self._cpu_copies = {}
        dev = lambda m: m.to(self.device).eval()
        park = lambda m: m.to("cpu" if offload else self.device).eval()
        self.seg_proc = T.CLIPSegProcessor.from_pretrained(SEG_MODEL)
        self.seg = park(T.CLIPSegForImageSegmentation.from_pretrained(SEG_MODEL))
        self.sam = _try("SAM", lambda: park(T.SamModel.from_pretrained(SAM_MODEL)))
        self.sam_proc = T.SamProcessor.from_pretrained(SAM_MODEL) if self.sam else None
        self.da = self.pro = self.zoe = self.owl = self.head = None
        if depth:
            self.da_proc = T.AutoImageProcessor.from_pretrained(DEPTH_MODEL)
            self.da = park(T.AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL, dtype=dt))
            self.pro = _try("Depth Pro", lambda: park(T.DepthProForDepthEstimation.from_pretrained(DEPTH_PRO_MODEL, dtype=dt)))
            self.pro_proc = T.DepthProImageProcessor.from_pretrained(DEPTH_PRO_MODEL) if self.pro else None
            if USE_ZOE:
                self.zoe = _try("ZoeDepth", lambda: park(T.ZoeDepthForDepthEstimation.from_pretrained(ZOE_MODEL)))
                self.zoe_proc = T.AutoImageProcessor.from_pretrained(ZOE_MODEL) if self.zoe else None
        if extras:
            self.owl = _try("OWLv2", lambda: park(T.Owlv2ForObjectDetection.from_pretrained(OWL_MODEL, dtype=dt)))
            self.owl_proc = T.Owlv2Processor.from_pretrained(OWL_MODEL) if self.owl else None
        if HEAD_FILE.exists():
            import joblib
            self.head = joblib.load(HEAD_FILE)
            self.head_bb = park(T.AutoModel.from_pretrained(self.head["backbone"]))
            self.head_proc = T.AutoImageProcessor.from_pretrained(self.head["backbone"])
        self.calib = json.loads(CALIBRATION.read_text()) if CALIBRATION.exists() else {}
        self.obj_prior = json.loads(OBJECT_PRIOR.read_text()) if OBJECT_PRIOR.exists() else {}
        if self.offload:   # adaptive: if all parked models + ~2.6 GB activation headroom fit, keep them resident
            parked = [m for m in (self.seg, self.sam, self.da, self.pro, self.zoe, self.owl,
                                  getattr(self, "head_bb", None)) if m is not None]
            need = sum(p.numel() * p.element_size() for m in parked for p in m.parameters())
            free = torch.cuda.mem_get_info()[0]
            if free > need + 2.6e9:      # measured activation peak ~2.2 GB (Depth Pro at 1536 px) + margin
                for m in parked:
                    m.to(self.device)
                self.offload = False

    def use(self, *models):
        """Context: these models on the GPU now, back to CPU RAM afterwards (offload mode)."""
        dim = self

        class _Ctx:
            def __enter__(self):
                for m in models:
                    if m is not None:
                        dim._gpu(m, True)

            def __exit__(self, *a):
                for m in models:
                    if m is not None:
                        dim._gpu(m, False)
        return _Ctx()

    # ------------------------------------------------------------------ outline sources
    @torch.inference_mode()
    def clip_prob(self, img, label):
        pos, texts = PROMPTS[label], PROMPTS[label] + BACKGROUND
        inp = self.seg_proc(text=texts, images=[img] * len(texts), padding=True, return_tensors="pt").to(self.device)
        prob = self.seg(**inp).logits.float().softmax(0)[:len(pos)].sum(0)
        return torch.nn.functional.interpolate(prob[None, None], size=img.size[::-1], mode="bilinear")[0, 0].cpu().numpy()

    @torch.inference_mode()
    def patch_features(self, img, size=448):
        inp = self.head_proc(images=img.resize((size, size)), do_resize=False, do_center_crop=False,
                             return_tensors="pt").to(self.device)
        tok = self.head_bb(**inp).last_hidden_state[0, 1:].float().cpu().numpy()
        n = int(np.sqrt(len(tok)))
        return tok[: n * n].reshape(n, n, -1)

    def head_prob(self, img):
        if self.head is None:
            return None
        f = self.patch_features(img, self.head["size"])
        p = self.head["clf"].predict_proba(f.reshape(-1, f.shape[-1]))[:, 1].reshape(f.shape[:2])
        return np.asarray(Image.fromarray(p.astype(np.float32)).resize(img.size, Image.BILINEAR))

    @torch.inference_mode()
    def sam_embed(self, img):
        inp = self.sam_proc(images=img, return_tensors="pt").to(self.device)
        return self.sam.get_image_embeddings(inp["pixel_values"]), inp

    @torch.inference_mode()
    def sam_masks(self, img, emb, inp, points=None, labels=None, boxes=None):
        """Prompts are mapped to SAM's input frame directly (x new/old, the processor's own rule), so the photo is
        not re-preprocessed on every call. This is the hot loop for grains."""
        oh, ow = [float(v) for v in inp["original_sizes"][0]]
        rh, rw = [float(v) for v in inp["reshaped_input_sizes"][0]]
        k = torch.tensor([rw / ow, rh / oh], device=self.device)
        kw = {}
        if points is not None:
            kw["input_points"] = torch.tensor(points, dtype=torch.float32, device=self.device) * k
            kw["input_labels"] = torch.tensor(labels, dtype=torch.long, device=self.device)
        if boxes is not None:
            kw["input_boxes"] = torch.tensor(boxes, dtype=torch.float32, device=self.device) * k.repeat(2)
        out = self.sam(image_embeddings=emb, multimask_output=True, **kw)
        masks = self.sam_proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]
        return masks.numpy(), out.iou_scores[0].float().cpu().numpy()

    def sam_soft(self, img, mask, prob, emb, inp, k=6, seed=0):
        """Mean of SAM masks over k jittered prompt sets: box +-5 %, 5 inside and 3 outside points redrawn.
        Each run keeps the candidate that best agrees with the current outline x SAM's quality score, and a
        run is dropped if that agreement is below 0.5. Returns the soft mask and the mean agreement."""
        rng = np.random.default_rng(seed)
        ys, xs = np.nonzero(mask)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        pin = np.argwhere(mask & (prob > np.percentile(prob[mask], 60)))
        pout = np.argwhere(~ndimage.binary_dilation(mask, iterations=15) & (prob < 0.15))
        if len(pin) < 5 or len(pout) < 3:
            return None, None
        acc, agrees = np.zeros(mask.shape, np.float32), []
        for _ in range(k):
            j = lambda v, L: float(np.clip(v + rng.normal(0, 0.05 * L), 0, None))
            box = [[[j(x0, x1 - x0), j(y0, y1 - y0), j(x1, x1 - x0), j(y1, y1 - y0)]]]
            pts = [pin[i][::-1].tolist() for i in rng.choice(len(pin), 5, replace=False)] + \
                  [pout[i][::-1].tolist() for i in rng.choice(len(pout), 3, replace=False)]
            m, iou = self.sam_masks(img, emb, inp, points=[[pts]], labels=[[[1] * 5 + [0] * 3]], boxes=box)
            best, bs, ba = None, 0, 0
            for c in range(m.shape[1]):
                a = (m[0, c] & mask).sum() / max((m[0, c] | mask).sum(), 1)
                if a * iou[0, c] > bs:
                    best, bs, ba = m[0, c], a * iou[0, c], a
            if best is not None and ba >= 0.5:
                acc += best
                agrees.append(ba)
        if not agrees:
            return None, None
        return acc / len(agrees), float(np.mean(agrees))

    # ------------------------------------------------------------------ depth
    def _gpu(self, model, on):
        """Swap a parked model in/out of the GPU WITHOUT new CPU allocations. Each tensor keeps one permanent CPU copy
        (pinned when possible, for faster copies). Going to the GPU copies it there; parking just points back at
        the CPU copy and frees the GPU one. A plain model.to("cpu") allocates fresh RAM on every move, which
        fragments memory until a long run fails with 'not enough memory'."""
        if not self.offload or model is None:
            return
        store = self._cpu_copies.setdefault(id(model), {})
        if not store:
            for name, t in list(model.named_parameters()) + list(model.named_buffers()):
                c = t.data if t.device.type == "cpu" else t.data.cpu()
                try:
                    c = c.pin_memory()
                except RuntimeError:
                    pass
                store[name] = c
        tensors = dict(list(model.named_parameters()) + list(model.named_buffers()))
        for name, t in tensors.items():
            t.data = store[name].to(self.device, non_blocking=True) if on else store[name]
        if not on and self.device == "cuda":
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def _da(self, im):
        inp = self.da_proc(images=im, return_tensors="pt").to(self.device, self.da.dtype)
        d = self.da(**inp).predicted_depth.float()
        return torch.nn.functional.interpolate(d[None], size=im.size[::-1], mode="bicubic")[0, 0].clamp(min=0.3).cpu().numpy()

    @torch.inference_mode()
    def _pro(self, im):
        inp = self.pro_proc(images=im, return_tensors="pt").to(self.device, self.pro.dtype)
        r = self.pro_proc.post_process_depth_estimation(self.pro(**inp), target_sizes=[im.size[::-1]])[0]
        return r["predicted_depth"].float().clamp(min=0.3).cpu().numpy(), float(r["focal_length"])

    @torch.inference_mode()
    def _zoe(self, im):
        inp = self.zoe_proc(images=im, return_tensors="pt").to(self.device)
        out = self.zoe(**inp)
        try:
            d = self.zoe_proc.post_process_depth_estimation(out, source_sizes=[im.size[::-1]])[0]["predicted_depth"]
        except Exception:
            d = torch.nn.functional.interpolate(out.predicted_depth[None].float(), size=im.size[::-1], mode="bicubic")[0, 0]
        return d.float().clamp(min=0.3).cpu().numpy()

    def depth(self, img, f_exif):
        """Fused metric depth. Returns depth, per-pixel log-variance, focal estimate (px), and per-model mean maps."""
        mir = ImageOps.mirror(img)
        gam = ImageEnhance.Brightness(img).enhance(0.8)
        flip = lambda d: d[:, ::-1]
        stacks, names, f_pro = [], [], None
        da = [self._da(img), flip(self._da(mir)), self._da(gam)]
        stacks.append(np.log(np.stack(da)))
        names.append("depth_anything")
        if self.pro is not None:
            self._gpu(self.pro, True)
            (p0, f0), (p1, f1) = self._pro(img), self._pro(mir)
            self._gpu(self.pro, False)
            f_pro = float(np.sqrt(f0 * f1))
            pro = np.stack([p0 * f_pro / f0, flip(p1) * f_pro / f1])   # one focal for both runs
            if f_exif:                       # Depth Pro depth scales with its focal guess: use the known focal
                pro = pro * f_exif / f_pro
            stacks.append(np.log(pro))
            names.append("depth_pro")
        if self.zoe is not None:
            self._gpu(self.zoe, True)
            stacks.append(np.log(np.stack([self._zoe(img), flip(self._zoe(mir))])))
            self._gpu(self.zoe, False)
            names.append("zoedepth")
        if self.device == "cuda":
            torch.cuda.empty_cache()
        # scale region: the lower 2/3 of the photo (ground, not sky), where the landslide usually is
        reg = np.zeros(img.size[::-1], bool)
        reg[img.size[1] // 3:] = True
        fused, var, means, model_scale_sd = S.fuse_depths(stacks, prior_sd=DEPTH_SD, region=reg)
        gray = np.asarray(img.convert("L"), np.float32) / 255
        r = max(4, int(0.008 * max(img.size)))
        fused = S.guided_filter(gray, fused, r, 1e-3)
        self._model_scale_sd = model_scale_sd
        return np.exp(fused), var, f_pro, dict(zip(names, means)), float(np.median(np.abs(stacks[0][0] - stacks[0][1])))

    # ------------------------------------------------------------------ scale from known objects
    @torch.inference_mode()
    def object_scale(self, img, depth, f):
        """Weighted median of the scale factor true_height / measured_height over detected known-height objects.
        Weights = detection score / height variance. Outliers beyond 3 MAD are dropped. Boxes touching the photo
        edge are rejected (truncated objects). Depth is read in the bottom-centre of the box, where the object
        meets the ground, not from the background behind it."""
        if self.owl is None:
            return None
        W, H = img.size
        q = [t for t, _, _ in KNOWN_OBJECTS]
        inp = self.owl_proc(text=[q], images=img, return_tensors="pt").to(self.device)
        inp["pixel_values"] = inp["pixel_values"].to(self.owl.dtype)
        out = self.owl(**inp)
        Sd = max(W, H)                      # OWLv2 pads to a square
        res = self.owl_proc.image_processor.post_process_object_detection(out, threshold=0.25, target_sizes=torch.tensor([[Sd, Sd]]))[0]
        votes = []
        for sc, lb, bx in zip(res["scores"].float().cpu().numpy(), res["labels"].cpu().numpy(), res["boxes"].float().cpu().numpy()):
            name, h_true, sd = KNOWN_OBJECTS[int(lb)]
            x0, y0, x1, y1 = bx
            if x0 < 3 or y0 < 3 or x1 > W - 3 or y1 > H - 3 or (y1 - y0) < 20:
                continue
            if "person" in name and (y1 - y0) < 1.3 * (x1 - x0):
                continue
            if "tree" in name and (y1 - y0) < 0.1 * H:
                continue
            ys, xs = slice(int(y0 + 0.7 * (y1 - y0)), int(y1)), slice(int(x0 + 0.3 * (x1 - x0)), int(x1 - 0.3 * (x1 - x0)) + 1)
            z = float(np.median(depth[ys, xs]))
            h_est = (y1 - y0) * z / f
            votes.append((np.log(h_true / h_est), float(sc) / sd ** 2, sd, name.replace("a photo of a ", "")))
        if not votes:
            return None
        lv = np.array([v[0] for v in votes])
        w = np.array([v[1] for v in votes])
        if len(lv) >= 3:
            med, mad = np.median(lv), 1.4826 * np.median(np.abs(lv - np.median(lv))) + 1e-6
            ok = np.abs(lv - med) <= 3 * mad
            votes, lv, w = [v for v, k in zip(votes, ok) if k], lv[ok], w[ok]
        o = np.argsort(lv)
        cw = np.cumsum(w[o]) / w.sum()
        m = float(lv[o][np.searchsorted(cw, 0.5)])         # weighted median
        se = float(np.sqrt(1 / np.sum([1 / v[2] ** 2 for v in votes])))
        spread = float(np.sqrt(np.sum(w * (lv - m) ** 2) / w.sum())) if len(lv) > 1 else 0.0
        strong = [v for v in votes if v[3] != "tree"]
        if not strong and len(votes) < 3:                  # trees alone are too weak unless there are several
            return None
        f_ = float(np.exp(m))
        if not 0.2 <= f_ <= 5:
            return None
        names = sorted({v[3] for v in votes})
        return {"factor": round(f_, 3), "log_sd": round(max(se, spread, 0.05), 3), "n": len(votes), "objects": names}

    # ------------------------------------------------------------------ grains (multi-zoom SAM)
    def _instances(self, im, region, emb, inp, grid, scale, offset, depth, f):
        """SAM instances from a point grid inside region (in im coordinates). Returns (b_axis_m, z) per grain:
        the minor axis of the convex hull (occlusion-completed), converted to metres in original pixels."""
        from skimage.measure import regionprops
        Hh, Ww = region.shape
        gy, gx = np.meshgrid(np.linspace(0, Hh - 1, grid), np.linspace(0, Ww - 1, grid), indexing="ij")
        pts = [[float(x), float(y)] for x, y in zip(gx.ravel(), gy.ravel()) if region[int(y), int(x)]]
        cand = []
        for i in range(0, len(pts), 32):        # 32 prompts/call: bounds the full-size mask tensor at ~300 MB of RAM
            ch = pts[i:i + 32]
            m, iou = self.sam_masks(im, emb, inp, points=[[[p] for p in ch]], labels=[[[1] for _ in ch]])
            for j in range(len(ch)):
                kk = int(np.argmin(m[j].reshape(3, -1).sum(1)))
                c, qv = m[j, kk], float(iou[j, kk])
                n = int(c.sum())
                if qv < 0.85 or n < 30 or n > 0.05 * region.sum() or (c & region).sum() < 0.9 * n:
                    continue
                cand.append((qv, c))
        cand.sort(key=lambda t: -t[0])
        taken, out = np.zeros(region.shape, bool), []
        oy, ox = offset
        for qv, c in cand:
            if (c & taken).sum() > 0.3 * c.sum():
                continue
            taken |= c
            rp = regionprops(c.astype(np.uint8))[0]
            hull = rp.image_convex
            b_px = regionprops(hull.astype(np.uint8))[0].axis_minor_length / scale
            cy, cx = rp.centroid
            yy, xx = int(oy + cy / scale), int(ox + cx / scale)
            z = float(depth[min(yy, depth.shape[0] - 1), min(xx, depth.shape[1] - 1)])
            out.append((float(b_px * z / f), z))
        return out

    def grains(self, img, mask, depth, f, emb, inp, max_tiles=6, tile=256):
        """Zoom 1 = whole photo. Zoom 2 = the deposit's best-covered tiles, upsampled 2x, so smaller grains can be
        resolved. Zoom-2 grains are weighted by mask area / tile coverage (counts per unit area must match).
        Each grain is taken from the zoom whose resolution limit it clears. The two zooms must agree in their
        overlap range."""
        g1 = self._instances(img, mask, emb, inp, 24, 1.0, (0, 0), depth, f)
        H, W = mask.shape
        tiles = []
        for y in range(0, H, tile):
            for x in range(0, W, tile):
                cov = mask[y:y + tile, x:x + tile].mean()
                if cov > 0.6:
                    tiles.append((cov, y, x))
        tiles = sorted(tiles, reverse=True)[:max_tiles]
        g2, covered = [], 0
        for _, y, x in tiles:
            crop = img.crop((x, y, min(x + tile, W), min(y + tile, H)))
            up = crop.resize((crop.width * 2, crop.height * 2), Image.BICUBIC)
            reg = np.asarray(Image.fromarray(mask[y:y + crop.height, x:x + crop.width]).resize(up.size, Image.NEAREST))
            e, i2 = self.sam_embed(up)
            g2 += self._instances(up, reg, e, i2, 12, 2.0, (y, x), depth, f)
            covered += mask[y:y + crop.height, x:x + crop.width].sum()
        z90 = float(np.percentile(depth[mask], 90))
        lim1, lim2 = 6 * z90 / f, 3 * z90 / f                # ~6 px b-axis at zoom 1, ~3 px (orig) at zoom 2
        wz2 = mask.sum() / max(covered, 1)
        b = [g[0] for g in g1 if g[0] >= lim1] + [g[0] for g in g2 if lim2 <= g[0] < lim1]
        cw = [1.0] * sum(g[0] >= lim1 for g in g1) + [wz2] * sum(lim2 <= g[0] < lim1 for g in g2)
        o1 = [g[0] for g in g1 if lim1 <= g[0] < 4 * lim1]
        o2 = [g[0] for g in g2 if lim1 <= g[0] < 4 * lim1]
        agree = (float(np.log(np.median(o2) / np.median(o1)))
                 if len(o1) >= 10 and len(o2) >= 10 else None)
        slope = None
        z1 = [(g[0], g[1]) for g in g1 if g[0] >= lim1]
        if len(z1) >= 30 and np.std(np.log([g[1] for g in z1])) > 0.1:
            slope = float(np.polyfit(np.log([g[1] for g in z1]), np.log([g[0] for g in z1]), 1)[0])
        return b, cw, lim2 if g2 else lim1, {"zoom1": len(g1), "zoom2": len(g2), "zoom_log_ratio": agree,
                                             "size_vs_distance_slope": slope}

    # ------------------------------------------------------------------ geometry
    @staticmethod
    def geometry(mask, depth, f, g, label):
        H, W = mask.shape
        v, u = np.nonzero(mask)
        rng = np.random.default_rng(0)
        if len(u) > 40000:
            sel = rng.choice(len(u), 40000, replace=False)
            u, v = u[sel], v[sel]
        P = S.backproject(u, v, depth[v, u], W, H, f)
        c, normal, inl = S.robust_plane(P)
        across = np.array([1.0, 0, 0]) - normal[0] * normal
        across /= np.linalg.norm(across)
        down = np.cross(normal, across)
        if down @ g < 0:
            down = -down
        a, d = (P - c) @ across, (P - c) @ down
        ext = lambda x: float(np.percentile(x, 98) - np.percentile(x, 2))
        zz = depth[mask]
        vv, uu = np.nonzero(mask)
        rays = np.stack([(uu - W / 2) / f, (vv - H / 2) / f, np.ones_like(zz)], 1)
        cos = np.abs(rays @ normal) / np.linalg.norm(rays, axis=1)
        area = float(np.sum((zz / f) ** 2 / np.clip(cos, 0.25, 1)))
        drop = ext(P @ g)
        length, width = ext(d), ext(a)
        horiz = float(np.sqrt(max(length ** 2 - drop ** 2, 0))) or length
        from skimage import measure
        perim = 0.0
        for cnt in measure.find_contours(np.pad(ndimage.binary_fill_holes(mask), 1).astype(float), 0.5):
            cnt = cnt - 1
            seg = np.linalg.norm(np.diff(cnt, axis=0), axis=1)
            iy, ix = np.clip(cnt[:-1, 0].astype(int), 0, H - 1), np.clip(cnt[:-1, 1].astype(int), 0, W - 1)
            perim += float(np.sum(seg * depth[iy, ix] / f))
        bins = np.linspace(np.percentile(d, 2), np.percentile(d, 98), 13)
        widths = [ext(a[(d >= lo) & (d < hi)]) for lo, hi in zip(bins[:-1], bins[1:]) if ((d >= lo) & (d < hi)).sum() > 30]
        base = {"width": width, "length": length, "area": area, "drop": drop, "horiz": horiz, "perimeter": perim,
                "median_width": float(np.median(widths)) if widths else width,
                "max_width": float(max(widths)) if widths else width,
                "slope_deg": float(np.degrees(np.arccos(np.clip(abs(normal @ g), 0, 1))))}
        return base, {"c": c, "normal": normal, "across": across, "down": down, "inliers": inl,
                      "widths": widths, "distance": float(np.median(zz))}

    # ------------------------------------------------------------------ main
    @staticmethod
    def _points_scale(depth, mask, mp, fsd):
        """Metric scale from triangulated points: s = median(z_tri / z_model), using points on or near the
        landslide when there are enough (the model's relative depth is most trusted where it is measured).
        log-sd = baseline knowledge (from the set) + the robust spread of the ratios over an effective sample
        of at most 10 (neighbouring points share the model's shape errors) + focal (the triangulated depth
        scales with focal too, but partly cancels: only 0.3 of its log-sd is kept)."""
        H, W = depth.shape
        xy = np.asarray(mp["xy"], float)
        z = np.asarray(mp["z"], float)
        ok = (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H) & (z > 0) & np.isfinite(z)
        xy, z = xy[ok], z[ok]
        xi, yi = xy[:, 0].astype(int), xy[:, 1].astype(int)
        near = ndimage.binary_dilation(mask, iterations=max(3, int(0.03 * max(H, W))))[yi, xi]
        use = near if near.sum() >= 12 else np.ones(len(z), bool)
        lr = np.log(z[use]) - np.log(depth[yi[use], xi[use]])
        med = float(np.median(lr))
        mad_sd = 1.4826 * float(np.median(np.abs(lr - med)))
        shape = mad_sd / np.sqrt(min(len(lr), 10))
        sd = float(np.sqrt(float(mp.get("log_sd", 0.15)) ** 2 + shape ** 2 + (0.3 * fsd) ** 2))
        src = (f"walk-along photos with a known baseline ({int(use.sum())} triangulated points"
               f"{' on the landslide' if use is near else ''}, x{np.exp(med):.2f})")
        return depth * np.exp(med), sd, src

    def __call__(self, img, label, f35=None, ref_distance_m=None, family=None, gravity=None, focal_sd=None,
                 focal_src=None, metric_points=None):
        """focal_sd/focal_src: when f35 comes from somewhere other than this photo's EXIF (e.g. a joined
        picture's focal recovered from the photo set's homographies), its log-sd and a name.
        metric_points: {"xy": N x 2 pixel coords in img, "z": N metric depths, "log_sd": scale log-sd of the
        source} from a walk-along set triangulated with a known baseline (overlap.stereo_points); it sets the
        metric scale (after a typed distance, before objects and the prior)."""
        if label not in PROMPTS:
            label = FAMILY_LABEL.get(family)
        if label is None:
            return {"available": False, "reason": "no landslide detected, so there is nothing to measure"}
        W, H = img.size
        gray = np.asarray(img.convert("L"), np.float32)
        fail = {"available": False, "label": label, "reason": "could not outline the landslide in this photo "
                "(take it from further away with the whole slope and some sky in view)"}

        # 1. depth (first: the graph cut uses depth edges)
        f_exif = f35 / 36.0 * max(W, H) if f35 else None
        with self.use(self.da):
            depth, logvar, f_pro, per_model, flip_stab = self.depth(img, f_exif)
        f_def = DEFAULT_F35 / 36.0 * max(W, H)
        if f_exif:
            f, focal_src = f_exif, focal_src or "EXIF"
        elif f_pro and 0.5 * f_def < f_pro < 3 * f_def:
            f, focal_src = f_pro, "Depth Pro estimate"
        else:
            f, focal_src = f_def, "typical phone camera"
        fsd = focal_sd if (focal_sd is not None and f_exif) else FOCAL_SD.get(focal_src, 0.25)

        # 2. outline: fuse CLIPSeg + head + SAM soft, damp vegetation/sky, graph cut
        with self.use(self.seg):
            p_clip = self.clip_prob(img, label)
        with self.use(self.head_bb if self.head else None):
            p_head = self.head_prob(img)
        bad = S.bad_prior(img)
        srcs, ws = [p_clip], [1.0]
        if p_head is not None:
            srcs.append(p_head)
            ws.append(float(self.head.get("weight", 1.0)))
        P0 = np.average(np.stack(srcs), axis=0, weights=ws)
        m0 = S.clean_mask(P0 > 0.5)
        sam_agree, emb = None, None
        if self.sam is not None and 0.005 < m0.mean() < 0.92:
            self._gpu(self.sam, True)          # SAM stays on the GPU through outline and grains
            emb, sam_inp = self.sam_embed(img)
            soft, sam_agree = self.sam_soft(img, m0, P0, emb, sam_inp)
            if soft is not None:
                srcs.append(soft)
                ws.append(1.5)                    # sharpest boundaries
        P = np.average(np.stack(srcs), axis=0, weights=ws) * (1 - 0.8 * bad)
        gc, lam, gc_stab = S.graph_cut_stable(P, img, np.log(depth))
        mask = S.clean_mask(gc)
        if not 0.005 < mask.mean() < 0.92:
            mask = S.clean_mask(P > 0.5)
        if not 0.005 < mask.mean() < 0.92:
            if self.sam is not None:
                self._gpu(self.sam, False)
            return fail
        variants_m = [mask] + [m for m in (S.clean_mask(P > 0.35), S.clean_mask(P > 0.65)) if 0.005 < m.mean() < 0.92]

        # 3. scale
        obj = None
        if ref_distance_m:
            depth = depth * float(ref_distance_m) / np.median(depth[mask])
            scale_sd, scale_src = float(np.hypot(0.07, fsd)), "your distance"
        elif metric_points and len(metric_points.get("z", [])) >= 8:
            depth, scale_sd, scale_src = self._points_scale(depth, mask, metric_points, fsd)
        else:
            with self.use(self.owl):
                obj = self.object_scale(img, depth, f)
            between = float(np.median(logvar[mask]))
            if obj:
                depth = depth * obj["factor"]
                scale_sd, scale_src = obj["log_sd"], f"{'/'.join(obj['objects'])} in the photo (x{obj['factor']})"
            elif self.calib:
                depth = depth * self.calib.get("per_class", {}).get(label, self.calib["factor"])
                cv = self.calib.get("cv_median_abs_pct_error_calibrated") or 30
                scale_sd, scale_src = float(np.log1p(cv / 100) * 1.25), "field calibration"
            else:
                scale_sd = float(np.sqrt(DEPTH_SD ** 2 + between + self._model_scale_sd ** 2
                                         + (0.6 * fsd) ** 2))
                scale_src = f"depth ensemble ({len(per_model)} models), focal from {focal_src}"
                if self.obj_prior.get("n", 0) >= 4:
                    fac = self.obj_prior["factor"]
                    if self.obj_prior.get("log_depth_slope"):   # farther scenes need less correction (refit, n=83)
                        fac = float(np.exp(np.log(fac) + self.obj_prior["log_depth_slope"]
                                           * (np.log(np.median(depth)) - np.log(self.obj_prior["depth_ref_m"]))))
                        fac = round(float(np.clip(fac, 0.5, 5.0)), 3)
                    depth = depth * fac
                    # the correction replaces the ensemble's unknown bias with the object evidence's own spread
                    scale_sd = float(np.sqrt(self.obj_prior["predictive_log_sd"] ** 2 + between
                                             + (0.6 * fsd) ** 2))
                    scale_src += f", corrected x{fac} by known-size objects (n={self.obj_prior['n']})"

        # 4. geometry for the outline and its variants
        g = np.array([0.0, 1.0, 0.0])
        if gravity is not None:
            gv = np.asarray(gravity, float)
            if np.linalg.norm(gv) > 0 and gv[1] > 0.3 * np.linalg.norm(gv):
                g = gv / np.linalg.norm(gv)
        base, aux = self.geometry(mask, depth, f, g, label)
        variants = [base] + [self.geometry(m, depth, f, g, label)[0] for m in variants_m[1:]]

        # 5. grains
        b, cw, lim, gdiag = ([], [], None, {})
        if emb is not None:
            b, cw, lim, gdiag = self.grains(img, mask, depth, f, emb, sam_inp)
        if self.sam is not None:
            self._gpu(self.sam, False)
        gs = S.grain_stats(b, lim, cw) if b else None
        if gs:
            zr, sl = gdiag.get("zoom_log_ratio"), gdiag.get("size_vs_distance_slope")
            gs["reliable"] = bool((zr is None or abs(zr) < 0.3) and (sl is None or abs(sl) < 0.5) and gs["ks"] < 0.15)
            gs.update({k: (round(v, 2) if isinstance(v, float) else v) for k, v in gdiag.items()})

        # 6. pre-failure surface volume
        pf = S.prefailure_volume(depth, mask, f, W, H, aux["c"], aux["across"], aux["down"], aux["normal"], avoid=bad > 0.7)

        # 7. class-specific volume model + Monte Carlo
        surf = None
        if pf and pf.get("available"):
            v_s = pf["depleted_m3"] if label in ("rotational_slide", "translational_slide") else pf["accumulated_m3"]
            # usable only when the relief clearly exceeds depth noise (TTA flip difference x distance); monocular
            # depth smooths relief, so even then the surface volume gets at least a 50 % (log) error
            # SOFT weight, not an on/off switch: the surface volume's log-error grows smoothly as the relief
            # approaches depth noise (TTA flip difference x distance), so small input changes cannot flip it in or out
            noise = max(2 * flip_stab * aux["distance"], 2 * pf["cv_rmse_m"], 1e-3)
            snr = pf["mean_abs_dh_m"] / noise
            if v_s > 0:
                rel = float(np.clip(pf["cv_rmse_m"] * pf["area_m2"] / v_s, 0.5, 1.5)) * (1 + 1 / max(snr, 1e-3) ** 2)
                surf = (v_s, min(rel, 5.0))
            pf["relief_to_noise"] = round(float(snr), 2)

        def volume_fn(bs, s, rng):
            A = bs["area"] * s ** 2
            if label in ("rotational_slide", "translational_slide"):
                al, ga, sd = GUZZETTI
                mu, sd_ = np.log(al * A ** ga), sd
            else:
                t, lo, hi = THICKNESS[label]     # no grain-size floor: it switched on/off with the grain check
                m_, sd_ = S.lognormal_from_interval(lo, hi)
                mu = np.log(A * t)
            draw_law = rng.normal(mu, sd_)
            if surf is None:
                return float(np.exp(draw_law))
            v_s, rel = surf
            draw_s = rng.normal(np.log(v_s * s ** 3), rel)
            w1, w2 = 1 / sd_ ** 2, 1 / rel ** 2                # inverse-variance combination in log space
            return float(np.exp((w1 * draw_law + w2 * draw_s) / (w1 + w2)))

        pct, vols = S.monte_carlo(variants, scale_sd, volume_fn)
        size_p, lo_ = {}, 0.0
        for lim_, name in SIZE_CLASSES:                  # probability of each Fell (1994) class from the draws
            frac = float(np.mean((vols >= lo_) & (vols < lim_)))
            lo_ = lim_
            if frac >= 0.05:
                size_p[name] = round(frac, 2)
        vol = float(np.median(vols))

        # 8. checks
        haze = S.haze_rho(gray, depth)
        tex = S.texture_depth_slope(gray, mask, depth)
        ratios = {}
        keys = list(per_model)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ratios[f"{keys[j]}/{keys[i]}"] = round(float(np.exp(np.median((per_model[keys[j]] - per_model[keys[i]])[mask]))), 2)
        hl = base["drop"] / base["horiz"] if base["horiz"] > 0 else None
        cor = S.corominas_residual(hl, vol) if label != "fall_topple" else None
        issues = []
        if flip_stab > 0.1:
            issues.append("depth changes when the photo is mirrored")
        if ratios and max(abs(np.log(r)) for r in ratios.values()) > np.log(2):
            issues.append("the depth models disagree by more than 2x (scale = their consensus; range widened)")
        if haze > 0.2:
            issues.append("contrast rises with distance (haze cue contradicts the depth)")
        # texture slope stays informational: real slopes mix materials, so it is not a reliable veto
        if aux["inliers"] < 0.5:
            issues.append("the surface is far from one plane (complex terrain)")
        if obj and abs(np.log(obj["factor"])) > np.log(2):
            issues.append(f"objects of known size say the depth models were off by x{obj['factor']}")
        if cor is not None and abs(cor) > 0.4:
            issues.append("mobility (H/L) is unusual for this volume (Corominas 1996)")
        border = mask[0].mean() + mask[-1].mean() + mask[:, 0].mean() + mask[:, -1].mean()

        # 9. assemble
        out = {"available": True, "label": label, "distance_m": round(aux["distance"], 1),
               "scale_source": scale_src, "scale_log_sd": round(scale_sd, 3), "focal_35mm": round(f / max(W, H) * 36, 1),
               "cut_off": bool(border > 0.15), "mask_fraction": round(float(mask.mean()), 3), "measures": [],
               "outline": "+".join(n for n, s_ in (("CLIPSeg", True), ("head", p_head is not None),
                                                   ("SAM", sam_agree is not None)) if s_) + " -> graph cut",
               "sam_agreement": round(sam_agree, 2) if sam_agree else None,
               "graph_cut": {"lambda": lam, "stability_iou": round(gc_stab, 3)},
               "tilt": "phone sensor" if gravity is not None else "assumed upright",
               "calibrated": "calibration" in scale_src, "objects_used": obj,
               "depth_checks": {"tta_flip_logdiff": round(flip_stab, 3), "model_ratios": ratios,
                                "haze_rho": round(haze, 2), "texture_depth_slope": round(tex, 2) if tex is not None else None,
                                "plane_inliers": round(aux["inliers"], 2),
                                "corominas_residual": round(cor, 2) if cor is not None else None},
               "depth_issues": issues, "depth_trust": "high" if not issues else ("medium" if len(issues) == 1 else "low"),
               "size_class": max(size_p, key=size_p.get) if size_p else None, "size_class_prob": size_p}
        r1 = lambda x: round(x, 2) if x < 10 else round(x, 1)

        def add(key, name, unit, src=None):
            val = base[src or key]
            p = pct[src or key]
            out["measures"].append({"key": key, "name": name, "value": r1(val), "unit": unit, "range": [r1(p[0]), r1(p[2])]})

        if label == "fall_topple":
            add("width", "Deposit width (across slope)", "m")
            add("length", "Deposit length (down slope)", "m")
            add("drop", "Height of the rock face + deposit", "m")
            add("area", "Deposit area", "m²")
        elif label in ("rotational_slide", "translational_slide"):
            add("width", "Width (across slope)", "m")
            add("length", "Length (down slope)", "m")
            add("drop", "Height (crown to toe)", "m")
            add("area", "Area of the scar", "m²")
            out["depth_length_ratio_typical"] = "0.15-0.33" if label == "rotational_slide" else "< 0.10"
        else:
            add("length", "Track length (down slope)", "m")
            add("median_width", "Median width", "m")
            add("max_width", "Max width (fan / lobe)", "m")
            add("drop", "Vertical drop along the track", "m")
            add("area", "Inundated area", "m²")
            out["width_profile_m"] = [r1(w) for w in aux["widths"]]
            if label == "debris_flow":
                out["volume_griswold_iverson_m3"] = r1((base["area"] / 200.0) ** 1.5)
        how = {"fall_topple": "talus area x thickness", "debris_flow": "area x deposit thickness",
               "earth_mud_flow": "area x deposit thickness"}.get(label, "Guzzetti 2009 law")
        out["measures"].append({"key": "volume", "name": f"Volume ({how}{' + pre-failure surface' if surf else ''})",
                                "unit": "m³", "value": r1(vol), "range": [r1(pct['volume'][0]), r1(pct['volume'][2])]})
        out["mean_thickness_m"] = r1(vol / max(base["area"], 1e-6))
        if pf:
            out["prefailure_surface"] = {k: (r1(v) if isinstance(v, float) else v) for k, v in pf.items()}
        if gs:
            out["grains"] = gs
            if label == "fall_topple":
                blocks = sorted([x for x in b if x >= 0.2], reverse=True)
                out["block_count"] = len(blocks)
                if blocks:
                    out["measures"].append({"key": "block", "name": "Largest block (b-axis)", "unit": "m",
                                            "value": r1(blocks[0]), "range": [r1(blocks[0] * np.exp(-1.645 * scale_sd)),
                                                                              r1(blocks[0] * np.exp(1.645 * scale_sd))]})
                    out["block_size_class"] = next(n for l_, n in BLOCK_CLASSES if blocks[0] < l_)
                    out["largest_block_volume_m3"] = r1(float(np.pi / 6 * blocks[0] ** 3))
        out["length_width_ratio"] = round(base["length"] / max(base["width"], 1e-6), 2)
        out["slope_angle_deg"] = round(base["slope_deg"])
        if hl and label != "fall_topple":
            out["travel_angle_deg"] = round(float(np.degrees(np.arctan(hl))))
            out["H_over_L"] = round(hl, 2)
        out["perimeter_m"] = r1(base["perimeter"])
        out["compactness"] = round(min(1.0, 4 * np.pi * base["area"] / max(base["perimeter"], 1e-6) ** 2), 2)
        notes = []
        if out["cut_off"]:
            notes.append("The landslide runs off the edge of the photo: the sizes are minimums.")
        if not ref_distance_m and not obj:
            notes.append("Enter the distance to the slope for a tighter estimate.")
        out["notes"] = notes
        out["overlay"] = mask
        return out


def measure_ranked(dim, img, pred, **kw):
    """Measure as the predicted class. When the classifier is unsure (status uncertain or the top-2 margin is
    below 0.2), also measure as the runner-up. Report it, plus a probability-weighted geometric mean of the
    measures both share."""
    main = dim(img, pred["label"], family=pred.get("family"), **kw)
    rk = pred.get("ranked") or []
    if (main.get("available") and len(rk) > 1 and rk[1][0] in PROMPTS and rk[1][0] != main["label"]
            and (pred.get("status") == "uncertain" or rk[0][1] - rk[1][1] < 0.2)):
        alt = dim(img, rk[1][0], **kw)
        if alt.get("available"):
            alt.pop("overlay", None)
            p1, p2 = rk[0][1], rk[1][1]
            am = {m["key"]: m for m in alt["measures"]}
            main["alternative"] = {"label": alt["label"], "prob": round(p2, 2), "measures": alt["measures"]}
            main["blended"] = {m["key"]: round(float(np.exp((p1 * np.log(m["value"]) + p2 * np.log(am[m["key"]]["value"]))
                                                            / (p1 + p2))), 2)
                               for m in main["measures"] if m["key"] in am and m["value"] > 0 and am[m["key"]]["value"] > 0}
    return main


def fuse_site(results):
    """Several photos of ONE landslide -> one estimate per measure. Each photo sees a different part from a different
    distance, so single-photo sizes of one site spread ~1.5-2x (same-site test). Per measure: median of the photos'
    values in log space (robust to one bad photo). Range: the photos' own P5-P95 combined with the between-photo
    spread, i.e. P5 = exp(median log P5 - 1.645 sd_between) and likewise for P95. The most common class label is
    used; photos measured as another class are ignored."""
    ok = [r for r in results if r.get("available")]
    if not ok:
        return {"available": False, "reason": "no photo could be measured"}
    from collections import Counter
    label = Counter(r["label"] for r in ok).most_common(1)[0][0]
    ok = [r for r in ok if r["label"] == label]
    keys = [m["key"] for m in ok[0]["measures"]]
    out = {"available": True, "label": label, "n_photos": len(ok), "measures": []}
    for k in keys:
        ms = [next(m for m in r["measures"] if m["key"] == k) for r in ok if any(m["key"] == k for m in r["measures"])]
        v = np.log([max(m["value"], 1e-6) for m in ms])
        lo = np.log([max(m["range"][0], 1e-6) for m in ms])
        hi = np.log([max(m["range"][1], 1e-6) for m in ms])
        sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        out["measures"].append({"key": k, "name": ms[0]["name"], "unit": ms[0]["unit"],
                                "value": round(float(np.exp(np.median(v))), 2),
                                "range": [round(float(np.exp(np.median(lo) - 1.645 * sd)), 2),
                                          round(float(np.exp(np.median(hi) + 1.645 * sd)), 2)],
                                "between_photo_log_sd": round(sd, 3), "n": len(ms)})
    return out


def draw_overlay(img, mask, color=(255, 120, 0)):
    """Photo with the measured area tinted and outlined."""
    m = Image.fromarray((mask * 255).astype(np.uint8))
    edge = Image.fromarray((ndimage.binary_dilation(mask, iterations=3) & ~mask).astype(np.uint8) * 255)
    out = Image.composite(Image.new("RGB", img.size, color), img, m.point(lambda p: p * 0.35))
    out.paste(Image.new("RGB", img.size, color), mask=edge)
    return out


def main():
    from predict import Predictor, load_image
    ap = argparse.ArgumentParser(description="classify a photo, then measure the landslide")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--label", help="skip classification and measure as this class")
    ap.add_argument("--distance", type=float, help="distance from camera to the slope in metres (optional)")
    ap.add_argument("--save-overlay", action="store_true", help="write <photo>_dims.jpg next to each photo")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    ap.add_argument("--site", action="store_true", help="all photos show ONE landslide: also print the fused estimate")
    args = ap.parse_args()
    pred = None if args.label else Predictor()
    dim = Dimensioner()
    site_results = []
    for path in args.images:
        img = load_image(path)
        r = {"label": args.label, "family": None} if args.label else pred([img])[0]
        res = measure_ranked(dim, img, r, f35=focal_35mm(path), ref_distance_m=args.distance)
        print(f"\n{Path(path).name}: {res.get('label', r['label'])}")
        if not res["available"]:
            print("  " + res["reason"])
            continue
        mask = res.pop("overlay")
        if args.json:
            print(json.dumps(res, indent=1, default=str))
        else:
            for m in res["measures"]:
                print(f"  {m['name']:52s} {m['value']:>9} {m['unit']}  (P5-P95 {m['range'][0]}-{m['range'][1]})")
            for k in ("size_class_prob", "scale_source", "depth_trust", "depth_issues", "grains", "prefailure_surface",
                      "slope_angle_deg", "travel_angle_deg", "outline", "alternative", "blended"):
                if res.get(k) is not None:
                    print(f"  {k:24s} {res[k]}")
        if args.save_overlay:
            draw_overlay(img, mask).save(Path(path).with_name(Path(path).stem + "_dims.jpg"), quality=88)
        site_results.append(res)
    if args.site and len(site_results) > 1:
        fused = fuse_site(site_results)
        print(f"\nSITE ESTIMATE from {fused.get('n_photos', 0)} photos ({fused.get('label')}):")
        for m in fused.get("measures", []):
            print(f"  {m['name']:52s} {m['value']:>9} {m['unit']}  (P5-P95 {m['range'][0]}-{m['range'][1]}, "
                  f"between-photo log-sd {m['between_photo_log_sd']})")


if __name__ == "__main__":
    main()
