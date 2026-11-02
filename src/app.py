"""Field app: an engineer takes a phone photo of a slope and gets the landslide type.

Runs a small web server on this laptop. Phones on the same Wi-Fi/hotspot open the printed
address, tap "Take photo", and get:
  - a photo check (asks for a better photo if it is aerial, a close-up, too dark, ...)
  - the landslide type with probabilities, its Hungr et al. (2014) names
  - what to verify on site, and immediate safety notes
The browser shrinks the photo to 1280 px and fixes its rotation before upload, so it works
on slow mobile data. Uploads are saved to data/app_uploads/ (with the prediction) so they can
be labeled later and added to the training data (`--no-save` turns this off).

Usage:  python src/app.py [--port 8000] [--model auto|probe_base|probe_large|probe_siglip|cnn]
                         [--vlm] [--no-save] [--max-saved 2000] [--token SECRET]
        --token: phones must open http://<laptop-ip>:8000/?t=SECRET (use on shared networks)
        --vlm adds a Qwen2.5-VL second opinion and description (needs ~6 GB GPU, ~6 s/photo)
"""
import common  # noqa: F401  (sets HF cache dir; must precede transformers/timm)
import argparse
import base64
import io
import json
import shutil
import socket
import ssl
import subprocess
import threading
import uuid
from collections import OrderedDict
import time
import warnings
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from common import CLASSES, DATA
from PIL import Image
from logo import LOGO_SVG, png as logo_png
from ui import build_page
from predict import Predictor, load_image
from dimension import Dimensioner, draw_overlay, focal_35mm, fuse_site, measure_ranked
import overlap

ICONS = {"/favicon.svg": (LOGO_SVG.encode(), "image/svg+xml"),
         "/favicon.ico": (logo_png(48), "image/png"),
         "/apple-touch-icon.png": (logo_png(180), "image/png")}

Image.MAX_IMAGE_PIXELS = 40_000_000   # larger uploads are refused (decompression-bomb guard)
warnings.simplefilter("error", Image.DecompressionBombWarning)   # 40-80 MP would otherwise only warn

UPLOADS = DATA / "app_uploads"

MAX_JOBS = 6       # photos kept in memory for the size step (each ~36 MB decoded at 12 MP)
MAX_SETS = 3       # overlapping photo sets kept between the overlap check and "join and analyse"
EDGE_CUT = 0.15    # same rule as dimension.py's "cut_off": >15% of an image border covered by the outline


def cut_edges(mask):
    """Which photo borders the landslide outline runs off (guides the retake: step back, tilt up ...)."""
    return [k for k, v in {"top": mask[0], "bottom": mask[-1], "left": mask[:, 0], "right": mask[:, -1]}.items()
            if v.mean() > EDGE_CUT]


SIZE_SHOTS = ("overview", "scale")   # whole-slope shots; scarp and deposit close-ups see only part of the landslide


def fuse_shots(shots):
    """Several survey photos of ONE site -> one answer.
    Type: mean of the photos' class probabilities (every shot is evidence about the type).
    Sizes: dimension.fuse_site over the whole-slope shots (all measured shots if there are none): the median
    across photos with the between-photo spread added to the range. Grains: the deposit close-up if reliable."""
    if not isinstance(shots, list) or not 1 <= len(shots) <= 8:
        raise ValueError("1-8 shots")
    probs = {c: 0.0 for c in CLASSES}
    for s in shots:
        for c, p in s["ranked"]:
            if c in probs:
                probs[c] += min(max(float(p), 0.0), 1.0) / len(shots)
    ranked = sorted(([c, round(p, 4)] for c, p in probs.items()), key=lambda x: -x[1])
    dims = {str(s.get("id")): (s.get("dimensions") or {}) for s in shots}
    use = ([k for k in SIZE_SHOTS if dims.get(k, {}).get("available")]
           or [k for k, d in dims.items() if d.get("available")])
    fused = fuse_site([dims[k] for k in use]) if use else {"available": False, "reason": "no shot could be measured"}
    grains = None
    for k in ("deposit", *dims):
        g = (dims.get(k) or {}).get("grains") or {}
        if g.get("reliable"):
            grains = g
            break
    return {"n": len(shots), "type": {"label": ranked[0][0], "ranked": ranked}, "dimensions": fused,
            "size_shots": use, "grains": grains}


GUIDE_LOG = UPLOADS / "guidance_log.csv"
GUIDE_FIELDS = ("flags", "override", "retakes", "live", "mode", "shot", "lang", "set")


def log_guidance(gx, stamp, r):
    """What the guidance did before this upload (photo-check flags, override, retakes, live camera, survey
    shot), so tools/guidance_report.py can test whether the guidance helps."""
    try:
        g = json.loads(gx) if gx and len(gx) <= 2000 else None
    except ValueError:
        g = None
    if not isinstance(g, dict):
        return
    row = {"file": f"{stamp}.jpg", "status": r["status"], "pred": r["label"], "conf": round(r["confidence"], 3),
           "usable": round(r["usable"], 3)}
    for k in GUIDE_FIELDS:
        v = g.get(k, "")
        row[k] = "|".join(map(str, v))[:200] if isinstance(v, list) else str(v)[:40]
    pd.DataFrame([row]).to_csv(GUIDE_LOG, mode="a", header=not GUIDE_LOG.exists(), index=False)


INFO = {
    "fall_topple": {
        "title": "Rock fall / topple",
        "hungr": "Rock/ice fall, boulder/debris fall, rock block topple, flexural topple",
        "check": ["Fresh, light-coloured scars on the rock face where blocks came from",
                  "Open joints or overhangs above the scar that could release more blocks",
                  "Size of the largest block on the road or at the toe",
                  "Impact marks on the road, damaged trees or barriers"],
        "safety": "More blocks can fall without warning, especially during/after rain or blasting. "
                  "Stay out of the run-out zone below the face; observe from the side.",
    },
    "rotational_slide": {
        "title": "Rotational slide (slump)",
        "hungr": "Rock / clay / silt rotational slide",
        "check": ["Curved (spoon-shaped) main scarp at the top",
                  "Ground or trees tilted back towards the slope",
                  "Tension cracks above the scarp, bulging at the toe",
                  "Road sagging or dropped with a curved crack"],
        "safety": "Slumps often keep moving and can retrogress upslope. Keep off the cracked crown "
                  "and the bulging toe; mark cracks and check them again for widening.",
    },
    "translational_slide": {
        "title": "Translational (planar) slide",
        "hungr": "Rock planar / wedge slide, gravel / sand / debris slide",
        "check": ["Flat sliding surface exposed (bedding, joint plane or soil-rock contact)",
                  "Dip of that surface compared with the slope (out-of-slope dip is unstable)",
                  "Shallow, elongated scar; displaced sheet of soil or slabs below",
                  "Seepage or wet zone along the sliding surface"],
        "safety": "Remaining material on the same plane can slide again, often after rain. "
                  "Do not stand on the displaced mass or directly below the scar.",
    },
    "debris_flow": {
        "title": "Debris flow",
        "hungr": "Debris flow, debris flood, debris avalanche",
        "check": ["Gully or stream channel feeding the deposit",
                  "Levees of boulders along the channel, mud lines on trees",
                  "Fan of mixed boulders and mud at the gully outlet / road crossing",
                  "Blocked culvert or bridge"],
        "safety": "Debris flows recur in the same channel and can arrive as fast surges during rain. "
                  "Never stand in or close to the channel; watch for rising water or noise upstream.",
    },
    "earth_mud_flow": {
        "title": "Earthflow / mudflow",
        "hungr": "Earthflow, mud flow",
        "check": ["Lobe or tongue of fine, wet soil; few large boulders",
                  "Hummocky surface, transverse cracks, wet springs",
                  "Source scar upslope, often in clayey or weathered soil",
                  "Movement indicators: bent fences, tilted poles, pushed drains"],
        "safety": "Wet earthflows can surge and are soft underfoot. Do not walk on the flow; "
                  "watch for new cracks and water ponding on the source area.",
    },
    "no_landslide": {
        "title": "No landslide visible",
        "hungr": "-",
        "check": ["Cracks at the top of the cut or slope",
                  "Bulging, seepage or wet patches on the face",
                  "Leaning trees, poles or retaining walls",
                  "Small fresh rock fragments on the road (early rock fall)"],
        "safety": "No failure is visible in this photo. If you saw cracks, seepage or fresh debris "
                  "on site, report it anyway: this tool only sees what is in the photo.",
    },
}

PHOTO_TIPS = ["Stand at the foot of the slope (or across the road) and point the phone up at it",
              "Fit the whole slope in the frame: the top (scarp/source) and the bottom (deposit/toe)",
              "Hold the phone level; take the photo in daylight",
              "Do not use drone, satellite or map images; one photo per slope"]

# The page (design system, glyphs, demo mode) lives in src/ui.py



class VLM:
    """Optional zero-shot second opinion; loaded once, used under a lock (one GPU job at a time)."""
    def __init__(self, name="Qwen/Qwen2.5-VL-7B-Instruct"):
        import vlm_label as V
        self.V = V
        self.model, self.proc = V.load_model(name, True)
        self.ids, self.prompt = V.letter_ids(self.proc.tokenizer), V.build_prompt()

    def __call__(self, img):
        img = img.copy()
        img.thumbnail((768, 768))
        probs = self.V.classify(self.model, self.proc, img, self.ids, self.prompt)
        best = max(range(len(probs)), key=probs.__getitem__)
        return {"label": self.V.OPTIONS[best], "confidence": probs[best],
                "description": self.V.explain(self.model, self.proc, img)}


def make_handler(pred, vlm, save, max_saved=2000, token=None, dim=None):
    lock = threading.Lock()       # one photo at a time: decode, predict, save (bounded memory)
    jobs = OrderedDict()          # recent classified photos: sizes follow (and re-measure) without a re-upload
    sets = OrderedDict()          # recent overlap checks: "join and analyse" follows without a re-upload
    olock = threading.Lock()      # overlap checks run on the CPU, beside the GPU photo lock
    names = {c: INFO[c]["title"] for c in CLASSES}
    page = build_page(LOGO_SVG, INFO, names, PHOTO_TIPS, pred.confident_min)

    class Handler(BaseHTTPRequestHandler):
        timeout = 30              # a stalled client cannot hold a thread forever

        def log_message(self, fmt, *a):
            pass

        def _authorized(self):
            return token is None or parse_qs(urlparse(self.path).query).get("t", [""])[0] == token

        def _send(self, code, body, ctype):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            icon = ICONS.get(urlparse(self.path).path)
            if icon:   # browsers fetch icons without the ?t= token; they reveal nothing
                return self._send(200, icon[0], icon[1])
            if not self._authorized():
                return self._send(403, "add ?t=<token> to the address", "text/plain")
            if urlparse(self.path).path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            else:
                self._send(404, "not found", "text/plain")

        def _json(self, code, obj):
            return self._send(code, json.dumps(obj), "application/json")

        def _q(self, key):
            return parse_qs(urlparse(self.path).query).get(key, [""])[0]

        def _dist_grav(self):
            try:
                dist = float(self._q("d") or 0) or None
            except ValueError:
                dist = None
            if dist and not 0.5 <= dist <= 5000:
                dist = None
            try:
                grav = [float(v) for v in self._q("g").split(",")]
                grav = grav if len(grav) == 3 and all(abs(v) <= 1.01 for v in grav) else None
            except ValueError:
                grav = None
            return dist, grav

        def _f35(self):
            """35 mm-equivalent focal length read by the phone from the original photo's EXIF (the browser's
            shrunk upload has none); None if absent or implausible."""
            try:
                v = float(self._q("f35") or 0)
            except ValueError:
                return None
            return v if 8 <= v <= 200 else None

        def _measure(self, img, raw, r, f35=None, **kw):
            """Dimensions for one classified photo (+ which photo edges the outline runs off).
            f35: focal length known by other means (phone EXIF, a joined picture); else the upload's own EXIF.
            kw: extra evidence for the size step (focal_sd, metric_points from a walk-along set)."""
            dist, grav = self._dist_grav()
            d = measure_ranked(dim, img, r, f35=f35 or self._f35() or focal_35mm(raw), ref_distance_m=dist, gravity=grav, **kw)
            if "overlay" in d:
                mask = np.asarray(d.pop("overlay"), bool)
                d["cut_edges"] = cut_edges(mask)
                buf = io.BytesIO()
                ov = draw_overlay(img, mask)
                ov.thumbnail((720, 720))
                ov.save(buf, "JPEG", quality=80)
                d["overlay"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
            return d

        def _save(self, img, r, stamp=None):
            """Keep the upload (to label later) with its prediction and what the guidance did."""
            if not save or len(list(UPLOADS.glob("*.jpg"))) >= max_saved:
                return
            stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            UPLOADS.mkdir(parents=True, exist_ok=True)
            img.save(UPLOADS / f"{stamp}.jpg", quality=92)
            log = UPLOADS / "log.csv"
            pd.DataFrame([{"file": f"{stamp}.jpg", "status": r["status"], "usable": round(r["usable"], 3),
                           "pred": r["label"], "conf": round(r["confidence"], 3), "model": r["model"]}]
                         ).to_csv(log, mode="a", header=not log.exists(), index=False)
            log_guidance(self._q("gx"), stamp, r)

        def _overlap(self, raw):
            """Check an overlapping set: which photos overlap, how well, in which order; join a turning set."""
            body = json.loads(raw)
            urls = body["images"]
            if not isinstance(urls, list) or not 2 <= len(urls) <= overlap.MAX_IMAGES:
                return self._json(400, {"error": f"send 2-{overlap.MAX_IMAGES} photos"})
            imgs, raws = [], []
            for u in urls:
                b = base64.b64decode(str(u).split(",", 1)[-1])
                imgs.append(load_image(io.BytesIO(b)))
                raws.append(b)
            f35 = [float(v) if isinstance(v, (int, float)) else None for v in (body.get("f35") or [])][:len(imgs)]
            kind = self._q("kind") or body.get("kind") or "turn"
            with olock:
                rep, ex = overlap.analyse(imgs, kind, f35)
            sid = uuid.uuid4().hex[:16]
            sets[sid] = {"imgs": imgs, "raws": raws, "ex": ex, "rep": rep, "f35": f35}
            while len(sets) > MAX_SETS:
                sets.popitem(last=False)
            rep["set"] = sid
            print(f"{datetime.now():%H:%M:%S} overlap    {len(imgs)} photos {rep['detected']} {rep['verdict']} ({rep['ms']} ms)")
            return self._json(200, rep)

        def _pano(self):
            """Type for a checked set (joined picture + single photos), then sizes via /measure?job=."""
            st = sets.get(self._q("set"))
            if st is None:
                return self._json(410, {"error": "the photo set is no longer on the server; sending it again"})
            ex, rep = st["ex"], st["rep"]
            main = ex["main"]
            t0 = time.time()
            with lock:
                batch = [st["imgs"][i] for i in main] + ([ex["mosaic"]] if ex["mosaic"] is not None else [])
                res = pred(batch)
                singles, rm = res[:len(main)], (res[-1] if ex["mosaic"] is not None else None)
                probs, u = overlap.combine(rm, singles)
                r = pred.decide(u, probs)
                r["model"] += " (photo set)"
                if ex["mosaic"] is not None:
                    img, f35 = ex["mosaic"], ex["mosaic_f35"]
                    raw = overlap.jpeg(img, f35)
                    r["image"] = "data:image/jpeg;base64," + base64.b64encode(raw).decode()
                    r["ref"] = ex["ref"]
                else:
                    c = ex["central"]
                    img, raw = st["imgs"][c], st["raws"][c]
                    f35 = (st["f35"][c] if c < len(st["f35"]) else None) or None
                    r["ref"] = c
                r["set"] = {"n": len(main), "joined": ex["mosaic"] is not None, "detected": rep["detected"],
                            "span_deg": rep["span_deg"], "focal_source": rep["focal_source"],
                            "photos": [{"i": i, "label": s["label"], "confidence": round(s["confidence"], 3)}
                                       for i, s in zip(main, singles)],
                            "joined_type": ({"label": rm["label"], "confidence": round(rm["confidence"], 3)} if rm else None)}
                r["measurable"] = bool(dim) and r["status"] != "bad_photo"
                if r["measurable"]:
                    r["job"] = uuid.uuid4().hex[:16]
                    jobs[r["job"]] = (img, raw, {k: r.get(k) for k in ("label", "ranked", "status", "family")}, f35)
                    while len(jobs) > MAX_JOBS:
                        jobs.popitem(last=False)
                r.update(ms=int((time.time() - t0) * 1000), names=names, info=INFO, tips=PHOTO_TIPS)
                self._save(img, r)
            print(f"{datetime.now():%H:%M:%S} {r['status']:10s} {r['label']} {r['confidence']:.0%} ({r['ms']} ms, set of {len(main)})")
            return self._json(200, r)

        def do_POST(self):
            path = urlparse(self.path).path
            if path not in ("/predict", "/measure", "/fuse", "/overlap", "/pano"):
                return self._send(404, "not found", "text/plain")
            try:
                n = int(self.headers.get("Content-Length", 0))
            except ValueError:
                n = 0
            limit = {"/predict": 25_000_000, "/measure": 0, "/fuse": 2_000_000, "/overlap": 40_000_000, "/pano": 0}[path]
            if not (0 < n <= limit or (path in ("/measure", "/pano") and n == 0)):
                self.close_connection = True   # body not read: do not reuse this connection
                return self._json(400, {"error": "send one photo (max 25 MB)" if path == "/predict" else "bad request"})
            raw = self.rfile.read(n) if n else b""   # read the (bounded) body before answering, even to refuse
            if not self._authorized():
                return self._json(403, {"error": "missing or wrong ?t= token"})
            if path == "/overlap":
                try:
                    return self._overlap(raw)
                except Exception as e:
                    print(f"overlap check failed: {e!r}")
                    return self._json(400, {"error": f"could not check the photo set ({type(e).__name__})"})
            if path == "/pano":
                try:
                    return self._pano()
                except Exception as e:
                    print(f"photo set failed: {e!r}")
                    return self._json(500, {"error": "the model failed on this photo set; try again"})
            if path == "/fuse":
                try:
                    return self._json(200, fuse_shots(json.loads(raw)["shots"]))
                except Exception as e:
                    return self._json(400, {"error": f"could not combine the shots ({type(e).__name__})"})
            if path == "/measure":
                if not dim:
                    return self._json(400, {"error": "sizes are switched off on this server (--no-dims)"})
                with lock:
                    job = jobs.get(self._q("job"))
                    if job is None:
                        return self._json(410, {"error": "the photo is no longer on the server; sending it again"})
                    t0 = time.time()
                    try:
                        d = self._measure(*job)
                    except Exception as e:
                        print(f"measuring failed: {e!r}")
                        return self._json(500, {"error": "the size estimate failed on this photo; try again"})
                    ms = int((time.time() - t0) * 1000)
                print(f"{datetime.now():%H:%M:%S} measured   {job[2]['label']} ({ms} ms)")
                return self._json(200, {"dimensions": d, "ms": ms})
            stage = self._q("stage") or "full"   # "type" = classify only; sizes follow via /measure?job=
            with lock:
                try:
                    img = load_image(io.BytesIO(raw))
                except Exception:   # not an image, truncated, or over the pixel limit
                    return self._json(400, {"error": "could not read the image (use a JPEG or PNG photo under 40 megapixels)"})
                t0 = time.time()
                try:
                    r = pred([img])[0]
                    if vlm:
                        r["vlm"] = vlm(img)
                    r["measurable"] = bool(dim) and r["status"] != "bad_photo"
                    if r["measurable"] and stage == "type":
                        r["job"] = uuid.uuid4().hex[:16]
                        jobs[r["job"]] = (img, raw, {k: r.get(k) for k in ("label", "ranked", "status", "family")}, self._f35())
                        while len(jobs) > MAX_JOBS:
                            jobs.popitem(last=False)
                    elif r["measurable"]:
                        r["dimensions"] = self._measure(img, raw, r)
                except Exception as e:   # e.g. out of GPU memory: answer the phone instead of hanging
                    print(f"prediction failed: {e!r}")
                    return self._json(500, {"error": "the model failed on this photo; try again"})
                r.update(ms=int((time.time() - t0) * 1000), names=names, info=INFO, tips=PHOTO_TIPS)
                self._save(img, r)
            print(f"{datetime.now():%H:%M:%S} {r['status']:10s} {r['label']} {r['confidence']:.0%} ({r['ms']} ms, {stage})")
            self._json(200, r)

    return Handler


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def self_signed_cert(ip):
    """A self-signed certificate for this laptop's LAN address (made once with openssl, then reused)."""
    d = DATA.parent / ".cache" / "tls"
    cert, key = d / f"{ip}.crt", d / f"{ip}.key"
    if not cert.exists():
        exe = shutil.which("openssl") or r"C:\Program Files\Git\usr\bin\openssl.exe"
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run([exe, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "825", "-keyout", str(key),
                        "-out", str(cert), "-subj", "/CN=landslide-app",
                        "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost"], check=True, capture_output=True)
    return str(cert), str(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable from phones on the same network")
    ap.add_argument("--model", default="auto",
                    help="auto = models/selected.json (chosen on train+val by tools/robustness_eval.py); "
                         "or probe_base, probe_large, probe_siglip, cnn")
    ap.add_argument("--vlm", action="store_true", help="add a Qwen2.5-VL second opinion (CUDA GPU, ~6 GB)")
    ap.add_argument("--no-dims", action="store_true", help="skip the dimension estimate (saves ~3 GB of GPU memory)")
    ap.add_argument("--no-save", action="store_true", help="do not keep uploaded photos")
    ap.add_argument("--max-saved", type=int, default=2000, help="stop saving uploads after this many")
    ap.add_argument("--token", help="require ?t=TOKEN in the address (recommended on shared networks)")
    ap.add_argument("--https", action="store_true",
                    help="serve over https so phones allow the live camera guide (self-signed certificate made "
                         "with openssl in .cache/tls; the phone shows a one-time warning to accept)")
    ap.add_argument("--cert", nargs=2, metavar=("CERT", "KEY"), help="use your own certificate (e.g. from mkcert)")
    args = ap.parse_args()

    pred = Predictor(args.model)
    vlm = VLM() if args.vlm else None
    dim = None if args.no_dims else Dimensioner()
    server = ThreadingHTTPServer((args.host, args.port),
                                 make_handler(pred, vlm, not args.no_save, args.max_saved, args.token, dim))
    scheme = "http"
    if args.https or args.cert:
        cert, key = args.cert or self_signed_cert(lan_ip())
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    q = f"/?t={args.token}" if args.token else ""
    print(f"Landslide app ready (model: {pred.model_name}, confident >= {pred.confident_min}).\n"
          f"  on this laptop: {scheme}://localhost:{args.port}{q}\n"
          f"  on a phone (same Wi-Fi/hotspot): {scheme}://{lan_ip()}:{args.port}{q}\n"
          + ("" if scheme == "https" else "  (the live camera guide on phones needs --https; otherwise the phone camera app is used)\n")
          + ("  uploads are NOT saved\n" if args.no_save else f"  uploads saved to {UPLOADS} (max {args.max_saved})\n")
          + "  (if the phone cannot connect, allow Python through Windows Firewall for private networks)")
    server.serve_forever()


if __name__ == "__main__":
    main()
