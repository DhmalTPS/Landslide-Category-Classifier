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
import io
import json
import socket
import threading
import time
import warnings
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

from common import CLASSES, DATA
from PIL import Image
from logo import LOGO_SVG, png as logo_png
from ui import build_page
from predict import Predictor, load_image

ICONS = {"/favicon.svg": (LOGO_SVG.encode(), "image/svg+xml"),
         "/favicon.ico": (logo_png(48), "image/png"),
         "/apple-touch-icon.png": (logo_png(180), "image/png")}

Image.MAX_IMAGE_PIXELS = 40_000_000   # larger uploads are refused (decompression-bomb guard)
warnings.simplefilter("error", Image.DecompressionBombWarning)   # 40-80 MP would otherwise only warn

UPLOADS = DATA / "app_uploads"

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


def make_handler(pred, vlm, save, max_saved=2000, token=None):
    lock = threading.Lock()       # one photo at a time: decode, predict, save (bounded memory)
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

        def do_POST(self):
            if urlparse(self.path).path != "/predict":
                return self._send(404, "not found", "text/plain")
            try:
                n = int(self.headers.get("Content-Length", 0))
            except ValueError:
                n = 0
            if not 0 < n <= 25_000_000:
                self.close_connection = True   # body not read: do not reuse this connection
                return self._send(400, json.dumps({"error": "send one photo (max 25 MB)"}), "application/json")
            raw = self.rfile.read(n)           # read the (bounded) body before answering, even to refuse
            if not self._authorized():
                return self._send(403, json.dumps({"error": "missing or wrong ?t= token"}), "application/json")
            with lock:
                try:
                    img = load_image(io.BytesIO(raw))
                except Exception:   # not an image, truncated, or over the pixel limit
                    return self._send(400, json.dumps({"error": "could not read the image (use a JPEG or PNG "
                                                                "photo under 40 megapixels)"}), "application/json")
                t0 = time.time()
                try:
                    r = pred([img])[0]
                    if vlm:
                        r["vlm"] = vlm(img)
                except Exception as e:   # e.g. out of GPU memory: answer the phone instead of hanging
                    print(f"prediction failed: {e!r}")
                    return self._send(500, json.dumps({"error": "the model failed on this photo; try again"}),
                                      "application/json")
                r.update(ms=int((time.time() - t0) * 1000), names=names, info=INFO, tips=PHOTO_TIPS)
                if save and len(list(UPLOADS.glob("*.jpg"))) < max_saved:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    UPLOADS.mkdir(parents=True, exist_ok=True)
                    img.save(UPLOADS / f"{stamp}.jpg", quality=92)
                    log = UPLOADS / "log.csv"
                    pd.DataFrame([{"file": f"{stamp}.jpg", "status": r["status"], "usable": round(r["usable"], 3),
                                   "pred": r["label"], "conf": round(r["confidence"], 3), "model": r["model"]}]
                                 ).to_csv(log, mode="a", header=not log.exists(), index=False)
            print(f"{datetime.now():%H:%M:%S} {r['status']:10s} {r['label']} {r['confidence']:.0%} ({r['ms']} ms)")
            self._send(200, json.dumps(r), "application/json")

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable from phones on the same network")
    ap.add_argument("--model", default="auto",
                    help="auto = models/selected.json (chosen on train+val by tools/robustness_eval.py); "
                         "or probe_base, probe_large, probe_siglip, cnn")
    ap.add_argument("--vlm", action="store_true", help="add a Qwen2.5-VL second opinion (CUDA GPU, ~6 GB)")
    ap.add_argument("--no-save", action="store_true", help="do not keep uploaded photos")
    ap.add_argument("--max-saved", type=int, default=2000, help="stop saving uploads after this many")
    ap.add_argument("--token", help="require ?t=TOKEN in the address (recommended on shared networks)")
    args = ap.parse_args()

    pred = Predictor(args.model)
    vlm = VLM() if args.vlm else None
    server = ThreadingHTTPServer((args.host, args.port),
                                 make_handler(pred, vlm, not args.no_save, args.max_saved, args.token))
    q = f"/?t={args.token}" if args.token else ""
    print(f"Landslide app ready (model: {pred.model_name}, confident >= {pred.confident_min}).\n"
          f"  on this laptop: http://localhost:{args.port}{q}\n"
          f"  on a phone (same Wi-Fi/hotspot): http://{lan_ip()}:{args.port}{q}\n"
          + ("  uploads are NOT saved\n" if args.no_save else f"  uploads saved to {UPLOADS} (max {args.max_saved})\n")
          + "  (if the phone cannot connect, allow Python through Windows Firewall for private networks)")
    server.serve_forever()


if __name__ == "__main__":
    main()
