"""Local web tool for human labeling. Opens http://localhost:8765

Keys: 1-6 = class, R = reject (unusable), V = toggle viewpoint "looking down" for the current
photo (default: up/level, the deployment case), Left/Right = previous/next.
Each reviewer's labels go to data/labels/<reviewer>.csv. Human labels always override the AI
annotator's (reviewer *_ai); several humans are combined by majority vote (common.final_labels).

Queues:
  default            photos no human has labeled yet (AI labels do not count) and you haven't
  --relabel-others   photos someone else labeled (blind second opinion, for agreement/kappa)
  --conflicts        photos whose human votes are tied (same rule final_labels uses)
Test-split photos are always shown WITHOUT the VLM suggestion (blind), so the VLM baseline is
not scored against labels it influenced. New photos get a split automatically on start.

Usage:
  python src/review.py --reviewer uday
  python src/review.py --reviewer uday --split test --relabel-others          # blind test check
  python src/review.py --reviewer faculty --split test --relabel-others --skip-ai-rejects --sample 50
  python src/review.py --reviewer uday --conflicts
"""
import argparse
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

import pandas as pd

from common import (CLASSES, DATA, LABELS_DIR, METADATA_CSV, REJECT, SPLITS_CSV, VLM_CSV,
                    all_labels, is_ai, read_csv, tied_images, write_csv)
from split import assign_splits

KEYS = {str(i + 1): c for i, c in enumerate(CLASSES)} | {"r": REJECT}


def build_queue(args):
    meta = read_csv(METADATA_CSV)
    splits = read_csv(SPLITS_CSV)
    if splits is None or not set(meta.image_id) <= set(splits.image_id):
        # New photos must get their split BEFORE anyone labels them, or a photo labeled with
        # the VLM suggestion visible could later land in the test set.
        splits = assign_splits(verbose=False)
    df = meta.merge(splits[["image_id", "split"]], on="image_id", how="left").fillna({"split": ""})
    if args.split:
        df = df[df.split == args.split]

    lab = all_labels()
    mine = lab[lab.reviewer == args.reviewer]
    others = lab[lab.reviewer != args.reviewer]
    human_labeled = set(lab.image_id[~lab.reviewer.map(is_ai)])
    if args.conflicts:
        df = df[df.image_id.isin(tied_images())]
    elif args.relabel_others:
        df = df[df.image_id.isin(others.image_id) & ~df.image_id.isin(mine.image_id)]
    else:
        df = df[~df.image_id.isin(human_labeled | set(mine.image_id))]
    if args.skip_ai_rejects:   # e.g. an expert sample: spend their time on usable photos
        ai_rej = set(lab.image_id[lab.reviewer.map(is_ai) & (lab.label == REJECT)])
        df = df[~df.image_id.isin(ai_rej)]

    vlm = read_csv(VLM_CSV)
    if vlm is not None:
        vlm = vlm.drop_duplicates("image_id", keep="last")[["image_id", "vlm_label", "vlm_conf", "cues"]]
        df = df.merge(vlm, on="image_id", how="left").fillna("")
    else:
        df = df.assign(vlm_label="", vlm_conf="", cues="")
    blind = df.split.isin(["test", ""]) | bool(args.blind) | bool(args.relabel_others) | bool(args.conflicts)
    df.loc[blind, ["vlm_label", "vlm_conf", "cues"]] = ""
    df["vlm_shown"] = (df.vlm_label != "").astype(int).astype(str)
    if args.sample:
        df = df.sample(min(args.sample, len(df)), random_state=0)
    df = df.sort_values(["site_id", "image_id"])
    cols = ["image_id", "split", "title", "description", "category", "url", "license",
            "hint_label", "vlm_label", "vlm_conf", "cues", "vlm_shown"]
    return df[cols].to_dict("records"), mine


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Landslide review</title>
<style>
body{margin:0;font:14px system-ui,sans-serif;background:#111;color:#eee;display:flex;height:100vh}
#img{flex:1;display:flex;align-items:center;justify-content:center;background:#000}
#img img{max-width:100%;max-height:100vh;object-fit:contain}
#side{width:420px;padding:14px;overflow:auto;box-sizing:border-box}
.btn{display:block;width:100%;text-align:left;margin:4px 0;padding:8px;border:1px solid #444;
 background:#1d1d1d;color:#eee;border-radius:6px;cursor:pointer}
.btn.cur{border-color:#4ade80;background:#14301f}.btn.vlm{outline:2px dashed #facc15}
.k{display:inline-block;width:18px;color:#facc15;font-weight:bold}
.d{color:#999;font-size:12px;margin-top:3px}.meta{color:#aaa;font-size:12px;word-break:break-word}
#prog{color:#4ade80;margin-bottom:8px}#view{margin:6px 0;color:#facc15}#err{color:#f87171}
</style></head><body><div id="img"><img id="im"></div><div id="side">
<div id="prog"></div><div id="err"></div><div id="info" class="meta"></div><div id="vlm" style="margin:8px 0"></div>
<div id="view"></div><div id="btns"></div>
<p class="meta">Keys: 1-6 class &middot; R reject &middot; V viewpoint (looking down) &middot; &larr;/&rarr; navigate</p></div>
<script>
const CL=__CLASSES__, KEYS=__KEYS__; let Q=[],i=0,labels={},down={},busy=false;
const esc=s=>String(s||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const safeUrl=u=>/^https?:\\/\\//i.test(u||'')?u:'';
async function load(){const r=await (await fetch('/api/queue')).json();Q=r.queue;labels=r.mine;down=r.down;show()}
function show(){
 if(!Q.length){document.getElementById('prog').textContent='Nothing to review.';return}
 i=Math.max(0,Math.min(i,Q.length-1));const q=Q[i],cur=labels[q.image_id];
 document.getElementById('im').src='/img/'+encodeURIComponent(q.image_id);
 const done=Q.filter(x=>labels[x.image_id]).length;
 document.getElementById('prog').textContent=`${i+1} / ${Q.length} · labeled ${done} · split: ${q.split||'?'}`;
 const u=safeUrl(q.url);
 document.getElementById('info').innerHTML=`<b>${esc(q.title)}</b><br>${esc(q.description)}<br>
  category: ${esc(q.category)}<br>hint: ${esc(q.hint_label)||'-'} · ${esc(q.license)}
  ${u?`· <a style="color:#93c5fd" href="${esc(u)}" target="_blank" rel="noopener">source</a>`:''}`;
 document.getElementById('vlm').innerHTML=q.vlm_label?`VLM suggests <b>${esc(q.vlm_label)}</b>
  (${(+q.vlm_conf*100).toFixed(0)}%) ${esc(q.cues)}`:'<span class="meta">VLM suggestion hidden</span>';
 document.getElementById('view').textContent='Viewpoint: '+(down[q.image_id]?'looking DOWN from above (V to change)':'up / level (V to change)');
 document.getElementById('btns').innerHTML=Object.entries(KEYS).map(([k,c])=>
  `<button class="btn ${cur===c?'cur':''} ${q.vlm_label===c?'vlm':''}" onclick="lab('${c}')">
  <span class="k">${k.toUpperCase()}</span>${c}<div class="d">${esc(CL[c]||'Unusable: aerial view, map, diagram, close-up, not a slope')}</div></button>`).join('');
 if(Q[i+1]) new Image().src='/img/'+encodeURIComponent(Q[i+1].image_id);
}
async function lab(c){if(busy)return;busy=true;const q=Q[i];
 try{const res=await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({image_id:q.image_id,label:c,vlm_shown:q.vlm_shown,viewpoint:c==='reject'?'':(down[q.image_id]?'down':'up')})});
  if(!res.ok)throw new Error('save failed ('+res.status+')');
  labels[q.image_id]=c;document.getElementById('err').textContent='';i++;show();
 }catch(e){document.getElementById('err').textContent=e.message+' - not saved, try again';}
 finally{busy=false}}
document.onkeydown=e=>{const k=e.key.toLowerCase();if(busy)return;
 if(KEYS[k])lab(KEYS[k]);else if(k==='v'&&Q[i]){down[Q[i].image_id]=!down[Q[i].image_id];show()}
 else if(e.key==='ArrowRight'){i++;show()}else if(e.key==='ArrowLeft'){i--;show()}};
load();
</script></body></html>"""


def make_handler(queue, mine, path, reviewer):
    rows = mine.to_dict("records")
    allowed = {q["image_id"] for q in queue} | {r["image_id"] for r in rows}
    files = {}
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        timeout = 60

        def log_message(self, *a):
            pass

        def send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                page = PAGE.replace("__CLASSES__", json.dumps(CLASSES)).replace("__KEYS__", json.dumps(KEYS))
                self.send(page.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/queue":
                with lock:
                    mine_map = {r["image_id"]: r["label"] for r in rows}
                    down = {r["image_id"]: True for r in rows if r.get("viewpoint") == "down"}
                self.send(json.dumps({"queue": queue, "mine": mine_map, "down": down}).encode(), "application/json")
            elif self.path.startswith("/img/"):
                image_id = unquote(self.path[5:])
                if not files:
                    meta = read_csv(METADATA_CSV)
                    files.update(zip(meta.image_id, meta.file))
                f = files.get(image_id)
                if f is None:
                    self.send_error(404)
                    return
                self.send((DATA / f).read_bytes(), "image/jpeg")
            else:
                self.send_error(404)

        def do_POST(self):
            # JSON only: a plain cross-site form/text post from another web page is refused
            if self.path != "/api/label" or not self.headers.get("Content-Type", "").startswith("application/json"):
                self.send_error(400)
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(min(n, 10_000)))
            except (ValueError, json.JSONDecodeError):
                self.send_error(400)
                return
            if (not isinstance(body, dict) or not isinstance(body.get("image_id"), str)
                    or body.get("label") not in KEYS.values() or body["image_id"] not in allowed):
                self.send_error(400)
                return
            view = body.get("viewpoint", "")
            rec = {"image_id": body["image_id"], "label": body["label"], "reviewer": reviewer,
                   "vlm_shown": "1" if body.get("vlm_shown") == "1" else "0",
                   "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "viewpoint": view if view in ("up", "down") and body["label"] != REJECT else ""}
            with lock:
                for r in rows:
                    if r["image_id"] == rec["image_id"]:
                        r.update(rec)
                        break
                else:
                    rows.append(rec)
                write_csv(pd.DataFrame(rows), path)
            self.send(b"{}", "application/json")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviewer", required=True)
    ap.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--blind", action="store_true", help="hide VLM suggestions everywhere")
    ap.add_argument("--relabel-others", action="store_true")
    ap.add_argument("--conflicts", action="store_true")
    ap.add_argument("--skip-ai-rejects", action="store_true", help="leave out photos the AI annotator rejected")
    ap.add_argument("--sample", type=int, help="random subset of N images (e.g. expert check)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if args.reviewer.endswith("_ai"):
        raise SystemExit("reviewer names ending in _ai are reserved for AI annotators")

    queue, mine = build_queue(args)
    path = LABELS_DIR / f"{args.reviewer}.csv"
    print(f"{len(queue)} images in queue; labels -> {path}")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(queue, mine, path, args.reviewer))
    url = f"http://127.0.0.1:{args.port}"
    print(f"open {url}  (Ctrl+C to stop)")
    webbrowser.open(url)
    srv.serve_forever()


if __name__ == "__main__":
    main()
