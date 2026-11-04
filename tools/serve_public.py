"""Put the app on a free public https link (Cloudflare quick tunnel) while this laptop runs it.

    python tools/serve_public.py                 # GPU app + sizes, photos not saved
    python tools/serve_public.py --save          # keep uploads in data/app_uploads (your own field tests)
    python tools/serve_public.py --no-dims       # type only (less GPU memory)

The app itself listens on 127.0.0.1 only; cloudflared carries the https traffic to it. cloudflared is a general
tool, so it lives outside the project: E:\cloudflared\cloudflared.exe (official release 2026.10.2, sha256
12c7c887...b70f199), or set CLOUDFLARED=<path>, or have it on PATH. The link
https://<random words>.trycloudflare.com changes on every start and works until you press Ctrl+C.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLOUDFLARED = Path(os.environ.get("CLOUDFLARED") or (r"E:\cloudflared\cloudflared.exe" if Path(r"E:\cloudflared\cloudflared.exe").exists()
                   else shutil.which("cloudflared") or r"E:\cloudflared\cloudflared.exe"))
LOGS = ROOT / "results" / "serve_public"


def wait_ready(url, proc, limit=600):
    t0 = time.time()
    while time.time() - t0 < limit:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--save", action="store_true", help="keep uploaded photos (default: not saved)")
    ap.add_argument("--no-dims", action="store_true", help="no size estimate")
    a = ap.parse_args()
    if not CLOUDFLARED.exists():
        sys.exit(f"missing {CLOUDFLARED} (download cloudflared-windows-amd64.exe from "
                 "github.com/cloudflare/cloudflared/releases, save it there or set CLOUDFLARED)")
    LOGS.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(ROOT / "src" / "app.py"), "--host", "127.0.0.1", "--port", str(a.port)]
    cmd += [] if a.save else ["--no-save"]
    cmd += ["--no-dims"] if a.no_dims else []
    print("starting the app (loading models, about a minute) ...", flush=True)
    app = subprocess.Popen(cmd, cwd=ROOT, stdout=open(LOGS / "app.log", "w", encoding="utf-8"),
                           stderr=subprocess.STDOUT)
    tun = None
    try:
        if not wait_ready(f"http://127.0.0.1:{a.port}/", app):
            sys.exit(f"the app did not start, see {LOGS / 'app.log'}")
        print("app ready; opening the tunnel ...", flush=True)
        tun = subprocess.Popen([str(CLOUDFLARED), "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", f"http://127.0.0.1:{a.port}"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                               errors="replace")
        found, url = threading.Event(), [None]
        log = open(LOGS / "tunnel.log", "w", encoding="utf-8")

        def read():
            for line in tun.stdout:
                log.write(line); log.flush()
                m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
                if m:
                    url[0] = m.group(0)
                if "Registered tunnel connection" in line and url[0] and not found.is_set():
                    found.set()   # print the link only once the tunnel is really connected
                    print("\n  PUBLIC LINK (open on any phone):  " + url[0] + "\n"
                          "  works while this window stays open; Ctrl+C stops it.\n"
                          + ("  uploaded photos ARE saved to data/app_uploads\n" if a.save else
                             "  uploaded photos are not saved\n"), flush=True)
        threading.Thread(target=read, daemon=True).start()
        if not found.wait(90):
            sys.exit(f"the tunnel did not connect within 90 s, see {LOGS / 'tunnel.log'}")
        while app.poll() is None and tun.poll() is None:
            time.sleep(1)
        print("stopped: " + ("the app exited" if app.poll() is not None else "the tunnel exited"))
    except KeyboardInterrupt:
        print("stopping ...")
    finally:
        for p in (tun, app):
            if p and p.poll() is None:
                p.terminate()
                try:
                    p.wait(20)
                except subprocess.TimeoutExpired:
                    p.kill()


if __name__ == "__main__":
    main()
