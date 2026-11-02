"""Overlapping photo sets: which photos overlap, how well, and joining them into one view.

A landslide is often too big for one frame. There are two ways to cover it with several photos:
  turn  stand in ONE spot and turn the phone. Neighbouring photos then differ by a pure camera rotation, so
        a homography (3x3 projective map) moves one exactly onto the other. The set is joined into one wide
        perspective image with the same focal length and a wider view. The classifier and the size pipeline
        (dimension.py) treat it as one photo, so a landslide cut off at the frame edge can be measured whole.
  walk  walk ALONG the slope and photograph every few steps (like a photogrammetry survey). Near ground and
        the far ridge then shift by different amounts (parallax), so no homography fits and a joined picture
        would show ghosts. These sets are checked (overlap, order, spacing) and classified together (mean of
        the photos' probabilities); sizes come from the most central photo. No 3-D model is built.
A slope that is nearly flat fits a homography even when the engineer walked, so such sets are joined too.

Checks per pair (SIFT features at 1024 px, mutual-nearest matches with Lowe's ratio test, MAGSAC++):
  overlap   share of each photo covered by the other (warped outline intersection; matched-point hull when no
            homography fits), the mean of the two shares
  motion    "turn" when the homography explains the matches that epipolar geometry accepts (75th percentile
            transfer error <= parallax_px), otherwise "moved"
  roll, zoom, light  tilt, scale and brightness change between the two photos
Thresholds live in src/overlap_check.json, written by tools/overlap_eval.py from simulated sets made from
train photos (link and parallax thresholds); val photos choose how the joined picture and the single photos
are combined for the type. Test photos are never read. The video feature feeds sampled frames into analyse().
"""
import io
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

FEAT = 1024                 # long side for features
MAX_IMAGES = 8
MOSAIC_MAX = 2400           # long side of the joined picture (the size pipeline runs at about this size)
MAX_SPAN_DEG = 120.0        # a flat (perspective) picture cannot show more than this without huge stretching
DEFAULT_F35 = 26.0          # same default as dimension.py

_CFG = Path(__file__).with_name("overlap_check.json")
_DEFAULT = {
    "min_inliers": 30,          # homography inliers needed to call two photos linked
    "parallax_px": 2.5,         # 75th pct transfer error (1024 px) above which the camera moved
    "bands": {                  # overlap share: <gap red, <weak amber, <=dup_from green, > dup_from duplicate
        "turn": {"gap": 0.10, "weak": 0.25, "dup": 0.85, "ideal": [0.30, 0.60]},
        "walk": {"gap": 0.35, "weak": 0.55, "dup": 0.92, "ideal": [0.60, 0.80]},
    },
    "roll_deg": 5.0, "zoom_ratio": 1.15, "light_ratio": 1.4,
    "mosaic_weight": 0.5,       # type = w * joined picture + (1 - w) * mean of the single photos
}


def config():
    try:
        c = json.loads(_CFG.read_text())
        return {**_DEFAULT, **c, "bands": {**_DEFAULT["bands"], **c.get("bands", {})}}
    except (OSError, ValueError):
        return dict(_DEFAULT)


CFG = config()
# log-sd of the focal length by source; "joined photos" measured on simulated turning sets (tools/focal_eval.py,
# train RMS incl. bias; val 90 % coverage in src/focal_check.json)
FOCAL_SD = {"EXIF": 0.02, "typical phone camera": 0.25,      # same values as dimension.FOCAL_SD (keep in step)
            "joined photos": {"2": 0.25, "3": 0.25}}
try:
    FOCAL_SD["joined photos"].update(json.loads(Path(__file__).with_name("focal_check.json").read_text())["focal_sd"])
except (OSError, ValueError, KeyError):
    pass


def focal_sd(src, n):
    v = FOCAL_SD.get(src, 0.25)
    return float(v[str(min(max(n, 2), 3))]) if isinstance(v, dict) else float(v)


_sift = None


def _detector():
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
    return _sift


def features(img):
    """PIL image -> (gray uint8 at FEAT px, scale, keypoints xy, RootSIFT descriptors)."""
    a = np.asarray(img.convert("L"))
    s = min(1.0, FEAT / max(a.shape))
    g = cv2.resize(a, (round(a.shape[1] * s), round(a.shape[0] * s)), interpolation=cv2.INTER_AREA) if s < 1 else a
    eq = cv2.createCLAHE(2.0, (8, 8)).apply(g)          # matches survive auto-exposure changes
    kp, d = _detector().detectAndCompute(eq, None)
    if d is None or len(kp) < 8:
        return g, s, np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    d = d / (d.sum(1, keepdims=True) + 1e-7)
    d = np.sqrt(d).astype(np.float32)                   # RootSIFT (Arandjelovic & Zisserman 2012)
    return g, s, np.float32([k.pt for k in kp]), d


def _match(d1, d2, ratio=0.8):
    """Mutual nearest neighbours passing Lowe's ratio test -> index pairs."""
    if len(d1) < 8 or len(d2) < 8:
        return np.zeros((0, 2), int)
    fl = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 64})
    m12 = fl.knnMatch(d1, d2, k=2)
    m21 = fl.knnMatch(d2, d1, k=1)
    back = {m[0].queryIdx: m[0].trainIdx for m in m21 if m}
    out = [(a.queryIdx, a.trainIdx) for a, b in (p for p in m12 if len(p) == 2)
           if a.distance < ratio * b.distance and back.get(a.trainIdx) == a.queryIdx]
    return np.array(out, int).reshape(-1, 2)


def _quad(H, w, h):
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(c, H).reshape(-1, 2)


def _convex(q):
    v = np.roll(q, -1, 0) - q
    z = v[:, 0] * np.roll(v[:, 1], -1) - v[:, 1] * np.roll(v[:, 0], -1)
    return bool((z > 0).all() or (z < 0).all())


def _sane(H, w, h):
    """A plausible photo-to-photo homography: convex outline, area change < 10x, mild perspective."""
    if H is None or not np.isfinite(H).all():
        return False
    q = _quad(H, w, h)
    if not _convex(q):
        return False
    a = cv2.contourArea(q.astype(np.float32)) / (w * h)
    return 0.1 < a < 10 and abs(H[2, 0]) * w < 1.5 and abs(H[2, 1]) * h < 1.5


def _inter(q, w, h):
    """Area of the convex polygon q inside the w x h frame, and that polygon."""
    rect = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    a, p = cv2.intersectConvexConvex(q.astype(np.float32), rect)
    return float(a), (p.reshape(-1, 2) if p is not None else np.zeros((0, 2), np.float32))


def _jac(H, x, y):
    """Local linear part of H at (x, y): roll angle (deg) and scale."""
    p = cv2.perspectiveTransform(np.float32([[[x, y]], [[x + 10, y]], [[x, y + 10]]]), H).reshape(-1, 2)
    ex, ey = (p[1] - p[0]) / 10, (p[2] - p[0]) / 10
    return math.degrees(math.atan2(ex[1], ex[0])), math.sqrt(abs(ex[0] * ey[1] - ex[1] * ey[0]))


def focals_from_h(H):
    """Focal lengths (px) of the two cameras of a pure-rotation homography in centred coordinates
    (Szeliski & Shum 1997; the formula OpenCV's stitcher uses). None where it is not determined."""
    h = (H / H[2, 2]).ravel()
    out = []
    for d1, d2, v1n, v2n in ((h[6] * h[7], (h[7] - h[6]) * (h[7] + h[6]),
                              -(h[0] * h[1] + h[3] * h[4]), h[0] ** 2 + h[3] ** 2 - h[1] ** 2 - h[4] ** 2),
                             (h[0] * h[3] + h[1] * h[4], h[0] ** 2 + h[1] ** 2 - h[3] ** 2 - h[4] ** 2,
                              -h[2] * h[5], h[5] ** 2 - h[2] ** 2)):
        if abs(d1) < 1e-12 or abs(d2) < 1e-12:
            out.append(None)
            continue
        v1, v2 = v1n / d1, v2n / d2
        if v1 < v2:
            v1, v2 = v2, v1
        out.append(math.sqrt(v1 if abs(d1) > abs(d2) else v2) if v1 > 0 and v2 > 0 else
                   math.sqrt(v1) if v1 > 0 else None)
    return out[1], out[0]          # (camera of the source photo, camera of the target photo)


def pair(fa, fb):
    """Geometry between two feature sets (from features()). Coordinates are at feature scale."""
    ga, _, ka, da = fa
    gb, _, kb, db = fb
    (ha, wa), (hb, wb) = ga.shape, gb.shape
    m = _match(da, db)
    res = {"matches": len(m), "inliers": 0, "H": None, "overlap": 0.0, "share": [0.0, 0.0], "motion": "none",
           "parallax": None, "poly": None}
    if len(m) < 12:
        return res
    p, q = ka[m[:, 0]], kb[m[:, 1]]
    try:
        H, hm = cv2.findHomography(p, q, cv2.USAC_MAGSAC, 4.0, maxIters=5000, confidence=0.999)
    except cv2.error:              # degenerate matches (e.g. all on a line): no model
        H, hm = None, None
    try:
        _, fm = cv2.findFundamentalMat(p, q, cv2.USAC_MAGSAC, 1.5, 0.999, 5000)
    except cv2.error:
        fm = None
    nh = int(hm.sum()) if hm is not None else 0
    nf = int(fm.sum()) if fm is not None else 0
    geo = (fm.ravel() > 0) if fm is not None and nf >= 12 else (hm.ravel() > 0 if hm is not None else None)
    if H is not None and _sane(H, wa, ha) and nh >= 12:
        err = np.linalg.norm(cv2.perspectiveTransform(p[geo].reshape(-1, 1, 2), H).reshape(-1, 2) - q[geo], axis=1)
        par = float(np.percentile(err, 75))
        sa, poly_b = _inter(_quad(H, wa, ha), wb, hb)            # photo a inside photo b
        sb, _ = _inter(_quad(np.linalg.inv(H), wb, hb), wa, ha)  # photo b inside photo a
        share = [sb / (wa * ha), sa / (wb * hb)]
        roll, zoom = _jac(H, wa / 2, ha / 2)
        ia = cv2.blur(ga, (5, 5))[p[hm.ravel() > 0, 1].astype(int).clip(0, ha - 1), p[hm.ravel() > 0, 0].astype(int).clip(0, wa - 1)]
        ib = cv2.blur(gb, (5, 5))[q[hm.ravel() > 0, 1].astype(int).clip(0, hb - 1), q[hm.ravel() > 0, 0].astype(int).clip(0, wb - 1)]
        res.update(inliers=nh, H=H, share=share, overlap=float(np.mean(share)), parallax=par,
                   motion="turn" if par <= CFG["parallax_px"] else "moved", roll=roll, zoom=zoom,
                   light=float(np.median((ib.astype(float) + 8) / (ia.astype(float) + 8))), poly=poly_b)
    elif nf >= 12:            # no homography: overlap from the matched-point outlines (an underestimate)
        g = fm.ravel() > 0
        sh = [cv2.contourArea(cv2.convexHull(p[g])) / (wa * ha), cv2.contourArea(cv2.convexHull(q[g])) / (wb * hb)]
        res.update(inliers=nf, share=sh, overlap=float(np.mean(sh)), motion="moved", parallax=float("inf"))
    return res


def stereo_points(ref, other, f_px, baseline_m, baseline_sd=0.15, min_angle_deg=1.0):
    """Metric 3-D points of a walk-along pair (camera moved between the photos), seen from `ref`.
    Essential matrix (MAGSAC++) with K from the focal length, pose by cheirality (recoverPose), linear
    triangulation; the unit-length translation is scaled by the walked distance between the two camera
    positions (baseline_m = steps x stride). Kept points: in front of both cameras, reprojection < 2 px,
    ray angle > min_angle_deg (smaller angles give noise-dominated depth).
    Returns {"xy": N x 2 pixel coords in ref (full resolution), "z": N metric depths in the ref camera,
    "log_sd": baseline_sd, "n": N, "angle_med": deg, "inliers": E inliers} or None."""
    fr, fo = features(ref), features(other)
    g, s, kr, dr = fr
    h, w = g.shape
    m = _match(dr, fo[3])
    if len(m) < 20 or not baseline_m or baseline_m <= 0:
        return None
    p, q = kr[m[:, 0]].astype(np.float64), fo[2][m[:, 1]].astype(np.float64)
    # each photo its own intrinsics (sizes/orientation may differ): same 35 mm-equivalent focal, centre point
    fs1 = f_px * s
    fs2 = f_px * max(other.size) / max(ref.size) * fo[1]
    h2, w2 = fo[0].shape
    # map the other photo's points into the ref camera's pixel frame (q' = K1 K2^-1 q): one K, pixel-unit
    # thresholds for MAGSAC (it is unstable with tiny normalised thresholds)
    q = (q - [w2 / 2, h2 / 2]) * (fs1 / fs2) + [w / 2, h / 2]
    K = np.array([[fs1, 0, w / 2], [0, fs1, h / 2], [0, 0, 1.0]])
    pn = (p - [w / 2, h / 2]) / fs1
    qn = (q - [w / 2, h / 2]) / fs1
    I3 = np.eye(3)
    try:
        E, em = cv2.findEssentialMat(p, q, K, cv2.USAC_MAGSAC, 0.999, 1.0)
        if E is None or E.shape != (3, 3):
            return None
        _, R, t, pm = cv2.recoverPose(E, p, q, K, mask=em.copy())
    except cv2.error:
        return None
    inl = pm.ravel() > 0
    if inl.sum() < 20:
        return None
    P1 = np.hstack([I3, np.zeros((3, 1))])
    P2 = np.hstack([R, t])
    X = cv2.triangulatePoints(P1, P2, pn[inl].T, qn[inl].T)
    X = (X[:3] / X[3]).T                                   # in ref camera, baseline units (|t| = 1)
    X2 = (R @ X.T + t).T
    c2 = (-R.T @ t).ravel()                                # other camera centre in ref frame
    r1, r2 = X, X - c2
    ang = np.degrees(np.arccos(np.clip((r1 * r2).sum(1) / (np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1) + 1e-12), -1, 1)))

    def proj(P, Y):
        u = (P @ np.hstack([Y, np.ones((len(Y), 1))]).T).T
        return u[:, :2] / u[:, 2:3]
    err = np.maximum(np.linalg.norm(proj(P1, X) - pn[inl], axis=1) * fs1,        # in pixels of each photo
                     np.linalg.norm(proj(P2, X) - qn[inl], axis=1) * fs2)
    keep = (X[:, 2] > 0) & (X2[:, 2] > 0) & (err < 2.0) & (ang > min_angle_deg)
    if keep.sum() < 8:
        return None
    return {"xy": (p[inl][keep] / s).tolist(), "z": (X[keep, 2] * baseline_m).tolist(), "log_sd": float(baseline_sd),
            "n": int(keep.sum()), "angle_med": float(np.median(ang[keep])), "inliers": int(inl.sum())}


def _components(n, links):
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in links:
        parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def _mst(nodes, links, weight):
    """Maximum spanning tree (Kruskal) over the given links inside one component."""
    parent = {i: i for i in nodes}

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    tree = []
    for (i, j) in sorted(links, key=lambda e: -weight[e]):
        if i in parent and j in parent and find(i) != find(j):
            parent[find(i)] = find(j)
            tree.append((i, j))
    return tree


def _path_order(nodes, tree, pairs, feats):
    """Order a walk set: longest path of the tree, direction = left to right by mean point shift."""
    if len(nodes) == 1:
        return list(nodes)
    adj = {i: [] for i in nodes}
    for i, j in tree:
        adj[i].append(j)
        adj[j].append(i)

    def far(s):
        dist, prev, todo = {s: 0}, {s: None}, [s]
        while todo:
            u = todo.pop()
            for v in adj[u]:
                if v not in dist:
                    dist[v], prev[v] = dist[u] + 1, u
                    todo.append(v)
        e = max(dist, key=dist.get)
        return e, prev
    a, _ = far(nodes[0])
    b, prev = far(a)
    path = [b]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    rest = [i for i in nodes if i not in path]
    for i in rest:                     # side branches go next to their tree neighbour
        k = next(path.index(v) for v in adj[i] if v in path) if any(v in path for v in adj[i]) else len(path) - 1
        path.insert(k + 1, i)
    # direction: content moves left in the next photo when the camera moves right
    i, j = path[0], path[1]
    pr = pairs.get((min(i, j), max(i, j)))
    if pr and pr.get("dx") is not None and ((pr["dx"] > 0) == (i < j)):
        path.reverse()
    return path


def _gains(warped, masks):
    """Exposure compensation (Brown & Lowe 2007; OpenCV GainCompensator formulation), one gain per photo."""
    n = len(warped)
    small = [cv2.resize(cv2.cvtColor(w, cv2.COLOR_RGB2GRAY), None, fx=0.25, fy=0.25).astype(float) for w in warped]
    ms = [cv2.resize(m, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST) > 0 for m in masks]
    A, b = np.zeros((n, n)), np.zeros(n)
    sn, sg = 10.0, 0.1
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            o = ms[i] & ms[j]
            N = int(o.sum())
            if N < 50:
                continue
            Ii, Ij = small[i][o].mean(), small[j][o].mean()
            A[i, i] += N * (2 * Ii * Ii / sn ** 2 + 1 / sg ** 2)
            A[i, j] -= N * 2 * Ii * Ij / sn ** 2
            b[i] += N / sg ** 2
    for i in range(n):
        if A[i, i] == 0:
            A[i, i], b[i] = 1, 1
    g = np.linalg.solve(A, b)
    return np.clip(g, 0.5, 2.0)


def _trim(valid):
    """Shave borders that are mostly empty (the jagged edges of a joined picture)."""
    t, b, l, r = 0, valid.shape[0], 0, valid.shape[1]
    while b - t > 64 and r - l > 64:
        rows = (valid[t, l:r].mean(), valid[b - 1, l:r].mean())
        cols = (valid[t:b, l].mean(), valid[t:b, r - 1].mean())
        worst = min(rows + cols)
        if worst >= 0.6:
            break
        k = (rows + cols).index(worst)
        step = max(1, (b - t if k < 2 else r - l) // 200)
        if k == 0:
            t += step
        elif k == 1:
            b -= step
        elif k == 2:
            l += step
        else:
            r -= step
    return t, b, l, r


def mosaic(images, Hs, ref, f_px):
    """Join the photos into one perspective picture in photo `ref`'s camera.
    Hs[i]: full-resolution homography photo i -> photo ref. f_px: focal length of the photos (px)."""
    W0, H0 = images[ref].size
    cx, cy = W0 / 2, H0 / 2
    quads = {i: _quad(H, *images[i].size) for i, H in Hs.items()}
    pts = np.concatenate(list(quads.values()))
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    lim = f_px * math.tan(math.radians(MAX_SPAN_DEG / 2))
    cropped = x0 < cx - lim or x1 > cx + lim or y0 < cy - lim or y1 > cy + lim
    x0, x1 = max(x0, cx - lim), min(x1, cx + lim)
    y0, y1 = max(y0, cy - lim), min(y1, cy + lim)
    m = min(1.0, MOSAIC_MAX / max(x1 - x0, y1 - y0))
    Wm, Hm = int(math.ceil((x1 - x0) * m)), int(math.ceil((y1 - y0) * m))
    T = np.array([[m, 0, -x0 * m], [0, m, -y0 * m], [0, 0, 1]])
    warped, masks = [], []
    for i, H in Hs.items():
        a = np.asarray(images[i].convert("RGB"))
        warped.append(cv2.warpPerspective(a, T @ H, (Wm, Hm), flags=cv2.INTER_LINEAR))
        masks.append(cv2.warpPerspective(np.full(a.shape[:2], 255, np.uint8), T @ H, (Wm, Hm), flags=cv2.INTER_NEAREST))
    gains = _gains(warped, masks)
    acc, wsum = np.zeros((Hm, Wm, 3)), np.zeros((Hm, Wm))
    for w, mk, g in zip(warped, masks, gains):
        wt = cv2.distanceTransform((mk > 0).astype(np.uint8), cv2.DIST_L2, 3) ** 2   # feather: centre counts most
        acc += w.astype(float) * g * wt[..., None]
        wsum += wt
    valid = wsum > 0
    out = np.clip(acc / np.maximum(wsum, 1e-6)[..., None], 0, 255).astype(np.uint8)
    t, b, l, r = _trim(valid)
    out, valid = out[t:b, l:r], valid[t:b, l:r]
    if not valid.all():          # fill the empty corners with a soft extension so models see no black wedges
        k = 8
        size = (max(1, out.shape[1] // k), max(1, out.shape[0] // k))
        vf = valid.astype(np.float32)
        num = cv2.resize(out.astype(np.float32) * vf[..., None], size, interpolation=cv2.INTER_AREA)
        den = cv2.resize(vf, size, interpolation=cv2.INTER_AREA)
        sm = np.clip(num / np.maximum(den, 1e-3)[..., None], 0, 255).astype(np.uint8)   # no black bleeding in
        sv = (den > 0.5).astype(np.uint8)
        fill = cv2.inpaint(sm, (1 - sv).astype(np.uint8), 3, cv2.INPAINT_TELEA)
        fill = cv2.GaussianBlur(cv2.resize(fill, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_LINEAR), (0, 0), 6)
        out = np.where(valid[..., None], out, fill)
    shift = np.array([[1, 0, -l], [0, 1, -t], [0, 0, 1]]) @ T
    span = [math.degrees(math.atan((x1 - cx) / f_px) - math.atan((x0 - cx) / f_px)),
            math.degrees(math.atan((y1 - cy) / f_px) - math.atan((y0 - cy) / f_px))]
    return {"image": Image.fromarray(out), "T": shift, "f_px": f_px * m, "valid": float(valid.mean()),
            "span_deg": [round(span[0], 1), round(span[1], 1)], "cropped": bool(cropped),
            "gains": [round(float(g), 3) for g in gains]}


def jpeg(img, f35=None, quality=90):
    """JPEG bytes; the joined picture carries its own 35 mm-equivalent focal length in EXIF, so the size
    pipeline (dimension.focal_35mm) knows how wide the view is."""
    buf = io.BytesIO()
    ex = Image.Exif()
    if f35:
        ex.get_ifd(0x8769)[0xA405] = int(round(min(max(f35, 1), 65535)))
    img.save(buf, "JPEG", quality=quality, exif=ex)
    return buf.getvalue()


def analyse(images, kind="turn", f35=None, stitch=True):
    """Check an overlapping set (2-8 PIL photos in the order taken). Returns (report, extras).
    report is JSON for the phone; extras holds the joined picture and the members for classification."""
    import time
    t0 = time.time()
    n = len(images)
    if not 2 <= n <= MAX_IMAGES:
        raise ValueError(f"2-{MAX_IMAGES} photos")
    kind = kind if kind in ("turn", "walk") else "turn"
    feats = [features(im) for im in images]
    pairs = {}
    for i in range(n):
        for j in range(i + 1, n):
            pr = pair(feats[i], feats[j])
            if pr["H"] is not None:   # mean content shift (feature px of photo j) for walk ordering
                ga = feats[i][0]
                c = cv2.perspectiveTransform(np.float32([[[ga.shape[1] / 2, ga.shape[0] / 2]]]), pr["H"]).ravel()
                pr["dx"] = float(c[0] - feats[j][0].shape[1] / 2) / feats[j][0].shape[1]
            pairs[(i, j)] = pr
    links = [e for e, p in pairs.items() if p["inliers"] >= CFG["min_inliers"] and p["overlap"] >= 0.03]
    comps = _components(n, links)
    main = comps[0]
    weight = {e: pairs[e]["inliers"] for e in links}
    tree = _mst(main, [e for e in links if e[0] in main and e[1] in main], weight)
    turn_links = [e for e in tree if pairs[e]["motion"] == "turn"]
    detected = "turn" if tree and len(turn_links) == len(tree) else "walk" if tree else kind
    h_ok = bool(tree) and all(pairs[e]["H"] is not None for e in tree)
    # the focal length: EXIF from the phone, else from the rotation homographies, else a typical phone
    W, Hh = images[main[0]].size
    f_src, f_px = "typical phone camera", DEFAULT_F35 / 36 * max(W, Hh)
    fx = [v for v in (f35 or []) if v and 8 <= float(v) <= 200]
    if fx:
        f_src, f_px = "EXIF", float(np.median(fx)) / 36 * max(W, Hh)
    elif turn_links:
        est = []
        for i, j in turn_links:
            si, sj = feats[i][1], feats[j][1]
            (hi, wi), (hj, wj) = feats[i][0].shape, feats[j][0].shape
            Hc = np.array([[1, 0, -wj / 2], [0, 1, -hj / 2], [0, 0, 1]]) @ pairs[(i, j)]["H"] @ \
                np.array([[1, 0, wi / 2], [0, 1, hi / 2], [0, 0, 1]])
            fa, fb = focals_from_h(Hc)
            est += [fa / si, fb / sj] if fa and fb else []
        if est:
            fe = float(np.median(est))
            fd = DEFAULT_F35 / 36 * max(W, Hh)
            if 0.5 * fd < fe < 2.5 * fd:
                f_src, f_px = "joined photos", fe
    # full-resolution homographies into the reference (the photo with the strongest links)
    out_m, Hs, ref, order = None, None, main[0], list(main)
    if len(main) >= 2 and h_ok:
        deg = {i: sum(weight[e] for e in tree if i in e) for i in main}
        ref = max(main, key=deg.get)
        S = {i: np.diag([feats[i][1], feats[i][1], 1.0]) for i in main}
        Hfeat = {ref: np.eye(3)}
        todo = [ref]
        while todo:
            u = todo.pop()
            for (i, j) in tree:
                if u in (i, j):
                    v = j if u == i else i
                    if v in Hfeat:
                        continue
                    Huv = pairs[(i, j)]["H"] if v == i else np.linalg.inv(pairs[(i, j)]["H"])   # v -> u
                    Hfeat[v] = Hfeat[u] @ Huv
                    todo.append(v)
        Hs = {i: np.linalg.inv(S[ref]) @ Hfeat[i] @ S[i] for i in main}
        cen = {i: cv2.perspectiveTransform(np.float32([[[images[i].size[0] / 2, images[i].size[1] / 2]]]), Hs[i]).ravel()
               for i in main}
        spread = np.ptp(np.array(list(cen.values())), axis=0)
        axis = 0 if spread[0] >= spread[1] else 1
        order = sorted(main, key=lambda i: cen[i][axis])
        if stitch and (detected == "turn" or kind == "turn" or max(pairs[e]["parallax"] for e in tree) <= 2 * CFG["parallax_px"]):
            out_m = mosaic(images, Hs, ref, f_px)
    elif len(main) >= 2:
        order = _path_order(main, tree, pairs, feats)
    order += [i for c in comps[1:] for i in c]

    # issues, with the photo indices (the phone numbers them by position in `order`)
    B = CFG["bands"]["walk" if (detected == "walk" and out_m is None) else kind]
    issues, status = [], {}
    pos = {i: k for k, i in enumerate(order)}
    for c in comps[1:]:
        if len(c) == 1:
            issues.append({"code": "unmatched", "sev": "bad", "a": c[0]})
        else:
            a = min(c, key=lambda i: min(abs(i - m) for m in main))
            b = min(main, key=lambda m: abs(m - a))
            issues.append({"code": "gap", "sev": "bad", "a": min(a, b), "b": max(a, b), "group": c})
    for (i, j) in sorted(tree, key=lambda e: min(pos[e[0]], pos[e[1]])):
        p = pairs[(i, j)]
        a, b = (i, j) if pos[i] < pos[j] else (j, i)
        ov = p["overlap"]
        s = "gap" if ov < B["gap"] else "weak" if ov < B["weak"] else "dup" if ov > B["dup"] else "good"
        status[(i, j)] = s
        if s in ("gap", "weak"):
            issues.append({"code": s, "sev": "bad" if s == "gap" else "warn", "a": a, "b": b, "val": round(ov, 3)})
        if s == "dup":
            issues.append({"code": "dup", "sev": "info", "a": a, "b": b, "val": round(ov, 3)})
        if kind == "turn" and p["motion"] == "moved":
            issues.append({"code": "moved", "sev": "warn", "a": a, "b": b,
                           "val": round(p["parallax"], 1) if math.isfinite(p["parallax"]) else None})
        if p.get("roll") is not None and abs(p["roll"]) > CFG["roll_deg"]:
            issues.append({"code": "roll", "sev": "warn", "a": a, "b": b, "val": round(abs(p["roll"]), 1)})
        if p.get("zoom") is not None and max(p["zoom"], 1 / p["zoom"]) > CFG["zoom_ratio"] and kind == "turn":
            issues.append({"code": "zoom", "sev": "warn", "a": a, "b": b, "val": round(max(p["zoom"], 1 / p["zoom"]), 2)})
        if p.get("light") is not None and max(p["light"], 1 / p["light"]) > CFG["light_ratio"]:
            issues.append({"code": "light", "sev": "warn", "a": a, "b": b, "val": round(max(p["light"], 1 / p["light"]), 2)})
    if kind == "walk" and detected == "turn":
        issues.append({"code": "turned", "sev": "info"})
    if kind == "turn" and detected == "walk" and out_m is None and len(main) >= 2:
        issues.append({"code": "not_joined", "sev": "warn"})
    if out_m and out_m["cropped"]:
        issues.append({"code": "wide", "sev": "warn", "val": round(max(out_m["span_deg"]), 0)})
    ovs = [pairs[e]["overlap"] for e in tree]
    if (detected == "walk" or kind == "walk") and len(ovs) >= 2 and max(ovs) > 1.8 * min(ovs) and min(ovs) > 0:
        issues.append({"code": "spacing", "sev": "warn", "val": [round(min(ovs), 2), round(max(ovs), 2)]})
    if order[:len(main)] != sorted(main):
        issues.append({"code": "order", "sev": "info"})
    bad = any(x["sev"] == "bad" for x in issues)
    warn = any(x["sev"] == "warn" for x in issues)
    verdict = "fix" if len(main) < 2 else "warn" if bad or warn else "ready"

    # what the phone draws: the joined preview with each photo's outline and each link's shared area
    prev = None
    if out_m:
        im = out_m["image"]
        k = min(1.0, 960 / max(im.size))
        pv = im.resize((round(im.size[0] * k), round(im.size[1] * k)), Image.LANCZOS)
        P = np.diag([k, k, 1.0]) @ out_m["T"]
        quads = {i: _quad(P @ Hs[i], *images[i].size) for i in main}
        shared = {}
        for (i, j) in tree:
            a, poly = cv2.intersectConvexConvex(quads[i].astype(np.float32), quads[j].astype(np.float32))
            shared[f"{i}-{j}"] = poly.reshape(-1, 2).round(1).tolist() if poly is not None else []
        buf = io.BytesIO()
        pv.save(buf, "JPEG", quality=78)
        import base64
        prev = {"src": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
                "w": pv.size[0], "h": pv.size[1],
                "quads": {str(i): q.round(1).tolist() for i, q in quads.items()}, "shared": shared}
    f35_out = None
    if out_m:
        f35_out = out_m["f_px"] / max(out_m["image"].size) * 36
    report = {
        "n": n, "kind": kind, "detected": detected, "order": order, "main": sorted(main), "ref": ref,
        "verdict": verdict, "joinable": len(main) >= 2, "joined": out_m is not None, "issues": issues,
        "links": [{"a": i, "b": j, "overlap": round(pairs[(i, j)]["overlap"], 3), "share": [round(v, 3) for v in pairs[(i, j)]["share"]],
                   "inliers": pairs[(i, j)]["inliers"], "motion": pairs[(i, j)]["motion"], "status": status.get((i, j), "good"),
                   "parallax": (round(pairs[(i, j)]["parallax"], 2) if pairs[(i, j)]["parallax"] is not None and
                                math.isfinite(pairs[(i, j)]["parallax"]) else None),
                   "roll": round(pairs[(i, j)].get("roll") or 0, 1), "zoom": round(pairs[(i, j)].get("zoom") or 1, 3),
                   "light": round(pairs[(i, j)].get("light") or 1, 3)} for (i, j) in tree],
        "pairs": [{"a": i, "b": j, "overlap": round(p["overlap"], 3), "inliers": p["inliers"]}
                  for (i, j), p in pairs.items() if p["inliers"] >= 12],
        "bands": B, "focal_source": f_src, "focal_35mm": round(f_px / max(W, Hh) * 36, 1),
        "focal_sd": round(focal_sd(f_src, len(main)), 3),
        "span_deg": out_m["span_deg"] if out_m else None, "valid": round(out_m["valid"], 3) if out_m else None,
        "preview": prev, "ms": int((time.time() - t0) * 1000),
    }
    extras = {"mosaic": out_m["image"] if out_m else None, "mosaic_f35": f35_out, "main": sorted(main),
              "ref": ref, "central": max(main, key=lambda i: sum(pairs[e]["overlap"] for e in tree if i in e)) if tree else main[0]}
    return report, extras


def combine(r_mosaic, r_singles, w=None):
    """Type probabilities for a set: w * joined picture + (1 - w) * mean of the single photos (w from val)."""
    w = CFG["mosaic_weight"] if w is None else w
    if r_mosaic is None:
        w = 0.0
    classes = [c for c, _ in (r_mosaic or r_singles[0])["ranked"]]
    mean = {c: float(np.mean([dict(r["ranked"])[c] for r in r_singles])) if r_singles else 0.0 for c in classes}
    m = dict(r_mosaic["ranked"]) if r_mosaic else {}
    probs = {c: w * m.get(c, 0.0) + (1 - w) * mean[c] for c in classes} if r_singles else m
    us = [r["usable"] for r in r_singles]
    usable = (w * r_mosaic["usable"] if r_mosaic else 0.0) + (1 - w) * (float(np.mean(us)) if us else 0.0) \
        if r_singles else r_mosaic["usable"]
    return probs, usable
