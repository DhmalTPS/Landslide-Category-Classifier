"""Pixel statistics and geometry for src/dimension.py (no neural networks in this file).

The deep-learning models in dimension.py propose: depth maps, outline probabilities and grain instances.
The functions here check, fuse and measure them using plain image statistics and robust estimators,
so that no single network decides a number on its own.
"""
import numpy as np
from scipy import ndimage, optimize, special, stats


# ----------------------------------------------------------------------------- depth
def box(x, r):
    return ndimage.uniform_filter(x, 2 * r + 1, mode="reflect")


def guided_filter(guide, src, r, eps):
    """He et al. (2010) guided filter. It smooths src (log-depth) but keeps the edges present in guide (the
    photo), so the scarp and block faces stay sharp while depth speckle is removed."""
    mI, mp = box(guide, r), box(src, r)
    cov = box(guide * src, r) - mI * mp
    var = box(guide * guide, r) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return box(a, r) * guide + box(b, r)


def fuse_depths(logds, prior_sd=0.3, region=None):
    """Per-pixel inverse-variance fusion of several depth models. Each model's entry is a stack of test-time
    augmented log-depth maps (n_aug, H, W). Its variance across augmentations measures how stable it is at
    that pixel, and prior_sd is the typical zero-shot metric error of a single model. Returns the fused
    log-depth, the per-pixel total variance (within + between models) and each model's mean map."""
    # SHAPE from every model, global SCALE from the median model. Monocular models differ mostly by an overall
    # scale factor (up to ~5x on landslide photos), so each model is shifted to the median of the models'
    # log-scales before the per-pixel fusion: one biased model cannot drag the scale (breakdown point of a median).
    region = np.ones(logds[0].shape[1:], bool) if region is None else region
    raw = [st.mean(0) for st in logds]
    scales = np.array([np.median(m[region]) for m in raw])
    s_star = float(np.median(scales))
    means, ws = [], []
    for st, m, sc in zip(logds, raw, scales):
        v = st.var(0) if len(st) > 1 else np.zeros(st.shape[1:])
        means.append(m - sc + s_star)
        ws.append(1.0 / (v + prior_sd ** 2))
    means, ws = np.stack(means), np.stack(ws)
    fused = (ws * means).sum(0) / ws.sum(0)
    between = (ws * (means - fused) ** 2).sum(0) / ws.sum(0)
    scale_sd = float(1.4826 * np.median(np.abs(scales - s_star))) if len(scales) > 2 else float(np.std(scales))
    return fused, 1.0 / ws.sum(0) + between, raw, scale_sd


# ----------------------------------------------------------------------------- outline
def bad_prior(img):
    """Soft probability that a pixel is vegetation or sky, from pixel statistics alone. Vegetation uses the
    Excess-Green index 2g - r - b (chromatic coordinates), with a sigmoid centred on its Otsu threshold.
    Sky is bright, blue-dominant and smooth."""
    from skimage.filters import threshold_otsu
    a = np.asarray(img, dtype=np.float32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    exg = (2 * G - R - B) / (R + G + B + 1e-6)
    try:
        t = max(0.05, float(threshold_otsu(exg[::4, ::4])))
    except ValueError:
        t = 0.05
    veg = special.expit((exg - t) / 0.02)       # logistic, overflow-safe
    gray = a.mean(-1)
    sd = np.sqrt(np.maximum(box(gray ** 2, 4) - box(gray, 4) ** 2, 0))
    sky = ((B > R + 12) & (B >= G) & (gray > 140)).astype(np.float32) / (1 + np.exp((sd - 6) / 1.5))
    return np.clip(np.maximum(veg, sky), 0, 1)


def graph_cut(prob, img, logz, lam=2.0, max_side=512):
    """Binary MRF segmentation by max-flow. The unary term is -log P (fused deep-learning probability). The
    pairwise term follows the Boykov-Jolly / GrabCut contrast weight exp(-beta |dI|^2), multiplied by a depth
    weight exp(-|dlogz|^2 / 2 s^2), so the cut prefers boundaries on colour edges AND depth jumps (scarp
    crests, deposit fronts). beta and s are set from the image's own mean squared differences."""
    import maxflow
    H, W = prob.shape
    k = max(1, int(np.ceil(max(H, W) / max_side)))
    sl = (slice(None, None, k), slice(None, None, k))
    P = np.clip(prob[sl], 1e-4, 1 - 1e-4)
    I = np.asarray(img, dtype=np.float32)[sl] / 255.0
    Z = logz[sl]

    def w(axis):
        dI = np.sum(np.diff(I, axis=axis) ** 2, -1)
        dZ = np.diff(Z, axis=axis) ** 2
        beta = 1 / (2 * dI.mean() + 1e-8)
        s2 = 2 * dZ.mean() + 1e-8
        e = lam * np.exp(-beta * dI) * np.exp(-dZ / s2)
        return np.pad(e, [(0, 1), (0, 0)] if axis == 0 else [(0, 0), (0, 1)])

    g = maxflow.Graph[float]()
    ids = g.add_grid_nodes(P.shape)
    g.add_grid_edges(ids, weights=w(1), structure=np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]), symmetric=True)
    g.add_grid_edges(ids, weights=w(0), structure=np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]), symmetric=True)
    g.add_grid_tedges(ids, -np.log(1 - P), -np.log(P))   # source = landslide
    g.maxflow()
    fg = ~g.get_grid_segments(ids)
    if k > 1:
        fg = np.repeat(np.repeat(fg, k, 0), k, 1)[:H, :W]
    return fg


def graph_cut_stable(prob, img, logz, lams=(0.5, 1.0, 2.0, 4.0, 8.0)):
    """Stability selection of the smoothness weight. Run the cut over a lambda ladder and pick the lambda whose
    mask agrees best with both neighbours (IoU), i.e. the middle of the most stable plateau, so no single
    hand-set lambda decides the outline. Returns the mask, the chosen lambda and its stability."""
    masks = [graph_cut(prob, img, logz, lam=l) for l in lams]
    iou = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)
    nb = [iou(masks[i], masks[i + 1]) for i in range(len(masks) - 1)]
    # interior rungs only, each judged by its WORSE neighbour (ends have one neighbour and would always win)
    score = {i: min(nb[i - 1], nb[i]) for i in range(1, len(masks) - 1)}
    valid = [i for i in score if 0.005 < masks[i].mean() < 0.92]
    if not valid:
        return masks[len(masks) // 2], lams[len(masks) // 2], 0.0
    k = max(valid, key=lambda i: (round(score[i], 3), -abs(i - len(masks) // 2)))
    return masks[k], lams[k], float(score[k])


def clean_mask(m, min_frac=0.05, near_px=None):
    """Fill holes, remove specks, keep the main region and big pieces near it."""
    m = ndimage.binary_fill_holes(ndimage.binary_opening(m, iterations=2))
    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    main = int(np.argmax(sizes)) + 1
    near = ndimage.distance_transform_edt(lab != main) < (near_px or 0.1 * max(m.shape))
    keep = [i + 1 for i, s in enumerate(sizes) if i + 1 == main or (s > min_frac * sizes.max() and near[lab == i + 1].any())]
    return np.isin(lab, keep)


# ----------------------------------------------------------------------------- geometry
def robust_plane(P, iters=200, seed=0):
    """Slope plane by RANSAC (threshold = 2.5 x MAD of the least-squares residuals), then a least-squares refit on
    the inliers. Returns the centre, the normal facing the camera and the inlier fraction."""
    rng = np.random.default_rng(seed)
    c0 = np.median(P, 0)
    _, _, vt = np.linalg.svd(P - c0, full_matrices=False)
    res = np.abs((P - c0) @ vt[2])
    thr = max(2.5 * 1.4826 * np.median(res), 1e-3)
    best, best_n, anchor = vt[2], (res < thr).sum(), c0
    sub = P[rng.choice(len(P), min(len(P), 5000), replace=False)]
    for _ in range(iters):
        a, b, c = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        k = (np.abs((P - a) @ n) < thr).sum()
        if k > best_n:
            best, best_n, anchor = n, k, a
    inl = np.abs((P - anchor) @ best) < thr
    c = P[inl].mean(0)
    _, _, vt = np.linalg.svd(P[inl] - c, full_matrices=False)
    n = vt[2] if vt[2][2] < 0 else -vt[2]
    return c, n, float(inl.mean())


def backproject(us, vs, z, W, H, f):
    return np.stack([(us - W / 2) * z / f, (vs - H / 2) * z / f, z], 1)


def prefailure_volume(depth, mask, f, W, H, c, across, down, normal, avoid=None, seed=0):
    """Volume against a reconstructed pre-failure surface. The intact ground in a ring around the landslide
    is fitted with a thin-plate spline h(a, d), giving the height above the slope plane over in-plane
    coordinates. The smoothing is chosen by 5-fold cross-validation. Interpolated under the landslide, it
    gives the surface before failure. Depleted volume = sum of (h_pre - h_now)+ and accumulated volume =
    sum of (h_now - h_pre)+, each times the in-plane pixel area. Valid only when the ring surrounds the
    landslide (angular coverage >= 50 %). The CV error gives the uncertainty."""
    from scipy.interpolate import RBFInterpolator
    rng = np.random.default_rng(seed)
    r = max(8, int(0.04 * max(W, H)))
    ring = ndimage.binary_dilation(mask, iterations=r) & ~ndimage.binary_dilation(mask, iterations=2)
    if avoid is not None:
        ring &= ~avoid
    rv, ru = np.nonzero(ring)
    if len(ru) < 200:
        return None
    sel = rng.choice(len(ru), min(1500, len(ru)), replace=False)
    Pr = backproject(ru[sel], rv[sel], depth[rv[sel], ru[sel]], W, H, f) - c
    X, h = np.stack([Pr @ across, Pr @ down], 1), Pr @ normal
    mv, mu = np.nonzero(mask)
    selm = rng.choice(len(mu), min(6000, len(mu)), replace=False)
    Pm = backproject(mu[selm], mv[selm], depth[mv[selm], mu[selm]], W, H, f) - c
    Xm, hm = np.stack([Pm @ across, Pm @ down], 1), Pm @ normal
    ang = np.arctan2(X[:, 1] - Xm[:, 1].mean(), X[:, 0] - Xm[:, 0].mean())
    coverage = len(np.unique(((ang + np.pi) / (2 * np.pi) * 12).astype(int) % 12)) / 12
    if coverage < 0.5:
        return {"available": False, "coverage": round(coverage, 2)}
    folds = rng.integers(0, 5, len(X))
    best = None
    scale = float(np.var(h) * len(X)) + 1e-9
    for s in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        err = []
        for k in range(5):
            tr, va = folds != k, folds == k
            if tr.sum() < 50 or va.sum() < 10:
                continue
            m = RBFInterpolator(X[tr], h[tr], kernel="thin_plate_spline", smoothing=s * scale)
            err.append(np.mean((m(X[va]) - h[va]) ** 2))
        e = float(np.mean(err)) if err else np.inf
        if best is None or e < best[0]:
            best = (e, s)
    model = RBFInterpolator(X, h, kernel="thin_plate_spline", smoothing=best[1] * scale)
    dh = model(Xm) - hm                                         # > 0: ground is now below the old surface
    zz = depth[mv[selm], mu[selm]]
    rays = np.stack([(mu[selm] - W / 2) / f, (mv[selm] - H / 2) / f, np.ones_like(zz)], 1)
    cos = np.clip(np.abs(rays @ normal) / np.linalg.norm(rays, axis=1), 0.25, 1)
    a_px = (zz / f) ** 2 / cos * (mask.sum() / len(selm))      # in-plane area per sampled pixel
    area = float(a_px.sum())
    return {"available": True, "coverage": round(coverage, 2),
            "depleted_m3": float(np.sum(np.clip(dh, 0, None) * a_px)),
            "accumulated_m3": float(np.sum(np.clip(-dh, 0, None) * a_px)),
            "cv_rmse_m": float(np.sqrt(best[0])), "smoothing": best[1],
            "mean_abs_dh_m": float(np.mean(np.abs(dh))), "area_m2": area}


# ----------------------------------------------------------------------------- checks
def haze_rho(gray, depth):
    """Aerial perspective: local contrast should fall with distance (Spearman rho < 0)."""
    c = np.sqrt(np.maximum(box(gray ** 2, 7) - box(gray, 7) ** 2, 0))
    s = (slice(None, None, 8), slice(None, None, 8))
    return float(stats.spearmanr(depth[s].ravel(), c[s].ravel())[0])


def texture_depth_slope(gray, mask, depth, patch=48):
    """Texture-frequency check with no segmentation. For one material, the image-space spatial frequency of
    its texture grows in proportion to distance, so log(spectral centroid) against log(depth) has slope ~1.
    Estimated robustly (Theil-Sen) on Hann-windowed FFT patches inside the outline. None when the depth
    spread is too small to test."""
    H, W = gray.shape
    win = np.outer(np.hanning(patch), np.hanning(patch))
    fy, fx = np.meshgrid(np.fft.fftfreq(patch), np.fft.fftfreq(patch), indexing="ij")
    rad = np.sqrt(fx ** 2 + fy ** 2)
    band = (rad > 1.5 / patch) & (rad <= 0.5)
    fc, zc = [], []
    for y in range(0, H - patch, patch):
        for x in range(0, W - patch, patch):
            if mask[y:y + patch, x:x + patch].mean() < 0.8:
                continue
            p = gray[y:y + patch, x:x + patch]
            if p.std() < 2:
                continue
            ps = np.abs(np.fft.fft2((p - p.mean()) * win)) ** 2
            pb, rb = ps[band], rad[band]
            pb = np.clip(pb - np.median(pb[rb > 0.35]), 0, None)    # remove the white-noise floor first
            if pb.sum() <= 0:
                continue
            fc.append(np.exp(np.sum(pb * np.log(rb)) / pb.sum()))  # power-weighted log-frequency mean
            zc.append(np.median(depth[y:y + patch, x:x + patch]))
    if len(fc) < 15 or np.std(np.log(zc)) < 0.15:
        return None
    return float(stats.theilslopes(np.log(fc), np.log(zc))[0])


def corominas_residual(h_over_l, volume):
    """Mobility check: Corominas (1996), all landslides, log10(H/L) = -0.047 - 0.085 log10(V). Returns the
    residual in log10 units. Its scatter is roughly 0.2, so |residual| > 0.4 is unusual."""
    if not h_over_l or not volume or volume <= 0 or h_over_l <= 0:
        return None
    return float(np.log10(h_over_l) - (-0.047 - 0.085 * np.log10(volume)))


# ----------------------------------------------------------------------------- grains
def grain_stats(b_axes, limit, count_w=None, n_boot=200, seed=0):
    """Grain-size distribution from photo-sieved b-axes (m), each completed to its convex hull, which corrects
    for partial occlusion.
    - Photo-sieving samples area-by-number; it is converted to the field standard grid-by-number
      (Wolman count) by weighting each grain by b^2 (Kellerhals & Bray 1971).
    - Grains below the resolution limit are missed, so a log-normal is fitted by weighted maximum
      likelihood, LEFT-TRUNCATED at the limit. That recovers D16/D50/D84 and the hidden fine fraction
      instead of ignoring it.
    - Bootstrap 90 % interval for D50. Kolmogorov-Smirnov distance as the fit check."""
    b_axes = np.asarray(b_axes, float)
    cw = np.ones_like(b_axes) if count_w is None else np.asarray(count_w, float)   # zoom coverage weights
    keep = b_axes >= limit
    b, cw = b_axes[keep], cw[keep]
    if len(b) < 20:
        return None
    y, w = np.log(b), cw * b ** 2 / np.sum(cw * b ** 2)
    ll = np.log(limit)

    def nll(th, y, w):
        mu, s = th[0], np.exp(th[1])
        tail = stats.norm.sf((ll - mu) / s)
        return -np.sum(w * (stats.norm.logpdf((y - mu) / s) - np.log(s) - np.log(tail + 1e-300)))

    def fit(y, w, x0=None):
        if x0 is None:
            mu0, s0 = np.sum(w * y), np.sqrt(np.sum(w * (y - np.sum(w * y)) ** 2)) + 1e-3
            x0 = [mu0, np.log(s0)]
        r = optimize.minimize(nll, x0, args=(y, w), method="Nelder-Mead",
                              options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
        return r.x[0], float(np.exp(r.x[1]))

    mu, s = fit(y, w)
    q = lambda p: float(np.exp(mu + s * stats.norm.ppf(p)))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        wi = w[i] / w[i].sum()
        m_, s_ = fit(y[i], wi, x0=[mu, np.log(s)])   # warm start at the full-sample optimum
        boots.append(np.exp(m_))
    # KS distance between weighted empirical CDF and the fitted truncated CDF
    o = np.argsort(y)
    ecdf = np.cumsum(w[o])
    F = lambda t: (stats.norm.cdf((t - mu) / s) - stats.norm.cdf((ll - mu) / s)) / max(stats.norm.sf((ll - mu) / s), 1e-12)
    ks = float(np.max(np.abs(ecdf - F(y[o]))))
    return {"count": int(len(b)), "D16_m": round(q(.16), 3), "D50_m": round(q(.5), 3), "D84_m": round(q(.84), 3),
            "D50_90ci_m": [round(float(np.percentile(boots, 5)), 3), round(float(np.percentile(boots, 95)), 3)],
            "D_max_m": round(float(b.max()), 3), "sorting_sigma_phi": round(s / np.log(2), 2),
            "fine_fraction_below_limit": round(float(stats.norm.cdf((ll - mu) / s)), 2),
            "resolution_limit_m": round(float(limit), 3), "ks": round(ks, 3),
            "method": "grid-by-number (K&B 1971), truncated log-normal MLE"}


# ----------------------------------------------------------------------------- uncertainty
def lognormal_from_interval(lo, hi):
    """(mu, sd) of a log-normal whose central 95 % interval is [lo, hi]."""
    return (np.log(lo) + np.log(hi)) / 2, (np.log(hi) - np.log(lo)) / (2 * 1.96)


def monte_carlo(variants, scale_sd, volume_fn, n=1000, seed=0):
    """Propagates the uncertainty jointly. Each draw picks one outline variant (boundary uncertainty from the
    SAM prompt jitter and the probability thresholds) and a log-normal scale error (depth, focal and
    calibration), then calls volume_fn(base, s, rng) for the class-specific volume with its own priors.
    Lengths scale with s and areas with s^2. Returns P5/P50/P95 for every measure and the volume draws."""
    rng = np.random.default_rng(seed)
    keys = [k for k in variants[0] if isinstance(variants[0][k], (int, float))]
    draws = {k: np.empty(n) for k in keys}
    vols = np.empty(n)
    for i in range(n):
        base = variants[rng.integers(len(variants))]
        s = float(np.exp(rng.normal(0, scale_sd)))
        for k in keys:
            p = 2 if k.endswith("area") else (0 if k.endswith(("deg", "ratio")) else 1)
            draws[k][i] = base[k] * s ** p
        vols[i] = volume_fn(base, s, rng)
    pct = {k: [float(np.percentile(v, 5)), float(np.percentile(v, 50)), float(np.percentile(v, 95))]
           for k, v in draws.items()}
    pct["volume"] = [float(np.percentile(vols, 5)), float(np.percentile(vols, 50)), float(np.percentile(vols, 95))]
    return pct, vols
