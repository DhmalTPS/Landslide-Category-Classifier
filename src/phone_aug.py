"""Augmentations that make training photos look like field phone photos.

Commons photos are mostly taken with good cameras by photographers; the app receives quick
phone shots from engineers on a road. The differences we simulate: framing (crop/zoom,
portrait or landscape), a slightly tilted phone, exposure and white balance, motion/focus
blur, low light noise, and heavy JPEG compression from messaging apps. No vertical flips:
the downhill viewpoint (sky at the top, slope rising) is part of the signal.
"""
import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def phone_like(img, rng=random):
    """Random phone-style variant of a PIL RGB image (same size class, PIL out)."""
    w, h = img.size
    # framing: zoom in on a region covering 55-100% of the area, aspect between 3:4 and 16:9
    area = rng.uniform(0.55, 1.0) * w * h
    aspect = rng.choice([rng.uniform(0.75, 1.0), rng.uniform(1.0, 1.78)])
    cw = min(w, int(round((area * aspect) ** 0.5)))
    ch = min(h, int(round((area / aspect) ** 0.5)))
    x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
    img = img.crop((x, y, x + cw, y + ch))
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.6:   # hand-held tilt; crop the black corners away
        a = rng.uniform(-7, 7)
        img = img.rotate(a, resample=Image.BILINEAR, expand=False)
        m = int(abs(a) / 90 * max(img.size)) + 1
        img = img.crop((m, m, img.width - m, img.height - m))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.75, 1.25))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.7, 1.3))
    if rng.random() < 0.3:   # white balance shift (shade, overcast, sodium light)
        arr = np.asarray(img, dtype=np.float32) * np.array([rng.uniform(0.9, 1.1) for _ in range(3)])
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.6)))
    if rng.random() < 0.2:   # sensor noise in poor light
        arr = np.asarray(img, dtype=np.float32)
        arr += np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(3, 10), arr.shape)
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    if rng.random() < 0.7:   # WhatsApp-style recompression
        if rng.random() < 0.5:
            s = rng.uniform(0.4, 0.8)
            img = img.resize((max(64, int(img.width * s)), max(64, int(img.height * s))), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(35, 85))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img


class PhoneLike:
    """torchvision-style transform wrapper."""
    def __call__(self, img):
        return phone_like(img)
