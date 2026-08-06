"""
Logo nokta-bulut ureticisi + traveller optimal transport eslesmesi.
Kaynak: img/fastapi.png (PNG, alfa), logo_nodejs.png / logo_archlinux.png (sharp ile rasterize edilmis SVG).
Sira: FastAPI -> Node.js -> Arch Linux (dongude bu sirayla morph olur).

Cikti (npy, kaynak-of-truth), hepsi GRID_W x GRID_H (300x340) koordinat uzayinda:
  - pts_fastapi.npy    (900,2) float32 -- traveller 0. faz (temel sira)
  - pts_nodejs.npy     (900,2) float32 -- fastapi sirasina optimal-transport ile hizalanmis
  - pts_archlinux.npy  (900,2) float32 -- nodejs (hizalanmis) sirasina optimal-transport ile hizalanmis
Onizleme: preview_logos.png (uc logo yan yana nokta bulutu olarak)
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from skimage.filters import threshold_otsu

GRID_W, GRID_H = 300, 340
N_TRAVELLERS = 900
BOX_FRAC = 0.74  # canvas min-boyutunun yuzdesi, logo bu kutuya sigacak sekilde olceklenir
RNG = np.random.default_rng(20260805)


def load_mask(path, threshold=127):
    img = Image.open(path).convert("RGBA")
    alpha = np.array(img)[:, :, 3]
    return alpha > threshold


def load_mask_isolate_glyph(path, threshold=127):
    """Zemin rozeti (dolu daire) + amblem iceren logolar icin: alfa yerine
    parlaklik-Otsu ayrimi kullanip iki opak renk kumesinden KUCUK alanli
    olani (asil amblem: simsek/dag) mask olarak doner. Buyuk alanli kume
    her zaman arka plan rozetidir (daire)."""
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    alpha = arr[:, :, 3] > threshold
    lum = arr[:, :, :3].astype(np.float64).mean(axis=2)
    vals = lum[alpha]
    thresh = threshold_otsu(vals)
    dark_mask = alpha & (lum < thresh)
    light_mask = alpha & (lum >= thresh)
    return dark_mask if dark_mask.sum() < light_mask.sum() else light_mask


def tight_bbox(mask):
    ys, xs = np.where(mask)
    return xs.min(), xs.max(), ys.min(), ys.max()


def sample_points_in_mask(mask, n_target, max_iter=25):
    """Izgara + jitter ile mask icinde yaklasik esit yogunlukta n_target nokta uretir."""
    h, w = mask.shape
    area = mask.sum()
    s = np.sqrt(area / n_target)
    pts = np.zeros((0, 2))
    for _ in range(max_iter):
        xs = np.arange(s / 2, w, s)
        ys = np.arange(s / 2, h, s)
        if len(xs) == 0 or len(ys) == 0:
            s *= 0.5
            continue
        gx, gy = np.meshgrid(xs, ys)
        gx = gx + RNG.uniform(-s / 2, s / 2, gx.shape)
        gy = gy + RNG.uniform(-s / 2, s / 2, gy.shape)
        gx = np.clip(gx, 0, w - 1)
        gy = np.clip(gy, 0, h - 1)
        ix, iy = gx.astype(int), gy.astype(int)
        inside = mask[iy, ix]
        pts = np.stack([gx[inside], gy[inside]], axis=1)
        count = len(pts)
        if abs(count - n_target) <= max(5, n_target * 0.02):
            break
        s *= np.sqrt(count / n_target) if count > 0 else 0.8

    if len(pts) > n_target:
        idx = RNG.choice(len(pts), n_target, replace=False)
        pts = pts[idx]
    elif len(pts) < n_target:
        ys_all, xs_all = np.where(mask)
        extra = []
        while len(extra) < n_target - len(pts):
            idx = RNG.integers(0, len(xs_all))
            extra.append((xs_all[idx] + RNG.uniform(-0.5, 0.5), ys_all[idx] + RNG.uniform(-0.5, 0.5)))
        pts = np.vstack([pts, np.array(extra)])
    return pts


def place_logo(path, n_target, isolate_glyph=False):
    mask = load_mask_isolate_glyph(path) if isolate_glyph else load_mask(path)
    x0, x1, y0, y1 = tight_bbox(mask)
    cropped = mask[y0:y1 + 1, x0:x1 + 1]
    ch, cw = cropped.shape

    box = BOX_FRAC * min(GRID_W, GRID_H)
    scale = box / max(cw, ch)
    new_w, new_h = max(1, round(cw * scale)), max(1, round(ch * scale))

    cropped_img = Image.fromarray((cropped * 255).astype(np.uint8))
    resized = cropped_img.resize((new_w, new_h), Image.LANCZOS)
    resized_mask = np.array(resized) > 127

    pts = sample_points_in_mask(resized_mask, n_target)
    offset_x = (GRID_W - new_w) / 2
    offset_y = (GRID_H - new_h) / 2
    pts = pts + np.array([offset_x, offset_y])
    return pts.astype(np.float32)


def match_order(src_pts, dst_pts):
    """dst_pts'i src_pts sirasina en kisa toplam mesafeyle (Hungarian/OT) hizalar."""
    cost = np.sum((src_pts[:, None, :] - dst_pts[None, :, :]) ** 2, axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    return dst_pts[col_ind]


def total_path(a, b):
    return float(np.sqrt(((a - b) ** 2).sum(axis=1)).sum())


def render_preview(point_sets, path):
    cell = 3
    img = Image.new("RGB", (GRID_W * cell * len(point_sets), GRID_H * cell), (10, 16, 31))
    draw = ImageDraw.Draw(img)
    for i, pts in enumerate(point_sets):
        ox = i * GRID_W * cell
        for x, y in pts:
            px, py = ox + x * cell, y * cell
            draw.ellipse([px - 1.5, py - 1.5, px + 1.5, py + 1.5], fill=(167, 139, 250))
    img.save(path)


def main():
    pts_fastapi = place_logo("img/fastapi.png", N_TRAVELLERS, isolate_glyph=True)
    pts_nodejs = place_logo("logo_nodejs.png", N_TRAVELLERS, isolate_glyph=False)
    pts_archlinux = place_logo("logo_archlinux.png", N_TRAVELLERS, isolate_glyph=True)

    nodejs_aligned = match_order(pts_fastapi, pts_nodejs)
    archlinux_aligned = match_order(nodejs_aligned, pts_archlinux)

    np.save("pts_fastapi.npy", pts_fastapi)
    np.save("pts_nodejs.npy", nodejs_aligned)
    np.save("pts_archlinux.npy", archlinux_aligned)

    d12 = total_path(pts_fastapi, nodejs_aligned)
    d23 = total_path(nodejs_aligned, archlinux_aligned)
    print(f"FastAPI noktalari  : {len(pts_fastapi)}")
    print(f"Node.js noktalari  : {len(nodejs_aligned)}")
    print(f"ArchLinux noktalari: {len(archlinux_aligned)}")
    print(f"Toplam yol FastAPI->Node.js : {d12:.1f} (ort {d12/N_TRAVELLERS:.2f}/nokta)")
    print(f"Toplam yol Node.js->ArchLinux: {d23:.1f} (ort {d23/N_TRAVELLERS:.2f}/nokta)")

    render_preview([pts_fastapi, nodejs_aligned, archlinux_aligned], "preview_logos.png")
    print("preview_logos.png yazildi")


if __name__ == "__main__":
    main()
