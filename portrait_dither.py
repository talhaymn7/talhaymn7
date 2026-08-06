"""
Portre dot-yogunluk haritasi uretici.
Girdi: subject_transparent.png (kirpilmis, rembg alfa), talha_photo.jpeg (orijinal, arkaplanli)
Cikti (npy, kaynak-of-truth):
  - dark_dots.npy   (340,300) bool  -- koyu mod: parlak/aydinlik pikseller = nokta, maskeyle temizlenmis
  - light_dots.npy  (340,300) bool  -- acik mod: koyu pikseller = nokta, arkaplan korunur
  - mask_clean.npy  (340,300) bool  -- koyu mod icin temizlenmis konu maskesi (debug/verify icin)
  - gray_dark.npy / gray_light.npy (340,300) uint8 -- dither ONCESI gri ton, korelasyon olcumu icin
Ayrica hizli gorsel kontrol icin preview_dark.png / preview_light.png üretir.
"""
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw
import numpy as np
from scipy import ndimage

GRID_W, GRID_H = 300, 340
CROP_TOP = 819  # crop_top.py ile ayni miktar (ust boslugun yarisi)
DARK_BG_HEX = "#0A101F"


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def cover_resize_crop(img, target_w, target_h):
    """Kirpilmadan tasan minimum olcek + merkezden dikey kirpma."""
    w, h = img.size
    scale = max(target_w / w, target_h / h)
    new_w, new_h = round(w * scale), round(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    top = (new_h - target_h) // 2
    left = (new_w - target_w) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def enhance(gray_img):
    g = ImageOps.autocontrast(gray_img, cutoff=1)
    g = ImageEnhance.Contrast(g).enhance(1.3)
    g = g.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    return g


def floyd_steinberg_serpentine(arr):
    """arr: float64 (h,w) 0-255. Doner: uint8 (h,w) sadece 0/255 degerleri, serpentine FS."""
    a = arr.copy()
    h, w = a.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        left_to_right = (y % 2 == 0)
        xs = range(w) if left_to_right else range(w - 1, -1, -1)
        for x in xs:
            old = a[y, x]
            new = 255.0 if old > 127 else 0.0
            out[y, x] = int(new)
            err = old - new
            if left_to_right:
                if x + 1 < w:
                    a[y, x + 1] += err * 7 / 16
                if y + 1 < h:
                    if x - 1 >= 0:
                        a[y + 1, x - 1] += err * 3 / 16
                    a[y + 1, x] += err * 5 / 16
                    if x + 1 < w:
                        a[y + 1, x + 1] += err * 1 / 16
            else:
                if x - 1 >= 0:
                    a[y, x - 1] += err * 7 / 16
                if y + 1 < h:
                    if x + 1 < w:
                        a[y + 1, x + 1] += err * 3 / 16
                    a[y + 1, x] += err * 5 / 16
                    if x - 1 >= 0:
                        a[y + 1, x - 1] += err * 1 / 16
    return out


def clean_mask(mask_bool):
    """binary closing -> fill holes -> en buyuk bagli bilesen."""
    closed = ndimage.binary_closing(mask_bool, structure=np.ones((3, 3)), iterations=2)
    filled = ndimage.binary_fill_holes(closed)
    labeled, n = ndimage.label(filled)
    if n == 0:
        return filled
    sizes = ndimage.sum(filled, labeled, range(1, n + 1))
    largest = np.argmax(sizes) + 1
    return labeled == largest


def correlation_metric(dot_bool, gray_before_dither, sigma=3.0):
    """Nokta yogunlugu (gaussian blur) ile orijinal gri ton (ayni blur) arasindaki korelasyon."""
    dot_f = dot_bool.astype(np.float64)
    gray_f = gray_before_dither.astype(np.float64)
    dot_blur = ndimage.gaussian_filter(dot_f, sigma=sigma)
    gray_blur = ndimage.gaussian_filter(gray_f, sigma=sigma)
    return float(np.corrcoef(dot_blur.ravel(), gray_blur.ravel())[0, 1])


def render_preview(dot_bool, bg_hex, dot_hex, path, cell=4):
    h, w = dot_bool.shape
    img = Image.new("RGB", (w * cell, h * cell), hex_to_rgb(bg_hex))
    draw = ImageDraw.Draw(img)
    dot_rgb = hex_to_rgb(dot_hex)
    ys, xs = np.where(dot_bool)
    for y, x in zip(ys, xs):
        draw.rectangle([x * cell, y * cell, x * cell + cell - 1, y * cell + cell - 1], fill=dot_rgb)
    img.save(path)


def main():
    # ---- DARK MODE kaynagi: subject_transparent.png (kirpilmis, alfa=konu) ----
    transparent = Image.open("subject_transparent.png")
    transparent = cover_resize_crop(transparent, GRID_W, GRID_H)
    alpha = np.array(transparent)[:, :, 3]
    mask_raw = alpha > 127
    mask_clean = clean_mask(mask_raw)

    dark_rgb = transparent.convert("RGB")
    dark_gray = enhance(dark_rgb.convert("L"))
    dark_gray_arr = np.array(dark_gray, dtype=np.float64)
    dark_quant = floyd_steinberg_serpentine(dark_gray_arr)
    dark_dots_raw = dark_quant == 255
    dark_dots = dark_dots_raw & mask_clean  # hard-clear: maske disindaki sizinti temizlenir

    # ---- LIGHT MODE kaynagi: orijinal foto (arkaplan korunur), ayni kirpma ----
    original = Image.open("talha_photo.jpeg").convert("RGB")
    w, h = original.size
    original_top_cropped = original.crop((0, CROP_TOP, w, h))
    light_src = cover_resize_crop(original_top_cropped, GRID_W, GRID_H)
    light_gray = enhance(light_src.convert("L"))
    light_gray_arr = np.array(light_gray, dtype=np.float64)

    # Arkaplan (mask_clean disi) beyaza dogru harmanlanir: gercek arkaplan
    # (cam korkuluk, tavan ekipmani) tam kontrastta dither'a girerse konuyu
    # bogar. Konu (mask_clean ici) tam kontrastta kalir, arkaplan "hayalet"
    # gibi seyrek iz birakir ama tamamen kaybolmaz.
    BG_LIGHTEN = 0.85
    lightened_bg = light_gray_arr + (255.0 - light_gray_arr) * BG_LIGHTEN
    light_gray_arr = np.where(mask_clean, light_gray_arr, lightened_bg)

    light_quant = floyd_steinberg_serpentine(light_gray_arr)
    light_dots = light_quant == 0  # koyu bolgeler = nokta

    # ---- kaydet: npy kaynak-of-truth ----
    np.save("dark_dots.npy", dark_dots)
    np.save("light_dots.npy", light_dots)
    np.save("mask_clean.npy", mask_clean)
    np.save("gray_dark.npy", dark_gray_arr.astype(np.uint8))
    np.save("gray_light.npy", light_gray_arr.astype(np.uint8))

    # ---- metrikler ----
    dark_coverage = dark_dots.mean()
    light_coverage = light_dots.mean()
    dark_corr = correlation_metric(dark_dots, dark_gray_arr)
    light_corr = correlation_metric(light_dots, 255 - light_gray_arr)  # ters kutup: koyu=nokta
    bled_before_clear = int((dark_dots_raw & ~mask_clean).sum())

    print(f"Grid: {GRID_W}x{GRID_H} = {GRID_W*GRID_H} hucre")
    print(f"Dark dots : {dark_dots.sum()}  (kapsama {dark_coverage:.2%}, korelasyon {dark_corr:.3f})")
    print(f"Light dots: {light_dots.sum()}  (kapsama {light_coverage:.2%}, korelasyon {light_corr:.3f})")
    print(f"Maske disinda temizlenen sizinti noktasi: {bled_before_clear}")
    print(f"Mask clean alani: {mask_clean.sum()} hucre ({mask_clean.mean():.2%})")

    # ---- gorsel onizleme ----
    render_preview(dark_dots, "#0A101F", "#A78BFA", "preview_dark.png")
    render_preview(light_dots, "#FFFFFF", "#7C3AED", "preview_light.png")
    print("preview_dark.png, preview_light.png yazildi")


if __name__ == "__main__":
    main()
