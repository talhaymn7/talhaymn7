"""
subject_transparent.png ve subject_on_dark.png icin ust bosluk kirpma.
Alpha kanalindaki ilk dolu (kisi) satirini bulup, ustundeki bos alanin
yarisini kirpar. Geri kalan yari bosluk portre cercevesinde nefes payi
birakir.
"""
from PIL import Image
import numpy as np

def find_top(alpha, threshold=10):
    rows = np.where((alpha > threshold).any(axis=1))[0]
    return int(rows[0]) if len(rows) else 0

def main():
    transparent = Image.open("subject_transparent.png")
    alpha = np.array(transparent)[:, :, 3]
    top = find_top(alpha)
    crop_amount = top // 2

    print(f"Goruntu boyu: {transparent.size}")
    print(f"Kisinin basladigi satir (ilk alpha>10): {top}px")
    print(f"Kirpilacak miktar (ustteki boslugun yarisi): {crop_amount}px")

    for name in ("subject_transparent.png", "subject_on_dark.png"):
        img = Image.open(name)
        w, h = img.size
        cropped = img.crop((0, crop_amount, w, h))
        cropped.save(name)
        print(f"{name}: {img.size} -> {cropped.size}")

if __name__ == "__main__":
    main()
