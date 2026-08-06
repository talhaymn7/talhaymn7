"""
Arka plan kaldırma — talha_photo.jpeg
Basit renk-eşik yöntemi bu foto için işe yaramaz çünkü arka plan düz değil
(cam korkuluk, tavan ışıkları, tabelalar). Onun yerine gerçek bir insan
segmentasyon modeli (u2net, rembg üzerinden) kullanıyoruz.

Çıktılar:
  - subject_transparent.png  -> arka planı şeffaf, sadece kişi
  - subject_on_dark.png      -> kişi, banner'ın koyu arkaplanı (#0A101F) üzerine
  - mask.png                 -> segmentasyon maskesi (kontrol için)
"""
from rembg import remove, new_session
from PIL import Image
import numpy as np

IN_PATH = "talha_photo.jpeg"
BG_HEX = "#0A101F"  # banner arkaplan rengi

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def main():
    img = Image.open(IN_PATH).convert("RGB")

    # u2net_human_seg: insan segmentasyonuna özel model, genel u2net'ten
    # daha keskin kenar verir (saç, omuz hattı)
    session = new_session("u2net_human_seg")
    cutout = remove(img, session=session)  # RGBA, arka plan şeffaf

    cutout.save("subject_transparent.png")

    # kontrol için maskeyi ayrı kaydet (alpha kanalı)
    alpha = np.array(cutout)[:, :, 3]
    Image.fromarray(alpha).save("mask.png")

    # koyu banner arkaplanına yerleştirilmiş hali — dithering pipeline'ı
    # için asıl kullanılacak versiyon bu
    bg = Image.new("RGB", cutout.size, hex_to_rgb(BG_HEX))
    bg.paste(cutout, (0, 0), cutout)
    bg.save("subject_on_dark.png")

    print(f"Boyut: {cutout.size}")
    print(f"Alpha kapsama oranı (kişi/arkaplan): {(alpha > 128).mean():.2%}")
    print("Çıktılar: subject_transparent.png, subject_on_dark.png, mask.png")

if __name__ == "__main__":
    main()
