"""Tarif arayüzü için kısa demo GIF üretir (Pillow)."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo.gif"
W, H = 880, 520
BG = (251, 243, 231)
PANEL = (255, 248, 239)
ACCENT = (217, 114, 76)
TITLE = (181, 84, 30)
INK = (111, 74, 38)
OK = (111, 143, 91)
SIDE = (243, 227, 204)


def font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\seguiemj.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def base_frame(subtitle):
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((36, 28, W - 36, H - 28), radius=28, fill=PANEL)
    draw.text((64, 52), "Tarif Defteri Asistanı", fill=TITLE, font=font(34, True))
    draw.text((64, 100), subtitle, fill=INK, font=font(20))
    tabs = [("Soru Sor", True), ("Malzemelerim", False), ("Galeri", False)]
    x = 64
    for label, active in tabs:
        color = ACCENT if active else SIDE
        text_color = PANEL if active else INK
        draw.rounded_rectangle((x, 140, x + 130, 178), radius=12, fill=color)
        draw.text((x + 16, 148), label, fill=text_color, font=font(16, True))
        x += 142
    return img, draw


def frame_search():
    img, draw = base_frame("Doğal dilde tarif ara")
    draw.rounded_rectangle((64, 200, 620, 250), radius=14, fill=BG, outline=(232, 212, 176), width=2)
    draw.text((82, 214), "vegan tarif öner", fill=INK, font=font(20))
    draw.rounded_rectangle((640, 200, 800, 250), radius=14, fill=ACCENT)
    draw.text((690, 214), "Ara", fill=PANEL, font=font(20, True))
    return img


def frame_results():
    img, draw = base_frame("Diyet filtreli sonuçlar")
    draw.rounded_rectangle((64, 200, 620, 250), radius=14, fill=BG, outline=(232, 212, 176), width=2)
    draw.text((82, 214), "vegan tarif öner", fill=INK, font=font(20))
    draw.rounded_rectangle((640, 200, 800, 250), radius=14, fill=ACCENT)
    draw.text((690, 214), "Ara", fill=PANEL, font=font(20, True))
    cards = ["İmam Bayıldı · vegan", "Kısır · vegan", "Sebze Güveç · vegan"]
    y = 280
    for card in cards:
        draw.rounded_rectangle((64, y, 800, y + 58), radius=14, fill=SIDE)
        draw.ellipse((84, y + 18, 108, y + 42), fill=OK)
        draw.text((124, y + 16), card, fill=INK, font=font(20, True))
        y += 70
    return img


def frame_match():
    img, draw = base_frame("Malzeme uyum yüzdesi")
    draw.rounded_rectangle((64, 140, 194, 178), radius=12, fill=SIDE)
    draw.text((80, 148), "Soru Sor", fill=INK, font=font(16, True))
    draw.rounded_rectangle((206, 140, 336, 178), radius=12, fill=ACCENT)
    draw.text((218, 148), "Malzemelerim", fill=PANEL, font=font(16, True))
    draw.rounded_rectangle((64, 200, 620, 250), radius=14, fill=BG, outline=(232, 212, 176), width=2)
    draw.text((82, 214), "yumurta, un, süt", fill=INK, font=font(20))
    draw.rounded_rectangle((640, 200, 800, 250), radius=14, fill=ACCENT)
    draw.text((668, 214), "Eşleştir", fill=PANEL, font=font(18, True))
    draw.rounded_rectangle((64, 290, 400, 420), radius=14, fill=SIDE)
    draw.rounded_rectangle((84, 310, 180, 340), radius=10, fill=OK)
    draw.text((96, 314), "%40 uyum", fill=PANEL, font=font(16, True))
    draw.text((84, 360), "Fırın Sütlaç", fill=TITLE, font=font(22, True))
    draw.text((84, 392), "eksik: pirinç, nişasta", fill=INK, font=font(16))
    return img


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames = [frame_search(), frame_results(), frame_results(), frame_match(), frame_match()]
    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=[900, 1200, 900, 1200, 900],
        loop=0,
        optimize=False,
    )
    print(OUT)


if __name__ == "__main__":
    main()
