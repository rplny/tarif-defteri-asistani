"""RAG CLI akışını gösteren demo GIF."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "rag-cli.gif"
W, H = 920, 560
BG = (18, 18, 22)
TERM = (24, 26, 32)
GREEN = (110, 190, 120)
CYAN = (90, 190, 210)
AMBER = (232, 176, 80)
WHITE = (230, 230, 230)
MUTED = (150, 155, 165)
RED = (220, 120, 100)


def font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\consolab.ttf" if bold else r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def terminal(lines):
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((28, 24, W - 28, H - 24), radius=18, fill=TERM)
    draw.ellipse((48, 42, 66, 60), fill=(232, 90, 80))
    draw.ellipse((76, 42, 94, 60), fill=(232, 176, 80))
    draw.ellipse((104, 42, 122, 60), fill=(110, 190, 120))
    draw.text((148, 40), "python main.py  -  Tarif Defteri RAG", fill=MUTED, font=font(16))
    y = 88
    for color, text in lines:
        draw.text((52, y), text, fill=color, font=font(18))
        y += 28
    return img


def frames():
    return [
        terminal(
            [
                (MUTED, "> python main.py --demo"),
                (GREEN, "Embedding modeli indiriliyor: 100.0%"),
                (GREEN, "21 parça indekslendi (10 kaynak dosya)."),
                (GREEN, "Sohbet modeli indiriliyor: 100.0%"),
                (WHITE, "Modeller yüklendi. Sorularınızı bekliyorum."),
                (MUTED, 'Çıkmak için "quit" yazın.'),
                (CYAN, "Soru: _"),
            ]
        ),
        terminal(
            [
                (WHITE, "[cevaplanabilir] Soru: Menemen nasıl yapılır?"),
                (AMBER, "Arama: menemen.txt (0.82)"),
                (GREEN, "Cevap: menemen.txt dosyasına göre soğan,"),
                (GREEN, "biber, domates ve yumurta ile yapılır."),
                (MUTED, "(yanıt süresi: 1.9 sn)"),
                (CYAN, "Soru: vegan tarif öner"),
            ]
        ),
        terminal(
            [
                (WHITE, "[vegan] Soru: vegan tarif öner"),
                (AMBER, "Arama: diyet.txt (0.54), kisir.txt"),
                (GREEN, "Cevap: diyet.txt'ye göre kısır, imam"),
                (GREEN, "bayıldı ve sebze güveç vegan tariflerdir."),
                (MUTED, "(yanıt süresi: 1.7 sn)"),
                (CYAN, "Soru: aylık barındırma maliyeti nedir?"),
            ]
        ),
        terminal(
            [
                (CYAN, "Soru: aylık barındırma maliyeti nedir?"),
                (RED, "Cevap: Bu bilgi context'te yok."),
                (MUTED, "(yanıt süresi: 0.6 sn)"),
                (CYAN, "Soru:"),
                (AMBER, "Boş soru gönderildi."),
                (CYAN, "Soru: quit"),
                (WHITE, "Modeller kapatıldı. Bitti!"),
            ]
        ),
    ]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    imgs = frames()
    imgs[0].save(
        OUT,
        save_all=True,
        append_images=imgs[1:],
        duration=[1400, 1800, 1800, 2200],
        loop=0,
        optimize=False,
    )
    print(OUT)


if __name__ == "__main__":
    main()
