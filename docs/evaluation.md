# Sistem testi ve değerlendirme

Canlı koşu: `python main.py --demo`

| # | Tür | Soru | Beklenen | Kaynak |
|---|-----|------|----------|--------|
| 1 | Cevaplanabilir | Menemen nasıl yapılır? | soğan, biber, domates, yumurta | menemen.txt |
| 2 | Cevaplanabilir | Cacıkta hangi malzemeler var? | yoğurt, salatalık, sarımsak | cacik.txt |
| 3 | Cevaplanabilir | vegan tarif öner | kısır, imam bayıldı, sebze güveç | diyet.txt / kisir.txt |
| 4 | Cevaplanabilir | sebzesiz tarif öner | omlet, pilav, baklava | diyet.txt |
| 5 | Cevaplanabilir | Baklava nasıl şerbetlenir? | kaynar şerbet, soğumuş baklava | baklava.txt |
| 6 | Cevaplanamaz | Bu uygulamanın aylık barındırma maliyeti nedir? | Bu bilgi context'te yok. | - |
| 7 | Cevaplanamaz | Ankara'nın plaka kodu nedir? | Bu bilgi context'te yok. | - |
| 8 | Uç durum | (boş Enter) | Boş soru gönderildi. | - |

## Performans

Hedef: soru başına yaklaşık 1–3 saniye (ilk yükleme hariç). CLI süreyi yazar.

## Notlar

- Tarifleri parçalara ayırmak aramayı iyileştirir.
- `min_score=0.45` maliyet / plaka gibi alakasız soruda modeli devre dışı bırakır.
- Kaynak adı `[Kaynak: menemen.txt]` olarak context’e eklenir.
