# Sistem testi ve değerlendirme

Canlı koşu: `python main.py --demo`

| # | Tür | Soru | Beklenen | Kaynak |
|---|-----|------|----------|--------|
| 1 | Cevaplanabilir | Menemen nasıl yapılır? | soğan, biber, domates, yumurta | menemen.txt |
| 2 | Cevaplanabilir | Cacıkta hangi malzemeler var? | yoğurt, salatalık, sarımsak | cacik.txt |
| 3 | Cevaplanabilir | vegan tarif öner | kısır, imam bayıldı, sebze güveç, gözleme | diyet.txt / kisir.txt |
| 4 | Cevaplanabilir | sebzesiz tarif öner | omlet, pilav, baklava, sütlaç, brownie | diyet.txt |
| 5 | Cevaplanabilir | Baklava nasıl şerbetlenir? | kaynar şerbet, soğumuş baklava | baklava.txt |
| 6 | Cevaplanabilir | vejetaryen tarif öner | menemen, cacık, omlet | diyet.txt |
| 7 | Cevaplanabilir | etli tarif öner | köfte, tavuk sote | diyet.txt |
| 8 | Cevaplanabilir | Tavuk sote nasıl yapılır? | tavuk göğsü, soğan, biber | tavuk_sote.txt |
| 9 | Cevaplanamaz | Bu uygulamanın aylık barındırma maliyeti nedir? | Bu bilgi context'te yok. | - |
| 10 | Cevaplanamaz | Ankara'nın plaka kodu nedir? | Bu bilgi context'te yok. | - |
| 11 | Uç durum | (boş Enter) | Boş soru gönderildi. | - |

## Performans

Hedef: soru başına yaklaşık 1–3 saniye (ilk yükleme hariç). CLI süreyi yazar.

## Notlar

- Tarifleri parçalara ayırmak aramayı iyileştirir.
- `min_score=0.45` maliyet / plaka gibi alakasız soruda modeli devre dışı bırakır.
- Kaynak adı `[Kaynak: menemen.txt]` olarak context’e eklenir.
