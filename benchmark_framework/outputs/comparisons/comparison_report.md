# Benchmark Comparison Report

## Backend Özeti

| Backend | Precision | mAP@0.5 | mAP@0.5:0.95 | Precision Metric | Recall | Avg Inference (ms) | Inference FPS | Avg Full Pipeline (ms) | Avg FPS | Avg CPU % | Avg Process RAM (MB) | Avg Temp (°C) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rknn | unknown | 0.103161 | 0.066766 | 0.627551 | 0.230769 | 43.972 | 22.7417 | 50.2737 | 19.8911 | 13.1 | 362.1758 | 29.615 |

## Kısa Yorum

- En yüksek **mAP@0.5**: `rknn` (0.103161)
- En düşük **inference süresi**: `rknn` (43.972 ms)
- En yüksek **uçtan uca FPS**: `rknn` (19.8911)

## Precision Sensitivity Analysis

Bu bölüm aynı modelin FP16 ve INT8 koşularını eşleştirerek quantization sonrası doğruluk kaybını raporlar.

- Eşleşen FP16/INT8 çift sayısı: **0**

_FP16 ve INT8 arasında eşleşen uygun run bulunamadı._

