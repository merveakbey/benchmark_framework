# Benchmark Comparison Report

## Backend Özeti

| Backend | Precision | mAP@0.5 | mAP@0.5:0.95 | Precision Metric | Recall | Avg Inference (ms) | Inference FPS | Avg Full Pipeline (ms) | Avg FPS | Avg CPU % | Avg Process RAM (MB) | Avg Temp (°C) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pytorch | fp32 | 0.109154 | 0.068656 | 0.663957 | 0.229831 | 59.8404 | None | 66.6237 | None | 52.7 | 484.5742 | 72.875 |
| onnxruntime | fp32 | 0.090252 | 0.055935 | 0.525532 | 0.231707 | 53.1402 | None | 57.551 | None | 48.6 | None | 0.0 |
| tensorrt | fp32 | 0.109158 | 0.068666 | 0.663957 | 0.229831 | 20.3276 | None | 27.8711 | None | 13.0 | 1229.2461 | 73.25 |
| rknn | unknown | 0.103161 | 0.066766 | 0.627551 | 0.230769 | 43.972 | 22.7417 | 50.2737 | 19.8911 | 13.1 | 362.1758 | 29.615 |

## Kısa Yorum

- En yüksek **mAP@0.5**: `tensorrt` (0.109158)
- En düşük **inference süresi**: `tensorrt` (20.3276 ms)
- En yüksek **uçtan uca FPS**: `rknn` (19.8911)