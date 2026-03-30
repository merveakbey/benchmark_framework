# Benchmark Framework

Bu proje, VisDrone veri seti üzerinde eğitilmiş YOLO tabanlı modelleri farklı backend’lerde ortak bir yapı içinde benchmark edebilmek için geliştirilmiştir. Framework; **PyTorch**, **ONNX Runtime**, **TensorRT** ve **RKNN** backend’lerini destekleyecek şekilde modüler olarak tasarlanmıştır.

Amaç, aynı model ailesini farklı çalışma ortamlarında test ederek:
- doğruluk farklarını,
- hız ve gecikme profillerini,
- sistem kaynağı kullanımını
ortak bir formatta ölçmek ve raporlamaktır.

---

## Temel Özellikler

- Modüler ve OOP tabanlı mimari
- Config tabanlı çalışma yapısı
- Çoklu backend desteği
  - PyTorch (`.pt`)
  - ONNX Runtime (`.onnx`)
  - TensorRT (`.engine`)
  - RKNN (`.rknn`)
- VisDrone veri seti desteği
- Accuracy metrikleri
  - `mAP@0.5`
  - `mAP@0.5:0.95`
  - `precision`
  - `recall`
- Latency metrikleri
  - preprocess
  - inference
  - postprocess
  - full pipeline
- FPS hesaplaması
- Monitoring
  - CPU kullanımı
  - RAM kullanımı
  - sıcaklık
- Otomatik raporlama
  - JSON
  - CSV
  - comparison CSV
  - comparison Markdown

---

## Proje Yapısı


benchmark_framework/
├── benchmark/
│   ├── adapters/
│   ├── core/
│   ├── datasets/
│   ├── evaluators/
│   ├── monitors/
│   ├── pipelines/
│   ├── profilers/
│   ├── reporters/
│   ├── schemas/
│   └── tools/
├── configs/
│   └── runs/
├── outputs/
│   ├── runs/
│   └── comparisons/
└── models/


## Mimari Yaklaşım

Framework, her backend için ortak bir arayüz sunacak şekilde tasarlanmıştır. Böylece aynı benchmark akışı, yalnızca config değiştirilerek farklı model formatları üzerinde çalıştırılabilir.

Genel akış şu şekildedir:

Config dosyası okunur
İlgili backend adapter’ı yüklenir
Veri seti loader başlatılır
Pipeline üzerinden inference yapılır
Accuracy, latency ve monitoring verileri toplanır
Sonuçlar JSON / CSV / Markdown formatında raporlanır

Bu yapı sayesinde yeni bir backend veya yeni bir veri seti desteği eklemek, mevcut sistemi bozmadan gerçekleştirilebilir.

## Çalışma Mantığı

Framework içindeki benchmark süreci temel olarak dört ana aşamadan oluşur:

1. Veri Hazırlama

Bu aşamada VisDrone veri seti okunur, görüntüler ve etiketler benchmark sürecine uygun hale getirilir. Gerekli ön işleme adımları burada uygulanır.

2. Model Çıkarımı

Seçilen backend’e göre uygun adapter devreye girer ve model inference işlemi gerçekleştirilir. Bu aşamada preprocess, inference ve postprocess süreleri ayrı ayrı ölçülür.

3. Değerlendirme

Model çıktıları ground-truth etiketlerle karşılaştırılır. Bu sayede doğruluk metrikleri hesaplanır:
mAP@0.5
mAP@0.5:0.95
precision
recall

4. Raporlama

Toplanan tüm performans verileri standart bir formatta dışa aktarılır. Böylece farklı koşullarda yapılan testler kolayca karşılaştırılabilir.


## Desteklenen Backend Yapısı

Her backend için ayrı bir adapter katmanı bulunmaktadır. Bu katmanların amacı, farklı inference motorlarını ortak bir arayüz altında toplamaktır.

PyTorch Adapter

.pt uzantılı modelleri yükler ve PyTorch üzerinden inference çalıştırır.

ONNX Runtime Adapter

.onnx modeller için ONNX Runtime kullanır. CPU veya uygun olduğu durumda farklı execution provider’lar ile genişletilebilir.

TensorRT Adapter

.engine formatındaki optimize edilmiş modellerin NVIDIA GPU üzerinde yüksek performansla çalıştırılmasını sağlar.

RKNN Adapter

.rknn modellerin RK3588 gibi Rockchip NPU tabanlı cihazlarda test edilmesini hedefler.

## Config Yapısı

Framework’ün temel çalışma mantığı config tabanlıdır. Her benchmark çalışması ayrı bir YAML dosyası ile tanımlanır. Bu yapı sayesinde kod değiştirmeden farklı model, backend, cihaz ve veri seti kombinasyonları test edilebilir.

Örnek config yapısı:

run:
  name: yolo26nvisdroneboat_onnxruntime_cpu_fp32
  output_root: ./outputs/runs
  global_summary_path: ./outputs/comparisons/all_runs_summary.csv

task:
  type: detection
  model_family: yolo

model:
  name: yolo26_visdroneboat
  format: onnx
  path: /home/merve/benchmark_framework/models/yolo26nvisdroneboat.onnx
  precision: fp32
  input_size: [640, 640]
  class_names:
    - pedestrian
    - people
    - bicycle
    - car
    - van
    - truck
    - tricycle
    - awning-tricycle
    - bus
    - motor
    - boat

backend:
  type: onnxruntime
  device: cpu

dataset:
  type: visdrone
  root_dir: /home/merve/benchmark_framework/data/VisDrone2019-DET-val
  split: val

Bu yapı ileride farklı dataset tipleri ve backend seçenekleri için genişletilebilir şekilde düşünülmüştür.

## Üretilen Çıktılar

Her benchmark çalışması sonunda ilgili run klasörü altında ayrıntılı çıktı dosyaları oluşturulur.

Örnek çıktı yapısı:

outputs/
├── runs/
│   └── yolo26nvisdroneboat_onnxruntime_cpu_fp32/
│       ├── run_summary.json
│       ├── run_summary.csv
│       ├── latency_profile.json
│       ├── monitoring.json
│       ├── accuracy.json
│       └── report.md
└── comparisons/
    ├── all_runs_summary.csv
    └── comparison_report.md

## Çıktı Dosyalarının İçeriği

run_summary.json
Tek bir benchmark çalışmasının tüm özet metriklerini içerir.
run_summary.csv
Sonuçların tablo halinde saklanmasını sağlar.
latency_profile.json
Preprocess, inference, postprocess ve full pipeline sürelerini içerir.
monitoring.json
CPU, RAM ve sıcaklık gibi sistem bilgilerini içerir.
accuracy.json
mAP, precision ve recall gibi doğruluk metriklerini içerir.
report.md
İnsan tarafından okunabilir özet benchmark raporudur.
all_runs_summary.csv
Tüm benchmark çalışmaları için ortak karşılaştırma tablosu oluşturur.
comparison_report.md
Farklı run’ların toplu karşılaştırmasını Markdown formatında sunar.

## Ölçülen Metrikler

Framework üç ana kategori altında metrik üretir:

1. Doğruluk Metrikleri

Modelin veri seti üzerindeki algılama başarısını ölçer.

mAP@0.5
mAP@0.5:0.95
precision
recall

2. Performans Metrikleri

Pipeline’ın farklı aşamalarındaki süreleri ölçer.

preprocess latency
inference latency
postprocess latency
full pipeline latency
FPS

3. Sistem Kaynak Kullanımı

Model çalışırken sistem üzerindeki yükü gözlemlemeyi amaçlar.

CPU kullanım yüzdesi
RAM kullanımı
sıcaklık bilgisi
Genişletilebilirlik

Framework başlangıçta VisDrone ve YOLO tabanlı modeller odak alınarak tasarlanmıştır. Ancak mimari, ileride aşağıdaki genişletmelere uygundur:

yeni veri setleri ekleme
yeni backend adapter’ları ekleme
classification / segmentation gibi yeni görev tipleri ekleme
farklı monitoring kaynakları ekleme
daha detaylı profiler entegrasyonları ekleme

Bu sayede proje yalnızca tek bir benchmark senaryosuna değil, daha genel bir model değerlendirme altyapısına dönüşebilecek şekilde kurgulanmıştır.

## Hedef Kullanım Senaryosu

Bu framework özellikle aşağıdaki ihtiyaçlara cevap vermek için geliştirilmiştir:

Aynı modelin farklı formatlardaki performansını karşılaştırmak
FP32 / FP16 / INT8 gibi hassasiyet seviyeleri arasındaki farkı gözlemlemek
Edge cihazlar ile masaüstü sistemleri aynı raporlama mantığında değerlendirmek
Eğitim sonrası model optimizasyonlarının doğruluk ve hız üzerindeki etkisini ölçmek
Tekrarlanabilir ve standart benchmark sonuçları elde etmek

## Sonraki Geliştirmeler

Planlanan sonraki geliştirmeler şunlardır:

RKNN backend entegrasyonunun tamamlanması
TensorRT profiler detaylarının artırılması
comparison Markdown raporlarının otomatik grafiklerle desteklenmesi
çoklu run batch benchmark desteği
CLI komut yapısının geliştirilmesi
görselleştirilmiş benchmark özetleri
Genel Değerlendirme

Bu benchmark framework, farklı backend’lerde çalışan YOLO tabanlı modellerin ortak bir yapı içinde değerlendirilmesini amaçlayan modüler bir altyapı sunmaktadır. Proje sayesinde doğruluk, hız ve sistem kullanımı tek bir standartta ölçülerek farklı çalışma ortamları arasında daha sağlıklı karşılaştırmalar yapılabilecektir. Özellikle edge AI, optimize inference ve backend karşılaştırma çalışmalarında tekrar kullanılabilir bir temel oluşturması hedeflenmektedir.