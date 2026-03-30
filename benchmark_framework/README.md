# Benchmark Framework

PyTorch, ONNX Runtime, TensorRT ve RKNN backend'leri için modüler, konfigürasyon tabanlı benchmark ve otomatik raporlama sistemi.

Bu proje, farklı model formatlarında çalışan nesne tespit modellerini **aynı veri seti**, **aynı değerlendirme mantığı** ve **aynı raporlama yapısı** ile karşılaştırmak için geliştirilmiştir. Framework; doğruluk, hız/gecikme ve sistem kaynak kullanımı metriklerini tek bir akış altında toplayarak tekrar üretilebilir benchmark senaryoları sunar.

---

## İçindekiler
- [Proje Amacı](#proje-amacı)
- [Öne Çıkan Özellikler](#öne-çıkan-özellikler)
- [Desteklenen Backend ve Formatlar](#desteklenen-backend-ve-formatlar)
- [Veri Seti](#veri-seti)
- [Mimari Yapı](#mimari-yapı)
- [Benchmark Akışı](#benchmark-akışı)
- [Raporlama Yapısı](#raporlama-yapısı)
- [Önerilen Dizin Yapısı](#önerilen-dizin-yapısı)
- [Kurulum ve Bağımlılıklar](#kurulum-ve-bağımlılıklar)
- [Konfigürasyon Mantığı](#konfigürasyon-mantığı)
- [Örnek Çalıştırma Komutları](#örnek-çalıştırma-komutları)
- [Karşılaştırma Çıktıları](#karşılaştırma-çıktıları)
- [Sınırlılıklar](#sınırlılıklar)
- [Sonuç](#sonuç)

---

## Proje Amacı

Bu framework'ün temel amacı, farklı backend ve model formatlarında çalışan nesne tespit modellerini ortak kurallar altında değerlendirmektir. Sistem, yalnızca modeli çalıştıran basit bir test script'i değildir; aynı zamanda şu bileşenleri tek çatı altında birleştirir:

- veri seti okuma,
- ön işleme,
- backend bağımsız inference,
- son işleme,
- doğruluk hesaplama,
- gecikme ve FPS ölçümü,
- sistem kaynak izleme,
- standart rapor üretimi.

Bu yapı sayesinde bir modelin yalnızca hızlı olması değil, aynı zamanda ne kadar doğru çalıştığı ve sistem üzerinde nasıl bir yük oluşturduğu da birlikte değerlendirilebilir.

---

## Öne Çıkan Özellikler

### 1) Doğruluk metrikleri
Framework, COCO tarzı değerlendirme yaklaşımı ile aşağıdaki metrikleri üretir:

- **mAP@0.5**
- **mAP@0.5:0.95**
- **Precision**
- **Recall**

Her benchmark koşusu sonunda bu metrikler `evaluation_summary` altında raporlanır.

### 2) INT8 vs FP16 doğruluk kaybı analizi
Karşılaştırma araçları içinde yer alan yapı sayesinde, aynı model ailesine ait FP16 ve INT8 çıktıları eşleştirilerek kuantizasyon kaynaklı doğruluk kaybı otomatik olarak raporlanabilir.

### 3) Hız ve gecikme profillemesi
Toplam süre yerine süreçler ayrı ayrı ölçülür:

- preprocess süresi
- inference süresi
- postprocess süresi
- toplam end-to-end süre
- ortalama FPS

Bu sayede darboğazın hangi aşamada oluştuğu daha net analiz edilir.

### 4) Donanım ve kaynak izleme
Benchmark sırasında sistem davranışı da kayıt altına alınır:

- CPU kullanımı
- sistem RAM kullanımı
- proses bazlı RAM kullanımı
- sıcaklık bilgileri

### 5) Çoklu backend desteği
Framework ortak bir arayüz üzerinden birden fazla backend'i destekleyecek şekilde tasarlanmıştır. Böylece yeni backend veya yeni model aileleri eklemek daha kolay hâle gelir.

### 6) Çoklu formatta raporlama
Sistem tekil ve çoklu karşılaştırma senaryoları için rapor üretir:

- JSON
- CSV
- Markdown

---

## Desteklenen Backend ve Formatlar

Framework aşağıdaki format ve çalışma altyapılarını destekleyecek şekilde yapılandırılmıştır:

- **PyTorch** (`.pt`)
- **ONNX Runtime** (`.onnx`)
- **TensorRT engine**
- **RKNN** (`.rknn`)

Bu yapı, benchmark mantığını belirli bir framework'e bağımlı olmaktan çıkarır ve farklı hedef platformlarda karşılaştırma yapmayı kolaylaştırır.

---

## Veri Seti

Ana veri seti olarak:

- **VisDrone2019-DET-val**

kullanılmaktadır.

Bu veri seti 11 sınıf içermektedir ve framework, VisDrone için ayrı bir dataset loader ile çalışacak şekilde tasarlanmıştır. Sınıflar arasında **boat** da yer almaktadır.

---

## Mimari Yapı

Framework tek parça bir script olarak değil, sorumlulukları ayrılmış modüller şeklinde tasarlanmıştır. Bu yaklaşım bakım, test ve genişletilebilirlik açısından avantaj sağlar.

### Çekirdek bileşenler

#### `ConfigLoader`
YAML tabanlı benchmark konfigürasyonlarını okur ve doğrular.

#### `Registry`
Backend adapter'ları, veri setleri ve evaluator bileşenleri için kayıt/çözümleme katmanı sağlar.

#### `BenchmarkRunner`
Uçtan uca akışı yönetir:

- veri setini yükler,
- uygun adapter'ı seçer,
- benchmark sürecini çalıştırır,
- rapor üretimini tetikler.

#### `ReportBuilder` / comparison tools
Tekil koşu sonuçlarını standartlaştırır ve çoklu koşu karşılaştırmalarını CSV ile Markdown formatına dönüştürür.

### Adapter katmanı

Her backend için ayrı adapter kullanılır:

- PyTorch adapter
- ONNX Runtime adapter
- TensorRT adapter
- RKNN adapter

Bu adapter'lar kendi backend'lerine özgü yükleme ve inference detaylarını içerir. Üst katman ise tümünü ortak bir `infer()` mantığı ile kullanır.

### Veri seti ve pipeline katmanı

VisDrone veri seti için ayrı dataset sınıfı bulunur. Pipeline tarafında şu işlemler merkezi olarak yönetilir:

- input size
- letterbox davranışı
- normalize / resize
- backend uyumlu tensor dönüşümü
- bbox / confidence / class ayrıştırma
- standart prediction objelerine dönüştürme

---

## Benchmark Akışı

Tipik bir benchmark koşusu aşağıdaki sırayla çalışır:

1. YAML konfigürasyon dosyası okunur.
2. İlgili veri seti yüklenir.
3. Seçilen backend için adapter başlatılır.
4. Görüntüler ön işleme katmanından geçirilir.
5. Model inference çalıştırılır.
6. Çıktılar postprocess katmanında standart forma dönüştürülür.
7. Tahminler ground truth ile karşılaştırılır.
8. Doğruluk, gecikme ve sistem izleme metrikleri toplanır.
9. Sonuçlar JSON / CSV / Markdown formatlarında raporlanır.

Bu akış sayesinde farklı model ve backend senaryoları tekrar üretilebilir şekilde koşturulabilir.

---

## Raporlama Yapısı

Her benchmark koşusu için standart bir çıktı klasörü oluşturulur. Temel artefakt `report.json` dosyasıdır.

JSON içinde tipik olarak şu ana alanlar yer alır:

- `run_metadata`
- `dataset_metadata`
- `evaluation_summary`
- `latency_summary`
- `monitoring_summary`

### `run_metadata`
Çalıştırılan benchmark'ın kimliğini taşır.

Örnek bilgiler:
- run adı
- backend
- precision
- device

### `dataset_metadata`
Veri setine ait temel bilgileri içerir.

### `evaluation_summary`
Doğruluk metriklerini içerir:
- mAP@0.5
- mAP@0.5:0.95
- precision
- recall

### `latency_summary`
Performans metriklerini içerir:
- preprocess süresi
- inference süresi
- postprocess süresi
- toplam süre
- FPS

### `monitoring_summary`
Benchmark sırasında toplanan sistem bilgilerini içerir:
- CPU
- RAM
- sıcaklık
- proses belleği
- zaman damgaları

---

## Önerilen Dizin Yapısı

```bash
benchmark_framework/
├── benchmark/
│   ├── core/          # runner, registry, config loader
│   ├── adapters/      # pytorch, onnxruntime, tensorrt, rknn
│   ├── datasets/      # visdrone dataset loader
│   ├── evaluators/    # COCO tarzı metric hesaplayıcılar
│   ├── profilers/     # timer profiler vb.
│   ├── monitors/      # cpu/ram/sıcaklık izleme
│   ├── reporters/     # json/csv/markdown raporlama
│   └── tools/         # compare_runs gibi yardımcı araçlar
├── configs/
│   └── runs/          # her benchmark senaryosu için yaml
├── data/
│   └── VisDrone2019-DET-val/
├── models/
├── outputs/
│   ├── runs/
│   └── comparisons/
└── README.md
```

---

## Kurulum ve Bağımlılıklar

Temel çalışma ortamı:

- **Ubuntu**
- **Python 3.10+**

Başlıca bağımlılıklar:

- `ultralytics`
- `torch`
- `onnxruntime`
- TensorRT Python binding'leri
- RKNN toolkit / lite bileşenleri
- `opencv-python`
- `numpy`
- `PyYAML`
- `psutil`

> Not: Özellikle TensorRT ve RKNN tarafında sürüm uyumlulukları dikkatle yönetilmelidir.

Örnek kurulum:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy opencv-python pyyaml psutil ultralytics torch onnxruntime
```

> TensorRT ve RKNN bağımlılıkları platforma göre ayrıca kurulmalıdır.

---

## Konfigürasyon Mantığı

Framework'te run mantığı YAML dosyaları üzerinden yönetilir. Her senaryo için ayrı bir Python script'i yazmak yerine aşağıdaki bilgiler config üzerinden belirlenir:

- model yolu
- backend tipi
- device
- precision
- input size
- class names
- veri seti yolu
- çıktı klasörleri

Bu yaklaşımın avantajları:

- yeni benchmark senaryosu eklemek kolaylaşır,
- kod içindeki sabit path kullanımı azalır,
- tekrar üretilebilirlik artar,
- backend değişiklikleri daha kontrollü yapılır.

---

## Örnek Çalıştırma Komutları

```bash
python3 -m benchmark.cli.main --config configs/runs/yolo26nvisdroneboat_pytorch_cpu_fp32.yaml
python3 -m benchmark.cli.main --config configs/runs/yolo26nvisdroneboat_onnxruntime_cpu_fp32.yaml
python3 -m benchmark.cli.main --config configs/runs/yolo26nvisdroneboat_tensorrt_cuda_fp32.yaml
python3 -m benchmark.cli.main --config configs/runs/yolo26nvisdroneboat_tensorrt_cuda_fp16.yaml
python3 -m benchmark.cli.main --config configs/runs/yolo26nvisdroneboat_tensorrt_cuda_int8.yaml
```

Geliştirme sürecinde kullanılan örnek run senaryoları:

- `yolo26nvisdroneboat_pytorch_cpu_fp32`
- `yolo26nvisdroneboat_onnxruntime_cpu_fp32`
- `yolo26nvisdroneboat_tensorrt_cuda_fp32`
- `yolo26nvisdroneboat_tensorrt_cuda_fp16`
- `yolo26nvisdroneboat_tensorrt_cuda_int8`
- `yolo26nvisdroneboat_rknn_rk3588`

---

## Karşılaştırma Çıktıları

Tekil run raporlarının yanında framework, çoklu koşuların karşılaştırılmasını da destekler.

Bu amaçla seçilen `report.json` dosyaları işlenerek şu çıktılar üretilebilir:

- `comparison_summary.csv`
- `comparison_report.md`

Bu sayede şu senaryolar tek tabloda karşılaştırılabilir:

- PyTorch
- ONNX Runtime
- TensorRT FP32
- TensorRT FP16
- TensorRT INT8
- RKNN

Bu karşılaştırmalar özellikle şu sorular için faydalıdır:

- En yüksek doğruluk hangi backend'te?
- En düşük gecikme hangi precision seviyesinde?
- INT8 dönüşümünün doğruluk kaybı ne kadar?
- Donanım üzerindeki yük nasıl değişiyor?

---

## Sınırlılıklar

Framework'ün ana gereksinimleri büyük ölçüde karşılanmış olsa da bazı alanlar hâlâ genişletilebilir durumdadır:

- Tracy ile tam fonksiyon seviyesinde enstrümantasyon henüz temel kullanım seviyesinde / opsiyoneldir.
- Tüm platformlar için standart NPU yük metriği henüz tek tip değildir.

Bununla birlikte mevcut mimari, yeni backend, evaluator ve raporlama tipleri eklemeye uygundur.

---

## Sonuç

Bu benchmark framework; farklı model formatlarını ortak kurallar altında değerlendiren, tekrar üretilebilir, konfigürasyon tabanlı ve rapor odaklı bir altyapı sunar. VisDrone üzerinde eğitilmiş YOLO modelleri ile başlatılan sistem, ileride farklı veri setleri ve model ailelerine genişleyebilecek şekilde tasarlanmıştır.

En güçlü yönü; **doğruluk**, **hız/gecikme** ve **kaynak kullanımı** verilerini aynı benchmark akışında ve aynı rapor yapısında bir araya getirmesidir. Bu sayede sistem, tek modele özel dağınık script'ler yerine bakım yapılabilir ve profesyonel bir benchmark iskeleti hâline gelmiştir.

---

## Kısa Özet

Bu proje, PyTorch, ONNX Runtime, TensorRT ve RKNN backend'lerinde çalışan nesne tespit modellerini VisDrone veri seti üzerinde ortak bir standartla kıyaslamak için geliştirilmiş modüler bir benchmark framework'üdür. Sistem; doğruluk metrikleri, gecikme/FPS ölçümleri, kaynak izleme ve otomatik raporlama yeteneklerini bir araya getirerek araştırma ve ürünleştirme süreçlerinde kullanılabilecek tekrar üretilebilir bir değerlendirme altyapısı sağlar.
