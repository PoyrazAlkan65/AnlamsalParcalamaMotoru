# Kızagan-TTS: mobil uyumluluk ve ONNX deneyleri

Bu klasördeki betikler Kızagan-TTS v1.0 modelini orijinal (PyTorch) hâliyle ölçmek, ONNX'e
dönüştürmek ve iki sürümü karşılaştırmak içindir. Ağır işler M4 Mac mini'de
(`~/kizagan-onnx`) çalıştırılır; Windows makinesi bu iş için yetersizdir (GPU yok, 11 GB RAM).

| Betik | Ne yapar | Durum |
|---|---|---|
| `kizagan_reference.py` | Orijinal modelle test cümlelerini seslendirir; WAV + `results.json` (süre, RTF, bellek) | Hazır |
| `export_onnx.py` | Modeli parçalara bölüp ONNX'e aktarır | Yazılacak |
| `benchmark.py` | PyTorch ve ONNX çıktılarını hız, boyut ve kalite açısından karşılaştırır | Yazılacak |

Test cümleleri: `data/tts_test/cumleler_tr.txt`.

## 1. Ortam kurulumu (macOS, Apple Silicon)

Gereken yer: ~3 GB Python ortamı + ~5 GB model.

```bash
mkdir -p ~/kizagan-onnx && cd ~/kizagan-onnx
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch torchaudio
VOX="git+https://github.com/OpenBMB/VoxCPM.git@f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69"
uv pip install "voxcpm @ $VOX" "huggingface_hub==1.30.0" "transformers>=4.53"
```

VoxCPM, yayımcının değerlendirmede kullandığı commit'e sabitlenmiştir. `transformers>=4.53`
eski sürümlerdeki bilinen açıkları dışarıda bırakmak içindir.

## 2. Modeli indirme ve doğrulama

```bash
hf download AlicanKiraz0/Kizagan-TTS-v1.0 --revision a22aac8 --local-dir models/kizagan
cd models/kizagan && shasum -a 256 -c SHA256SUMS; cd ../..
```

Bu revizyonda 7 dosya (NOTICE, README.md, inference.py, iki manifest, iki evaluation json)
`FAILED` verir. Nedeni: `a22aac8` commit'i "Kahya → Kızagan" adını değiştirmiş, `SHA256SUMS`
güncellenmemiştir. `inference.py`'nin önceki sürümü (`3bcf201`) listedeki hash'le eşleşir ve
fark yalnızca iki satırlık isim değişikliğidir. Ağırlıklar (`model.safetensors`, `audiovae.pth`),
`config.json`, tokenizer dosyaları ve `tokenization_voxcpm2.py` `OK` verir.

## 3. Orijinal model ölçümü

```bash
python scripts/kizagan/kizagan_reference.py --model models/kizagan \
    --reference models/kizagan/examples/quickstart.wav \
    --sentences-file data/tts_test/cumleler_tr.txt --device mps --output-dir outputs/ref_mps
```

- `--device`: `auto`, `mps`, `cpu` veya `cuda`. CPU ölçümü için `--device cpu`.
- `--cfg`, `--steps`, `--seed`: üretim ayarları; karşılaştırmalarda sabit tutulmalı.
- Çıktı klasörü önceden var olmamalı. İçinde `01.wav`, `02.wav`… ve `results.json` oluşur.
- RTF = işlem süresi / ses süresi; 1'in altı canlı kullanım için yeterlidir. İlk ısınma
  çalıştırması ve model yükleme süresi RTF'ye dahil değildir.
