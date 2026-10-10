## English

### Purpose
A reproducible bearing-condition classification study using the official Case Western Reserve University 12 kHz drive‑end vibration recordings. The study evaluates whether models can recognize the same bearing condition at a motor load held out from training.

### Verified Features
- Download script for the 16 CWRU `.mat` files  
- Classical baseline: Extra Trees classifier with engineered features (RMS, kurtosis, skewness, crest/impulse/shape factors, FFT band energy, spectral centroid, entropy, Daubechies‑4 wavelet energy)  
- CNN baseline: fixed 1D‑CNN trained on 0–1 HP, validated on 2 HP, tested on 3 HP  
- Leave‑one‑motor‑load‑out evaluation ensuring no window overlap between train and test folds  
- Artifact generation: per‑load accuracy, macro‑F1, class‑level metrics, normalized confusion matrix, feature importance, calibration error, Brier score, worst‑class recall, additive Gaussian‑noise stress test, feature‑family ablation  
- Single‑file inference script for MATLAB files  

### Stack
- Python (version used in the repository)  
- Dependencies managed via `pyproject.toml` (including scikit‑learn, NumPy, SciPy, PyTorch or TensorFlow as required by the CNN script, ruff, pytest)  

### Setup & Usage
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,cnn]"
python scripts/download_data.py
ruff check .
pytest -q
python scripts/train_classical.py
python scripts/train_cnn.py
```
After training, a CWRU‑style MATLAB file can be checked with:
```bash
python scripts/predict_mat.py data/raw/108.mat
```

### Testing
- Code style: `ruff check .`  
- Unit tests: `pytest -q`

### Limitations
The dataset contains faults seeded by electrical‑discharge machining on a laboratory rig. Strong results do not guarantee performance on naturally developing faults, other sensors, other machines, or noisy industrial installations. This repository is a condition‑monitoring study, not a deployable safety system.

### License
Code is MIT licensed. The dataset remains subject to the terms of its original publisher.

---

## Türkçe

### Amaç
Case Western Reserve Üniversitesi'nin resmi 12 kHz sürüş‑endi titreşim kayıtlarını kullanan, yeniden üretilebilir bir motor pulu arıza tanıma çalışması. Çalışma, modellerin eğitimde görmediği motor yükünde aynı pulu koşulunu tanıyıp tanıyamayacağını değerlendirir.

### Doğrulanmış Özellikler
- 16 CWRU `.mat` dosyasını indirme betiği  
- Klasik baz model: RMS, kurtozis, çarpıklık, crest/impulse/shape faktörleri, FFT bant enerjisi, spektral centroid, entropi, Daubechies‑4 dalga enerjisi mühendis özellikleriyle Extra Trees sınıflandırıcı  
- CNN baz model: 0–1 HP üzerinde eğitilen, 2 HP üzerinde doğrulanan ve 3 HP üzerinde test edilen sabit 1D‑CNN  
- Leave‑one‑motor‑load‑out değerlendirme: eğitim ve test arasında pencerelerin çakışmaması sağlanır  
- Artifact üretimi: yük başına doğruluk, macro‑F1, sınıf‑bazlı metrikler, normalize edilmiş karışıklık matrisi, özellik önemi, kalibrasyon hatası, çok sınıflı Brier skoru, en kötü sınıf geri çağrısı, ek gauss gürültü stres testi, özellik ailesi ablaysonu  
- Tek dosya üzerinden çıkarım için MATLAB dosyası betiği  

### Yığın
- Python (depo tarafından kullanılan sürüm)  
- Bağımlılıklar `pyproject.toml` üzerinden yönetilir (scikit‑learn, NumPy, SciPy, CNN betiği gerektiğinde PyTorch veya TensorFlow, ruff, pytest)  

### Kurulum & Kullanım
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,cnn]"
python scripts/download_data.py
ruff check .
pytest -q
python scripts/train_classical.py
python scripts/train_cnn.py
```
Eğitimden sonra, bir CWRU‑stil MATLAB dosyası şu şekilde kontrol edilebilir:
```bash
python scripts/predict_mat.py data/raw/108.mat
```

### Test
- Kod stili: `ruff check .`  
- Birim testleri: `pytest -q`

### Sınırlamalar
Veri seti, laboratuvar riginde elektrik descarj işlemeyle üretilen hataları içerir. İyi sonuçlar, doğal olarak gelişen hatalar, diğer sensörler, diğer makineler veya gürültülü endüstriyel ortamlar için performans garantisi değildir. Bu depo, bir koşul izleme çalışmasıdır, dağıtılabilir bir güvenlik sistemi değildir.

### Lisans
Kod MIT lisansıyla lisanslanmıştır. Veri seti, orijinal yayıncısının koşullarına tabi kalır.

---

## Deutsch

### Zweck
Eine reproduzierbare Studie zur Klassifizierung von Lagerzuständen anhand der offiziellen 12 kHz Antriebsend‑Vibrationsaufnahmen der Case Western Reserve University. Es wird untersucht, ob Modelle denselben Lagerzustand bei einer beim Training nicht gesehenen Motorlast erkennen können.

### Verifizierte Funktionen
- Skript zum Herunterladen der 16 CWRU `.mat`‑Dateien  
- Klassisches Baseline‑Modell: Extra Trees Klassifikator mit konstruierten Merkmalen (RMS, Kurtosis, Schiefe, Crest/Impulse/Shape‑Faktoren, FFT‑Bandenergie, spektraler Zentroid, Entropie, Daubechies‑4‑Wavelet‑Energie)  
- CNN‑Baseline‑Modell: fester 1D‑CNN, trainiert auf 0–1 HP, validiert auf 2 HP, getestet auf 3 HP  
- Leave‑one‑motor‑load‑out‑Evaluation: Sicherstellung, dass keine Überschneidung zwischen Trainings‑ und Testfenstern besteht  
- Artefakterzeugung: Pro‑Last‑Genauigkeit, Macro‑F1, Klassen‑level‑Metriken, normalisierte Konfusionsmatrix, Feature‑Wichtigkeit, Kalibrierungsfehler, multiclass Brier‑Score, schlechtesten‑Klasse‑Recall, additive Gauß‑Rausch‑Stress‑Test (20 dB bis 0 dB SNR), Feature‑Familien‑Ablation  
- Einzel‑Datei‑Inferenz‑Skript für MATLAB‑Dateien  

### Stack
- Python (Version, wie im Repository verwendet)  
- Abhängigkeiten über `pyproject.toml` verwaltet (scikit‑learn, NumPy, SciPy, bei Bedarf PyTorch oder TensorFlow für das CNN‑Skript, ruff, pytest)  

### Einrichtung & Nutzung
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,cnn]"
python scripts/download_data.py
ruff check .
pytest -q
python scripts/train_classical.py
python scripts/train_cnn.py
```
Nach dem Training kann eine CWRU‑typische MATLAB‑Datei wie folgt überprüft werden:
```bash
python scripts/predict_mat.py data/raw/108.mat
```

### Tests
- Code‑Stil: `ruff check .`  
- Unit‑Tests: `pytest -q`

### Einschränkungen
Der Datensatz enthält künstlich mittels elektrischer Entladungsbearbeitung erzeugte Fehler auf einem Laborprüfstand. Gute Ergebnisse bedeuten nicht, dass das Modell auf natürlich entstandene Fehler, andere Sensoren, andere Maschinen oder rauschhafte Industrieumgebungen übertragbar ist. Dieses Repository dient der Zustandsüberwachung, nicht als einsetzbares Sicherheits‑System.

### Lizenz
Der Code steht unter der MIT‑Lizenz. Der Datensatz unterliegt den Bedingungen des ursprünglichen Herausgebers.
