# GTF_ResNet50tuned — Gulf Toadfish Boatwhistle Detector

Matched-filter detector and fine-tuned ResNet50 classifier for automated detection of Gulf toadfish (*Opsanus beta*) boatwhistle calls in passive acoustic recordings.

Implementations are provided in both MATLAB (Deep Learning Toolbox) and Python (PyTorch).

**Authors:** D. Bohnenstiehl, North Carolina State University

---

## Repository structure

```
GTF_ResNet50tuned/
├── matlab/
│   ├── gtfDeepToadFinetuneScript.m   % fine-tune ResNet50 on spectrogram images
│   ├── wrapper.m                      % main processing loop (per-file detection)
│   ├── gtfDeepCatalogger_ver1.m       % per-segment detector + classifier
│   ├── gtfMatchedFilterDet_ver1.m     % matched-filter boatwhistle detector
│   └── gtfMakespectro_ver1.m          % spectrogram image generator (parula, 224×224)
├── python/
│   ├── gtfDeepToadFinetuneScript.py   # fine-tune ResNet50 on spectrogram images
│   ├── wrapper.py                     # main processing loop (per-file detection)
│   ├── gtfDeepCatalogger_ver1.py      # per-segment detector + classifier
│   ├── gtfMatchedFilterDet_ver1.py    # matched-filter boatwhistle detector
│   ├── gtfMakespectro_ver1.py         # spectrogram image generator
│   ├── gtfParseTrainJpg.py            # utility: generate training JPEG images
│   ├── parula256.csv                  # exact MATLAB parula(256) colormap export
│   └── requirements.txt
├── timetables_csv/                    # site master index tables (CSV) — not distributed
├── timetables_mat/                    # site master index tables (MAT) — not distributed
├── betaout/                           # MATLAB detection output — not distributed
└── betaout_py/                        # Python detection output — not distributed
```

---

## Model weights

Pre-trained classifier weights (`gtfclassifier_tuned.mat` / `gtfclassifier_tuned_py.pt`) are included in the repository. 
---

## MATLAB usage

Requires MATLAB with the **Deep Learning Toolbox** and **Signal Processing Toolbox**.

Set MATLAB's working directory to `matlab/` before running any script.

**Retrain the ResNet50 classifier in MATLAB:**
```matlab
% You can edit datasetDir and thr_method at the top of the script, then run:
gtfDeepToadFinetuneScript
```

**Run the detector on a site:**
```matlab
% Edit 'site' at the top of wrapper.m, then run:
wrapper
```

Site master index `.mat` files (`tt_*.mat`) must be present in `timetables_mat/`.

---

## Python usage

**Install dependencies:**
```bash
# PyTorch — install via conda/mamba for CUDA support
mamba install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Remaining packages
pip install -r python/requirements.txt
```

All scripts are run from the `python/` directory.

**Retrain the ResNet50 classifier in python classifier:**
```bash
cd python
python gtfDeepToadFinetuneScript.py
```
Edit `DATA_DIR`, `THR_METHOD`, and other settings in the `Config` block at the top of the script.

**Run the detector on a site:**
```bash
cd python
python wrapper.py
```
Edit `SITE` and `CLASSIFIER_PATH` in the `Config` block. Site master index CSV files must be present in `timetables_csv/`.

---

## Spectrogram consistency

Training images and inference spectrograms must use the same pipeline. Both implementations:
- Apply the exact MATLAB `parula(256)` colormap (exported to `parula256.csv`)
- Encode spectrograms as quality-95 JPEG before classifier input
- Use 224×224 px output with robust contrast scaling

The `parula256.csv` file ensures colormap consistency across platforms. But there are minor difference in the way MATLAB and Python do spectrograms and filtering. 

---

## Notes on Threshold selection

After training, the decision threshold is selected from the test-set sweep. Both scripts support three methods (set `thr_method` / `THR_METHOD` in the config block):

| Method | Description |
|---|---|
| `F1 plateau` | Midpoint of the threshold range within 0.1% of peak F1 **(default)** |
| `max F1` | Threshold that maximises F1 on the test set |
| `min cost` | Threshold that minimises weighted FP/FN cost (FP weight = 1.075) |
