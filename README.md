# Bird Species Identification from Audio

A deep learning system that identifies 30 UK bird species from audio recordings. Six independently trained models are combined into a weighted ensemble that reaches **97.28% accuracy** on a held-out test set. The project includes the full pipeline: audio preprocessing, model training, ensemble inference, a Flask API and a web interface.

Built as my MSc Computer Science dissertation at the University of Liverpool.

![Web interface](docs/screenshot.png)

## Results

| Model | Test accuracy | Parameters |
|-------|--------------|------------|
| ResNet-50 + CBAM attention | 98.15% | 28.2M |
| ConvNeXt-Tiny | 96.52% | 27.8M |
| EfficientNet-B1 | 95.77% | 7.2M |
| PANNs CNN14 | 88.59% | 88.1M |
| DenseNet-121 | 86.33% | 8.0M |
| Audio Spectrogram Transformer | 85.60% | 10.9M |
| **Weighted ensemble** | **97.28%** | **170.2M** |

Each model's vote is weighted according to its measured accuracy, and the ensemble reports how many models agree on the prediction alongside the confidence score.

The full test-set report (23,793 segments, per-species precision and recall) is in [`results/test_set_report.txt`](results/test_set_report.txt).

![Ensemble confusion matrix](results/normalized_confusion_matrix.png)

## How it works

```
audio file
   │
   ▼
preprocessing ── resample to 44.1 kHz mono
   │             split into 3 s segments with 50% overlap
   │             pre-emphasis filter (0.97)
   │             128-band mel spectrogram (300 Hz – 15 kHz), dB scale, normalised
   ▼
6 models ─────── EfficientNet-B1 · ResNet-50 CBAM · DenseNet-121
   │             AST · ConvNeXt-Tiny · PANNs CNN14
   ▼
ensemble ─────── weighted average / majority voting / stacking (configurable)
   ▼
species, scientific name, confidence, model agreement
```

## Repository layout

| Folder | Contents |
|--------|----------|
| `preprocessing/` | Converts raw recordings into memory-mapped spectrogram datasets for training |
| `training/` | One folder per model, each with its training script, config and helper modules |
| `ensemble/` | Model loading, ensemble strategies, inference and `configs/ensemble_config.yaml` |
| `backend/` | Flask API (`/api/classify`, `/api/health`) that runs the ensemble on uploaded audio |
| `frontend/` | Web interface: audio upload, spectrogram view, per-model predictions, agreement matrix and PDF report export |
| `tests/` | Ensemble tests and test-set evaluation |
| `results/` | Confusion matrices and the test-set evaluation report |

## Tech stack

Python, PyTorch, torchaudio, librosa, timm, scikit-learn, Flask, JavaScript, Chart.js

## Species covered

Barn Owl, Black-headed Gull, Blackcap, Blue Tit, Bullfinch, Chaffinch, Chiffchaff, Coal Tit, Common Blackbird, Coot, Dunnock, Eurasian Magpie, Eurasian Wren, European Greenfinch, European Robin, Fieldfare, Goldcrest, Great Spotted Woodpecker, Great Tit, House Sparrow, Jackdaw, Long-tailed Tit, Mallard, Moorhen, Nuthatch, Pied Wagtail, Starling, Swallow, Tawny Owl, Water Rail

## Running it

The trained model weights (about 1.2 GB) are too large for the repository itself, so they are published as a [GitHub release](https://github.com/calvin223/Bird-Species-Identification/releases/tag/v1.0.0). Download the six `.pth` files into a `weights/` folder at the repository root, either from the release page or with the GitHub CLI:

```bash
gh release download v1.0.0 --repo calvin223/Bird-Species-Identification --dir weights
```

The file names are set in `ensemble/configs/ensemble_config.yaml`:

```
weights/
├── efficientnet_b1.pth
├── resnet50_cbam.pth
├── densenet121.pth
├── ast.pth
├── convnext_tiny.pth
└── panns_cnn14.pth
```

With the weights in place, run these from the repository root:

```bash
pip install -r requirements.txt

# Command-line identification
python ensemble/bird_identifier_fixed.py path/to/recording.wav

# Web interface
python backend/backend_direct.py      # API on http://localhost:5000
# then open frontend/index_final.html in a browser
```

### Retraining

1. Put the raw recordings in `data/raw/bird_recordings/`, one folder per species, and run `python preprocessing/preprocess_selected_30_species_memmap.py`. This writes the spectrogram dataset to `data/bird_calls_30species_memmap/`.
2. Each folder in `training/` is self-contained. Run its script from inside that folder, for example `cd training/resnet50_cbam` and then `python train_resnet50_cbam.py`. In `training/ast/`, `train_ast_bird.py` trains the AST used in the ensemble, while `train_ast.py` builds the larger Hugging Face AST instead.
3. `python tests/evaluate_test_set.py` scores the ensemble on the full test set.

### Reading the confidence score

| Confidence | Meaning |
|------------|---------|
| Above 70% | Strong identification with model consensus |
| 35% to 70% | Probable identification; check recording quality if unexpected |
| Below 35% | Likely an out-of-scope species or a poor recording |

## License

The code is released under the [MIT License](LICENSE). The bird photos in `frontend/assets/birds/` are not covered by it; see that folder's README. The model weights were trained on recordings from [xeno-canto](https://xeno-canto.org), many of which are licensed for non-commercial use only, so the weights are shared for research and educational use.
