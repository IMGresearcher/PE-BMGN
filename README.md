# Bi-KAGN

PyTorch implementation of **Bi-KAGN**: an interpretable bearing remaining useful life (RUL) prediction framework based on a **Bidirectional Kolmogorov-Arnold-informed Graph Neural Network**.

## Project structure

```text
Bi-KAGN-open-source/
├── bikagn/
│   ├── configs.py                 # Dataset and model configuration dataclasses
│   ├── kan_layers.py              # KANLinear and Chebyshev GraphKAN convolution
│   ├── wavelet.py                 # Learnable wavelet decomposition module
│   ├── graph.py                   # Adaptive graph, GraphKAN/GAT backbones, gated fusion
│   ├── memory.py                  # Dynamic memory bank and weighted prototype retrieval
│   ├── regressors.py              # KAN, MLP, and MultKAN regression heads
│   ├── model.py                   # Full Bi-KAGN model
│   ├── losses.py                  # Joint training objective
│   ├── training.py                # Generic train/evaluation loops
│   ├── visualization.py           # RUL curves and adjacency heatmaps
│   ├── explainability.py          # KAN function-response visualization
│   ├── utils.py                   # Metrics, seeds, normalization, labels
│   └── data/
│       ├── base.py                # Dataset wrapper and collate function
│       ├── xjtu.py                # XJTU-SY data loading and leave-one-out split
│       └── phm2012.py             # PHM2012/PRONOSTIA data loading and condition split
├── scripts/
│   ├── train_xjtu.py              # XJTU-SY leave-one-out experiment
│   └── train_phm2012.py           # PHM2012 condition-wise experiment
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

Create a clean Python environment first.

```bash
conda create -n bikagn python=3.10 -y
conda activate bikagn
```

Install PyTorch according to your CUDA version from the official PyTorch instructions. For example:

```bash
pip install torch torchvision torchaudio
```

Then install PyTorch Geometric according to your PyTorch/CUDA version. See the official PyG installation guide if the default installation does not match your environment.

```bash
pip install torch-geometric
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

For editable local development:

```bash
pip install -e .
```

## Dataset preparation

### XJTU-SY

Expected layout:

```text
XJTU/
├── Bearing1_1/
│   ├── 1.csv
│   ├── 2.csv
│   └── ...
├── Bearing1_2/
├── ...
└── Bearing3_5/
```

Each CSV file should contain two vibration channels. The loader uses the first two numeric columns.

### PHM2012 / PRONOSTIA

Expected layout:

```text
ieee-phm-2012-data-challenge-dataset-master/
├── Learning_set/
│   ├── Bearing1_1/
│   │   ├── acc_00001.csv
│   │   └── ...
│   ├── Bearing1_2/
│   └── ...
└── Full_Test_Set/
    ├── Bearing1_3/
    ├── Bearing1_4/
    └── ...
```

The default PHM2012 split follows the original challenge protocol:

- Condition I: train `Bearing1_1`, `Bearing1_2`; test `Bearing1_3`–`Bearing1_7`
- Condition II: train `Bearing2_1`, `Bearing2_2`; test `Bearing2_3`–`Bearing2_7`
- Condition III: train `Bearing3_1`, `Bearing3_2`; test `Bearing3_3`

## Quick start

### Train on XJTU-SY

```bash
python scripts/train_xjtu.py \
  --data-root ./XJTU \
  --device cuda:0 \
  --epochs 300 \
  --regressor kan \
  --output-root ./outputs
```

Use the GAT baseline instead of GraphKAN:

```bash
python scripts/train_xjtu.py \
  --data-root ./XJTU \
  --use-gat \
  --regressor kan
```

Use the MultKAN regression head:

```bash
python scripts/train_xjtu.py \
  --data-root ./XJTU \
  --regressor multkan
```

### Train on PHM2012

```bash
python scripts/train_phm2012.py \
  --data-root ./ieee-phm-2012-data-challenge-dataset-master \
  --device cuda:0 \
  --epochs 300 \
  --regressor kan \
  --output-root ./outputs
```

Save KAN function-response explanations during PHM2012 training:

```bash
python scripts/train_phm2012.py \
  --data-root ./ieee-phm-2012-data-challenge-dataset-master \
  --device cuda:0 \
  --epochs 300 \
  --regressor kan \
  --explain
```

## Outputs

Training scripts create timestamped folders under `outputs/` by default.

Typical outputs include:

```text
outputs/
└── xjtu/
    └── YYYYMMDD_HHMMSS/
        └── GraphKAN+KAN/
            ├── xjtu_all_results.csv
            ├── xjtu_condition_mean.csv
            └── Condition1/
                └── rul_plots/
```

For PHM2012:

```text
outputs/
└── phm2012/
    └── YYYYMMDD_HHMMSS/
        └── GraphKAN+KAN/
            ├── phm2012_condition_summary.csv
            ├── phm2012_all_conditions_per_bearing.csv
            └── Condition_I/
                └── rul_plots/
```


## Citation

If this repository is useful, please cite the corresponding paper once it is available.

```bibtex
@article{bikagn2026,
  title   = {A novel interpretable method for bearing remaining useful life prediction based on bidirectional Kolmogorov-Arnold-informed graph neural network},
  author  = {Zhang, Haoxuan and others},
  journal = {Advanced Engineering Informatics},
  year    = {2026}
}
```

## License

Please add a `LICENSE` file before public release. If no license is included, the code is not formally open-source under common GitHub conventions.
