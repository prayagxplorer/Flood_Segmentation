# FloodGateAI Model Training

This repository contains the deep learning training pipeline for FloodGateAI: a PyTorch UNet model for binary flood segmentation.

## Repository Layout

```text
Flood_Segmentation/
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- Model.ipynb
`-- sample_data/
    |-- images/
    `-- masks/
```

The `sample_data/` folder contains exactly two real image/mask pairs for quick dataloader verification. It is not meant for real model training.

## Data Setup

Do not commit the full dataset to this repository. Place the external dataset in a local `data/` folder, which is ignored by git.

Expected local dataset layout:

```text
data/
|-- metadata.csv
|-- Image/
`-- Mask/
```

`metadata.csv` must contain `Image` and `Mask` columns that point to matching filenames.

When running on Kaggle, attach the dataset to the notebook. The notebook searches `/kaggle/input/` automatically.

## Model Architecture

- Custom PyTorch UNet for single-channel binary semantic segmentation.
- Raw logits output, with sigmoid applied inside the loss and metric logic.
- Albumentations for resizing, flips, rotation, affine transforms, brightness/contrast changes, dropout, and ImageNet normalization.
- Combined BCE + Dice loss to handle class imbalance between flood and background pixels.
- Adam optimizer with weight decay and cosine annealing learning-rate scheduling.
- Dice coefficient validation metric with early stopping.
- `nn.DataParallel` support for multi-GPU training, including Kaggle dual T4 sessions.

## Local Setup

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Then open and run:

```bash
jupyter notebook Model.ipynb
```

Training outputs are written to `working/` locally or `/kaggle/working/` on Kaggle. Generated checkpoints and zip files are ignored by git.
