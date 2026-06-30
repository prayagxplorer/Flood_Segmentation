# FloodGateAI Model Training

This repository contains the deep learning training pipeline for FloodGateAI: a PyTorch UNet model for binary flood segmentation.

## Repository Layout

```text
Flood_Segmentation/
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- flood-segmentation-model.ipynb
`-- dataset/
    |-- Image/
    `-- Mask/
```

The `dataset/` folder is where you should place the external dataset. 

## Data Setup

Do not commit the full dataset to this repository. Place the extracted Kaggle dataset directly inside the `dataset/` folder, which is ignored by git. For details, see `dataset/README.md`.

Expected local dataset layout:

```text
dataset/
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
jupyter notebook flood-segmentation-model.ipynb
```

Training outputs are written to `working/` locally or `/kaggle/working/` on Kaggle. Generated checkpoints and zip files are ignored by git.
