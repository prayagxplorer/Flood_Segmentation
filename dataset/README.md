# Dataset Setup Instructions

To train or test the FloodGateAI model locally, you need to download the dataset from Kaggle and place it in this directory.

**Dataset Link:** [Flood Area Segmentation](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation)

## Instructions:

1. Visit the Kaggle dataset link above and click **Download** (you may need to create a Kaggle account if you don't have one).
2. Extract the downloaded ZIP file.
3. Move the contents of the extracted folder directly into this `dataset/` directory.

Your `dataset/` directory should look exactly like this:

```text
dataset/
|-- README.md
|-- metadata.csv
|-- Image/
`-- Mask/
```

**Note:** The actual dataset files (`metadata.csv`, `Image/`, and `Mask/`) are ignored by Git to keep the repository lightweight. Do not commit them.
