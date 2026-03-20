# Lymphoma Detection

Classification of histopathological lymphoma images using feature extraction from pre-trained CNNs (MobileNetV3 Small, ResNet50) and k-NN classification. This project was developed as part of a university lab assignment on image representation and classification.

The dataset is a subset of the [Multi-Cancer Dataset](https://www.kaggle.com/datasets/obulisainaren/multi-cancer) from Kaggle, containing 15,000 images equally divided into three classes: CLL (chronic lymphocytic leukemia), FL (follicular lymphoma), and MCL (mantle cell lymphoma).

## Setup

```bash
python3 -m venv lymphoma-detection-venv
source lymphoma-detection-venv/bin/activate
pip install -r requirements.txt
```

## Download Dataset

Use the provided script to download the dataset from Kaggle (requires Kaggle API credentials):

```bash
bash kaggle-data.sh
```

Extract the lymphoma subset into a `lymphoma-database/` directory with subdirectories `lymph_cll/`, `lymph_fl/`, and `lymph_mcl/`.

## Usage

Run the full pipeline (feature extraction + k-NN classification):

```bash
bash exec.sh
```

This executes `feature_extraction.py` followed by `knn.py` and logs the output to `output.log`.

You can also run each step individually:

```bash
python3 feature_extraction.py   # Extract features from images
python3 knn.py                  # Train and evaluate k-NN
```

## Project Structure

| Path | Description |
|------|-------------|
| `feature_extraction.py` | Extracts feature vectors from images using a pre-trained CNN |
| `knn.py` | Trains and evaluates a k-NN classifier on the extracted features |
| `exec.sh` | Runs the full pipeline and logs output |
| `kaggle-data.sh` | Downloads the dataset from Kaggle |
| `requirements.txt` | Python dependencies |
| `results/` | Logs from each experiment with configurations and metrics |
| `report/` | Academic report in IEEE two-column format (LaTeX) |