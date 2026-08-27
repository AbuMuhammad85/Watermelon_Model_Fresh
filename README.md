# Watermelon Disease Classifier Baseline

This project establishes a clean, reproducible dataset pipeline and baseline architecture using `MobileNetV3-Small` to classify watermelon leaf diseases.

## Classes
The dataset classifies leaf images into four distinct classes:
- `Anthracnose`
- `Downy_Mildew`
- `Healthy`
- `Mosaic_Virus`

---

## Project Structure
```
Fresh_Model/
├── data/
│   ├── Watermelon/          # Original dataset (UNMODIFIED)
│   └── processed/           # Created by split script (containing train, val, test)
├── src/
│   ├── __init__.py
│   ├── prepare_data.py      # Stratified train/val/test split and hash check
│   ├── dataset.py           # tf.data pipeline with 224x224 resize and moderate train augmentation
│   ├── baseline_model.py    # MobileNetV3-Small architecture with frozen backbone
│   └── verify_pipeline.py   # Dry-run verification script
├── requirements.txt         # Project dependencies
├── train.py                 # Manual training script with callbacks and class weights
└── evaluate.py              # Validation and locked test evaluation metrics
```

---

## Quick Start

### 1. Requirements
Install the requirements in your Python environment:
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation & Verification
The dataset has already been split. If you ever need to recreate it:
```bash
python src/prepare_data.py
```
This performs a stratified `70% train / 15% val / 15% test` split with `seed=42` and verifies that no cross-split duplicate file content leakage exists.

### 3. Run Pipeline and Model Verification
To check the integrity of the data loader pipelines and the neural network shape compatibility, run:
```bash
python -m src.verify_pipeline
```

### 4. Start Training
To start training the classification head of the model manually:
```bash
python train.py
```
This script will:
- Load the datasets.
- Automatically compute balanced class weights from the training split.
- Initialize MobileNetV3-Small with the frozen pretrained ImageNet backbone and its internal scaling layers enabled.
- Compile and run training using the callbacks: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`, and `CSVLogger`.

### 5. Evaluation
Once training is complete, run the evaluation script to compute accuracy, balanced accuracy, Macro F1, and output the per-class precision/recall and confusion matrix:
```bash
python evaluate.py
```
