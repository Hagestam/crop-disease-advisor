
````markdown
# Crop Disease Advisor

A machine learning project for detecting crop diseases from leaf images using computer vision and deep learning.

The project uses the PlantVillage dataset to develop an image classification pipeline capable of identifying different plant disease classes from leaf images.

## Overview

Crop diseases can significantly affect agricultural productivity, while identifying diseases manually can require considerable time and agricultural expertise.

This project explores how machine learning can assist with crop disease identification by analysing images of plant leaves.

The current implementation focuses on building the machine learning pipeline, from dataset acquisition and exploration to model training and evaluation.

## How It Works

```text
PlantVillage Dataset
        ↓
Dataset Download
        ↓
Image Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Disease Classification
````

## Features

* PlantVillage dataset integration
* Automated dataset download
* Image dataset exploration
* Image preprocessing
* Multi-class crop disease classification
* Model training and evaluation
* Automatic generation of class names
* Reproducible Python environment

## Tech Stack

**Language**

* Python

**Machine Learning & Data Science**

* TensorFlow
* Pandas
* NumPy
* Scikit-learn

**Development**

* Jupyter Notebook
* Git
* Kaggle

## Dataset

This project uses the **PlantVillage dataset**, a publicly available dataset containing images of healthy and diseased plant leaves.

The dataset is downloaded programmatically rather than being stored directly in the repository.

This keeps large image files and dataset archives out of Git while allowing the project to be reproduced using the dataset download script.

## Project Structure

```text
crop-disease-advisor/
│
├── data/
│   └── raw/                  # Raw dataset files
│
├── scripts/
│   ├── download_dataset.py   # Downloads the PlantVillage dataset
│   └── save_class_names.py   # Extracts and saves class names
│
├── training/
│   ├── train.py              # Model training pipeline
│   └── explore.ipynb         # Dataset exploration and analysis
│
├── .env                      # Local environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Hagestam/crop-disease-advisor.git
cd crop-disease-advisor
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Kaggle

The dataset is downloaded through Kaggle.

Create a `.env` file containing the Kaggle credentials/configuration required by the dataset download script.

**Do not commit your credentials or `.env` file to GitHub.**

### 5. Download the dataset

```bash
python scripts/download_dataset.py
```

The dataset will be stored locally under:

```text
data/raw/
```

### 6. Explore the dataset

Open:

```text
training/explore.ipynb
```

The notebook can be used to inspect the dataset, understand the class distribution, and examine sample images before training.

### 7. Train the model

```bash
python training/train.py
```

The training script contains the machine learning pipeline used to train the crop disease classification model.

## Development Workflow

The project follows a simple machine learning workflow:

1. **Acquire** — Download the PlantVillage dataset.
2. **Explore** — Inspect images and disease classes.
3. **Prepare** — Preprocess the image data for training.
4. **Train** — Train the disease classification model.
5. **Evaluate** — Measure model performance on unseen data.
6. **Iterate** — Improve preprocessing, architecture, and training configuration.

## Current Focus

The current version of the project focuses primarily on the **machine learning pipeline and model development**.

Future development can extend the trained model into a complete crop advisory application with:

* Image-based disease detection
* Confidence scores
* Disease information
* Treatment and management guidance
* Web or mobile deployment
* API-based inference

## Important Note

This project is intended for educational and research purposes.

Machine learning predictions should not be treated as a definitive agricultural diagnosis. Real-world deployment would require validation using field images and consultation with agricultural experts.

## Author

**Sudi Hagestam**

Computer Science | AI/ML | Data Science

[LinkedIn](https://www.linkedin.com/in/sudihagestam)

[GitHub](https://github.com/Hagestam)

```
