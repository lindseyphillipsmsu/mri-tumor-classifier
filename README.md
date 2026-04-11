# 🧠 MRI Tumor Classifier

> A convolutional neural network that detects the presence of brain tumors from MRI scans — because early detection matters and ML can help get us there.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

---

## What This Is

This project trains a binary image classifier to identify whether an MRI brain scan contains a tumor. It's one of the first things I built as I move toward a career in computer vision for healthcare imaging.

The model is trained on publicly available MRI data, preprocessed with OpenCV, and built using a custom CNN architecture in TensorFlow/Keras. Everything is documented so you can follow the logic, not just run the code.

---

## Why I Built This

I'm studying AI at Mississippi State University with a specific goal: build tools that work in clinical and medical imaging environments. That means I need to understand the full pipeline — from raw image data to a model that generalizes well enough to be trusted.

This project is step one. It forced me to think about class imbalance in medical datasets, what accuracy actually means when one wrong prediction could matter, and how preprocessing decisions directly affect what a model learns.

---

## Dataset

**Brain MRI Images for Brain Tumor Detection**
- Source: [Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- Classes: Tumor / No Tumor
- Format: JPEG MRI scan images

Download the dataset and place it in the `/data` folder before running.

---

## Project Structure

```
mri-tumor-classifier/
│
├── data/
│   ├── yes/               # MRI scans with tumor
│   └── no/                # MRI scans without tumor
│
├── notebooks/
│   ├── 01_exploration.ipynb       # Data exploration & visualization
│   ├── 02_preprocessing.ipynb     # Image preprocessing pipeline
│   └── 03_model_training.ipynb    # Model architecture & training
│
├── src/
│   ├── preprocess.py      # Image loading, resizing, normalization
│   ├── model.py           # CNN architecture definition
│   └── evaluate.py        # Metrics, confusion matrix, ROC curve
│
├── results/
│   └── training_curves.png
│
├── requirements.txt
└── README.md
```

---

## Model Architecture

```
Input (224x224x3 MRI Image)
        ↓
Conv2D(32) → ReLU → MaxPooling
        ↓
Conv2D(64) → ReLU → MaxPooling
        ↓
Conv2D(128) → ReLU → MaxPooling
        ↓
Flatten → Dense(128) → Dropout(0.5)
        ↓
Dense(1) → Sigmoid
        ↓
Output: Tumor (1) / No Tumor (0)
```

Dropout is intentional — medical imaging models overfit aggressively on small datasets. Regularization isn't optional here.

---

## How To Run

**1. Clone the repo**
```bash
git clone https://github.com/lindseyphillipsmsu/mri-tumor-classifier.git
cd mri-tumor-classifier
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**
Place the Kaggle dataset in `/data/yes` and `/data/no` respectively.

**4. Run preprocessing**
```bash
python src/preprocess.py
```

**5. Train the model**
```bash
python src/model.py
```

**6. Evaluate**
```bash
python src/evaluate.py
```

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | *updating as training runs* |
| Precision | *updating* |
| Recall | *updating* |
| F1 Score | *updating* |

> Note on metrics: In medical classification tasks, **recall matters more than accuracy**. A false negative — missing a tumor that's there — is clinically far worse than a false positive. This informed every tuning decision.

---

## What I Learned

- How class imbalance in medical datasets silently destroys model performance if you don't catch it
- Why recall is the right metric to optimize in diagnostic ML, not accuracy
- How different preprocessing decisions (normalization range, resize method) shift what features the CNN latches onto
- The practical difference between overfitting on a clean benchmark vs. a messy real-world medical dataset

---

## What's Next

- [ ] Add data augmentation to improve generalization
- [ ] Experiment with transfer learning using VGG16 or ResNet50
- [ ] Add Grad-CAM visualization to show *where* the model is looking in the scan
- [ ] Deploy as a Streamlit web app

---

## Connect

Built by **Lindsey Phillips** — AI student at Mississippi State University, working toward CV engineering in healthcare imaging.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-lindseyphillipsmsu-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/lindseyphillipsmsu)
[![GitHub](https://img.shields.io/badge/GitHub-lindseyphillipsmsu-181717?style=flat&logo=github)](https://github.com/lindseyphillipsmsu)

---

*Part of a larger portfolio focused on machine learning for healthcare imaging and neurotech.*
