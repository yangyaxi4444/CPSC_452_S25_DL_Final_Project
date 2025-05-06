
# Evaluating Multimodal Models for In‑Hospital Mortality Prediction

This repository contains the code and data needed to our paper:  
**“Evaluating the Efficacy of Multimodal Models in Clinical Prediction: A Comparative Study of BERT and Multimodal Architectures.”**

We show that a simple TF‑IDF keyword filtering step on clinical notes enables a vanilla BERT model to outperform more complex multimodal fusion architectures on in‑hospital mortality prediction.

---

## Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/clinical-bert-multimodal.git
cd clinical-bert-multimodal
````

## Prerequisites
- Python 3.8+
- Conda (recommended)

### Setup
```bash
# Create and activate conda environment
conda create -n clinical_pred python=3.8 -y
conda activate clinical_pred

# Install dependencies
pip install -r requirements.txt
```


## Project Structure
```
.
├── notebooks/
│ ├── 1_DataProcess.ipynb # Data cleaning and feature engineering
│ └── six_models_wl_visualization.ipynb # Model training and evaluation
└── requirements.txt # Python dependencies
```

## Dependencies

* Python ≥ 3.8
* PyTorch ≥ 1.10
* Transformers (Hugging Face)
* scikit-learn, pandas, NumPy, tqdm
* matplotlib, seaborn (optional for plotting)
* jupyter (for notebooks)

*All dependencies listed in* `requirements.txt`.

---

## Pretrained Models & Data

* **Text encoder:** Hugging Face `bert-base-uncased` (or `emilyalsentzer/Bio_ClinicalBERT`).
* **Radiology embeddings:** precomputed 1024‑dim vectors from a CNN pretrained on CheXpert.
* **Datasets:**

  * **MIMIC‑IV‑Note v2.2** (free‑text clinical notes)
    Johnson et al. (2023), [https://doi.org/10.13026/1n74-ne17](https://doi.org/10.13026/1n74-ne17)
  * **MIMIC‑CXR v2.0.0** (chest x‑ray embeddings)
    Johnson et al. (2019), [https://doi.org/10.13026/C2JT1Q](https://doi.org/10.13026/C2JT1Q)

Scripts to fetch and preprocess these are in `scripts/`.

---

## Reproducibility

* We fix random seeds for NumPy and PyTorch.
* Training, validation splits are stratified on the mortality label (80/20).
* We apply dynamic threshold calibration to match the positive class prevalence.

---

## Citation

If you use this code in your work, please cite:

> Yang, A.*, Wu, R.* & Krishnaswamy, S. (2024). Evaluating the Efficacy of Multimodal Models in Clinical Prediction: A Comparative Study of BERT and Multimodal Architectures. *NeurIPS 2024*.

---

## Acknowledgments

* This work uses MIMIC‑IV and MIMIC‑CXR from PhysioNet (Johnson et al.).
* We thank **Smita Krishnaswamy** (Yale) for guidance and the TensorFlow team for open‐source tools.
* Code template inspired by Hugging Face and PyTorch examples.

---

