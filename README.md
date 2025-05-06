
# Evaluating Multimodal Models for In‑Hospital Mortality Prediction

This repository contains the code and data needed to reproduce our NeurIPS 2024 paper:  
**“Evaluating the Efficacy of Multimodal Models in Clinical Prediction: A Comparative Study of BERT and Multimodal Architectures.”**

We show that a simple TF‑IDF keyword filtering step on clinical notes enables a vanilla BERT model to outperform more complex multimodal fusion architectures on in‑hospital mortality prediction.

---

## 🚀 Quickstart

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/clinical‑bert‑multimodal.git
   cd clinical‑bert‑multimodal
````

2. **Create your environment**

   ```bash
   # using conda
   conda env create -f environment.yml
   conda activate clinical‑bert
   ```

   *or*

   ```bash
   pip install -r requirements.txt
   ```

3. **Download and prepare data**

   ```bash
   bash scripts/download_data.sh
   python notebooks/01_data_preprocessing.ipynb
   ```

4. **Run experiments**

   ```bash
   # text‐only baseline
   python src/train.py --mode text_only --output_dir models/baseline/

   # text + keyword extraction
   python src/train.py --mode text_keyword --output_dir models/bert_filtered/

   # text + structured
   python src/train.py --mode text_struct --output_dir models/text_struct/

   # full multimodal
   python src/train.py --mode text_struct_image --output_dir models/multimodal/
   ```

5. **Evaluate and plot**

   ```bash
   python src/evaluate.py --model_dir models/bert_filtered/ --metrics results/bert_filtered_metrics.csv
   python notebooks/03_plot_results.ipynb
   ```

---

## 📦 Contents

* `data/` — clinical notes CSV, lab values, image embeddings.
* `src/` — core Python modules:

  * **`dataset.py`** defines PyTorch `Dataset` classes for each modality.
  * **`models.py`** implements BERT, 2‑modal, and 3‑modal fusion architectures.
  * **`train.py`** contains training loops with threshold‐calibration and class‑imbalance handling.
  * **`evaluate.py`** computes AUC, F1, precision, recall, etc.
  * **`utils.py`** includes TF‑IDF keyword extraction and cleaning routines.
* `notebooks/` — exploratory and plotting Jupyter notebooks.
* `scripts/` — helper shell scripts to download data and batch‐launch experiments.
* `models/` — saved checkpoints for best‐performing runs.
* `figures/` — high‑resolution PNGs for paper figures.
* `results/` — CSV summaries and training curves.

---

## 📋 Dependencies

* Python ≥ 3.8
* PyTorch ≥ 1.10
* Transformers (Hugging Face)
* scikit-learn, pandas, NumPy, tqdm
* matplotlib, seaborn (optional for plotting)
* jupyter (for notebooks)

*All dependencies listed in* `environment.yml` *or* `requirements.txt`.

---

## 📂 Pretrained Models & Data

* **Text encoder:** Hugging Face `bert-base-uncased` (or `emilyalsentzer/Bio_ClinicalBERT`).
* **Radiology embeddings:** precomputed 1024‑dim vectors from a CNN pretrained on CheXpert.
* **Datasets:**

  * **MIMIC‑IV‑Note v2.2** (free‑text clinical notes)
    Johnson et al. (2023), [https://doi.org/10.13026/1n74-ne17](https://doi.org/10.13026/1n74-ne17)
  * **MIMIC‑CXR v2.0.0** (chest x‑ray embeddings)
    Johnson et al. (2019), [https://doi.org/10.13026/C2JT1Q](https://doi.org/10.13026/C2JT1Q)

Scripts to fetch and preprocess these are in `scripts/`.

---

## 🔄 Reproducibility

* We fix random seeds for NumPy and PyTorch.
* Training, validation splits are stratified on the mortality label (80/20).
* Hyperparameters (learning rate, batch size, weight decay, dropout) are logged in `src/train.py`.
* We apply dynamic threshold calibration to match the positive class prevalence.

---

## 📑 Citation

If you use this code in your work, please cite:

> Yang, A.*, Wu, R.* & Krishnaswamy, S. (2024). Evaluating the Efficacy of Multimodal Models in Clinical Prediction: A Comparative Study of BERT and Multimodal Architectures. *NeurIPS 2024*.

---

## 🤝 Acknowledgments

* This work uses MIMIC‑IV and MIMIC‑CXR from PhysioNet (Johnson et al.).
* We thank **Smita Krishnaswamy** (Yale) for guidance and the TensorFlow team for open‐source tools.
* Code template inspired by Hugging Face and PyTorch examples.

---

## 🔒 License

This project is released under the [MIT License](LICENSE).


