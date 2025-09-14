# SNV-GenCD

## ⚙️ Requirements

This repo was tested with:
- **Python** 3.8  
- **PyTorch** 1.7.1  

Create the conda environment from the provided `environment.yaml`:

```bash
conda env create -f environment.yaml
conda activate snv-gencd

## 💬 Data Preparation

Download the LEVIR-CD and WHU-CD datasets.

Data preprocessing is introduced in the original paper.

Check the file `loaders/datasets.py` and adjust it if needed for your own dataset.

Modify the argument `--data_root` in the training scripts to the dataset path.

## 💬 Training

We provide training scripts for LEVIR-CD and WHU-CD datasets.



