# SNV-GenCD

## ⚙️ Requirements

This repo was tested with:
- **Python** 3.11  
- **torch** 2.4.0  
- **CUDA** 12.4  

Create the conda environment from the provided `environment.yaml`:

```bash
conda env create -f environment.yaml
conda activate snv-gencd
💬 Data Preparation
Download the LEVIR-CD and WHU-CD datasets.
Please download them and organize them in the following folder/file structure:

bash
复制代码
├── T1
│   ├── 1.png
│   ├── 2.png
│   ├── 3.png
│   └── ...
│
├── T2
│   ├── 1.png
│   └── ... 
│
├── GT
│   ├── 1.png 
│   └── ...   
│
├── test
│   └── ...
│
├── train_list.txt   
└── test_list.txt
💬 Model Training
Before training models, please enter the changedetection folder, which contains all the code for network definitions, training, and testing:

bash
复制代码
cd <project_path>/MambaCD/changedetection
python script/train_MambaBCD.py
