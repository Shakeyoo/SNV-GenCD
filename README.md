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

Download the [LEVIR-CD](https://justchenhao.github.io/LEVIR/) and [WHU-CD](http://gpcv.whu.edu.cn/data/building_dataset.html) datasets.
Please download them and make them have the following folder/file structure:

```bash
│   ├── T1
│   │   ├──1.png
│   │   ├──2.png
│   │   ├──3.png
│   │   ...
│   │
│   ├── T2
│   │   ├──1.png
│   │   ... 
│   │
│   └── GT
│       ├──1.png 
│       ...   
│   
├── test
│   ├── ...
│   ...
│  
├── train_list.txt   
└── test_list.txt


💬Model Training
Before training models, please enter into [changedetection] folder, which contains all the code for network definitions, training and testing.
cd <project_path>/MambaCD/changedetection
python script/train_MambaBCD.py






