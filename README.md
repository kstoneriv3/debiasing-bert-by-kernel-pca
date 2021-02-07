# Debiasing Bert By Kernel PCA

This repository contains the code for the project "Debiasing BERT by Kernel PCA" carried out as a part of NLP class at ETH Zurich. The report of the project is available [here](https://github.com/kstoneriv3/debiasing-bert-by-kernel-pca/blob/main/final_report.pdf).

## Directory Structure

The structure of the directory is as follows.

```
debiasing-bert-by-kernel-pca
├── __init__.py
├── notebooks
│   └── *.ipynb
├── out
│   ├── logs
│   └── plots
├── README.md
└── src
    ├── bert
    ├── experiments
    ├── kernel_pca
    └── seat
```

The directory `notebook` can be used to keep random jupyter notebook used during the development. The directory `src` contains the code of models and experiments.

## How to Run

```bash
# 1. download dataset to the environment
python ./src/experiments/download_data.py --data_dir {your_data_path} --tasks CoLA,QNLI,SST
# 2. generate embeddings of defining sets and test data
# 3. debias the test data
# 4. evaluate the quality of debiasing
# 5. train the downstream task
python ./src/bert/bert_pipeline.py --data_dir {your_data_path} --out_path {path_to_save_embeddings} --combine_data --debias_mode pca
```
The flag combine data results in combining all available datasets (CoLA, SST2, QNLI) to create the defining sets for debiasing and for metric calculation. Without the flag --data_name should be specified to run the procedure for a specific dataset. 
The argument --debias_method allows you to choose which debiasing method is used.
### How to Run on Leonhard Cluster

```bash
bsub -n 4 -W 24:00 "rusage[mem=8048,ngpus_excl_p=1]" python ./src/bert/bert_pipeline.py --data_dir {your_data_path} --out_path {path_to_save_embeddings} --combine_data --debias_mode kpca
```
