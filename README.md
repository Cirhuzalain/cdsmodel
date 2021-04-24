# CDS : Cross-Lingual Document Summarization (WIP)

## Activate virtual environment
1. Use `conda create -n cdsmodel python=3.7` to create a new environment
2. Use `conda activate cdsmodel` to activate the previously created environment

## Clone the project
You can clone the repository with `git clone https://github.com/Cirhuzalain/cdsmodel`
After cloning the project use `cd cdsmodel`

## Install Dependencies on a CUDA enable device
Use `pip install -r requirements.txt` command

## Training & Evaluation on Dummy Data
1. Use `python train.py -config config/train.json` command for training
2. Use `python train.py -config config/test.json` command for evaluation

## Resources
* AMMI NLP 2 Course
* Hugging Face Transformer
* Optuna (https://optuna.org/)
* NCLS : (https://arxiv.org/abs/1909.00156)

## Built with
* Python
* Numpy
* PyTorch
* Rouge Score
* Tensorboard