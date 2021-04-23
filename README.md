# Transformer Model for Cross-Lingual Document Summarization

## Activate virtual environment
1. Use `conda create -n cdsmodel python=3.7` to create a new environment
2. Use `conda activate cdsmodel` to activate the previously created environment

## Clone the project
You can clone the repository with `git clone https://github.com/Cirhuzalain/cdsmodel`
After cloning the project use `cd cdsmodel`

## Install Dependencies on a CUDA enable device
Use `pip install -r requirements.txt` command

## Training & Evaluation
1. Use `python train.py -config config/train.json` command for training
2. Use `python train.py -config config/test.json` command for evaluation

## Built with
* Python
* Numpy
* PyTorch
* Rouge Score
* Tensorboard