
# Tripple GAN for Occular Image Regeneration

This is the implementation of the paper "Tripple GAN for Occular Image Regeneration"

## Requirements

- Python 3.10 or later
- Pytorch
- Numpy
- Matplotlib
- tensorflow
- keras

### Installation

```bash scripts/install_deps.sh``` to install the dependencies
or
```pip3 install -r requirements.txt``` to install the dependencies

### Dataset

Download dataset from kaggle, first add your kaggle.json file to your home environment (~/.kaggle/kaggle.json)
then run the following command to download the dataset
```bash scripts/download_dataset.sh``` to download the dataset
or
```python3 download_dataset.py --dataset_name 'data_user/dataset_name' --dataset_path 'datasets/data_user/``` in your terminal or command prompt to download the dataset

## Usage

```bash scripts/train_model.sh``` to train the model
or
```python3 train.py --root_dir '/path/to/dataset/' --train``` to train the model

### NOTE

There are other parameters that can easily be modified either in the options.py file
or by appending them in the scripts/train_model.sh file through the argument parser
