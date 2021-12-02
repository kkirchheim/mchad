<div align="center">

# Multi-Class Hypersphere Anomaly Detection


<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.5+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.1-89b8cd?style=for-the-badge&labelColor=gray"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

This Repository contains the source code for the paper
_Multi-Class Hypersphere Anomaly Detection_ (MCHAD).

![mchad](img/mchad.png)

</div>

## Setup
This repository is a fork of the
[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template), so you might
want to read their thorough instructions on how to use this software.

Create a python virtual environment, install dependencies, and
add the `src`  directory to your python path.

```
python -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="src/"
```

## Run

Experiments are defined in `config/experiments`.
To run MCHAD on a GPU, run:

```
python run.py experiment=cifar10-mchad-o
```

Each experiment will create a `results.csv` file that contains metrics for all datasets, as
well as a CSV log of the metrics during training, and a TensorBoard log.

### Override Configuration
You can override configuration parameters via the command line, such as:
```shell
python run.py experiment=cifar10-mchad trainer.gpus=1
```
to train on the GPU.

## Replicate Experiments
Experiments can be replicated by running `bash/run-rexperiments.sh`,
which also accepts command line overrides, such as:
```
bash/run-rexperiments.sh "dataset_dir=/path/to/your/dataset/directory/"
```

All datasets will be downloaded automatically to the given `dataset_dir`,
except for the 80 Million TinyImages Dataset, which has to be downloaded and placed there manually.

These resulting `csv` files have to be aggregated.
You can find our results in the `notebooks/eval.ipynb`.

### Ablations

To replicate the ablation experiments, run:
```shell
bash/run-ablation.sh "dataset_dir=/path/to/your/dataset/directory/"
```
