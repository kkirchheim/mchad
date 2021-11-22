# Multi-Class Hypersphere Anomaly Detection 

This Repository contains the source code for the paper 
_Multi-Class Hypersphere Anomaly Detection_ (MCHAD). 

## Setup 

Create a python virtual environment, install dependencies, and 
add the `src`  directory to your python path. 

```
python -m virtualenv venv 
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="src/"
```

## Run
```
python run.py --help
```


Experiments are defined in `config/experiments`. 
To run MCHAD on a GPU, run:

```
python run.py experiment=cifar10-mchad trainer.gpus=1
```


## Replicate Experiments

```
bash/run-rexperiments.sh
```
Each experiment will create a `results.csv` file that contains all the metrics. 

These files have to be aggregated in a notebook. 
You can find our results in the `notebooks/eval.ipynb`.

## Credits
This repository is a fork of the
[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template). 