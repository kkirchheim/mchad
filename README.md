<div align="center">

# Multi-Class Hypersphere Anomaly Detection


<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.5+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.1-89b8cd?style=for-the-badge&labelColor=gray"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

This Repository contains the source code for the paper
_Multi-Class Hypersphere Anomaly Detection_(MCHAD).

![mchad](img/mchad.png)

</div>

## Setup
This repository is a fork of the
[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template), so you might
want to read their excellent instructions on how to use this software.
We configured the Ray Launcher for parallelization.

First, create a python virtual environment, install dependencies, and
add the `src`  directory to your python path.

```
python -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="src/"
```

## Run

Experiments are defined in `config/experiments`.
To run MCHAD on CIFAR10 run:

```
python run.py experiment=cifar10-mchad
```

Each experiment will create a `results.csv` file that contains metrics for all datasets, as
well as a CSV log of the metrics during training, and a TensorBoard log.

### Override Configuration
You can override configuration parameters via the command line, such as:
```shell
python run.py experiment=cifar10-mchad trainer.gpus=1
```
to train on the GPU.



<details>
<summary><b>Live training metrics, embeddings etc. can be visualized with Tensorboard.</b></summary>

```shell
tensorboard --logdir logs/
```

![mchad](img/tb.png)

</details>


## Replication

### Experiments
Experiments can be replicated by running `bash/run-rexperiments.sh`,
which also accepts command line overrides, such as:
```
bash/run-rexperiments.sh "dataset_dir=/path/to/your/dataset/directory/"
```

All datasets will be downloaded automatically to the given `dataset_dir`,
except for the 80 Million TinyImages Dataset ([mirror](http://www.archive.org/download/80-million-tiny-images-2-of-2/tiny_images.bin)), which has to be downloaded and placed there manually.
If you do not want to use the TinyImages Dataset, you can get a cleaned version from
[Hendrycks et al.](https://github.com/hendrycks/outlier-exposure). However, the results might be
different.

Results for each run will be written to `csv` files which have to be aggregated.
You can find our results in the `notebooks/eval.ipynb`.
Per default, we run experiments in parallel on 6 GPUs.
You might have to adjust `config/hydra/launcher/ray.yaml` to configure parallelization.

### Ablations

To replicate the ablation experiments, run:
```shell
bash/run-ablation.sh "dataset_dir=/path/to/your/dataset/directory/"
```

## Results

We average all results over 6 seed replicates and several benchmark outlier datasets.

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">Accuracy</th>
      <th colspan="2" halign="left">AUROC</th>
      <th colspan="2" halign="left">AUPR-IN</th>
      <th colspan="2" halign="left">AUPR-OUT</th>
      <th colspan="2" halign="left">FPR95</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
    </tr>
    <tr>
      <th>Dataset</th>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">CIFAR10</th>
      <th>G-MCHAD</th>
      <td>91.22</td>
      <td>0.22</td>
      <td>85.58</td>
      <td>11.93</td>
      <td>88.75</td>
      <td>10.02</td>
      <td>80.43</td>
      <td>14.04</td>
      <td>55.44</td>
      <td>31.26</td>
    </tr>
    <tr>
      <th>MCHAD</th>
      <td>89.34</td>
      <td>0.27</td>
      <td>83.16</td>
      <td>7.13</td>
      <td>83.14</td>
      <td>7.20</td>
      <td>79.14</td>
      <td>9.74</td>
      <td>66.05</td>
      <td>13.78</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">CIFAR100</th>
      <th>G-MCHAD</th>
      <td>69.56</td>
      <td>0.51</td>
      <td>76.02</td>
      <td>7.91</td>
      <td>80.88</td>
      <td>7.15</td>
      <td>66.88</td>
      <td>10.35</td>
      <td>87.13</td>
      <td>17.48</td>
    </tr>
    <tr>
      <th>MCHAD</th>
      <td>69.80</td>
      <td>0.41</td>
      <td>74.85</td>
      <td>12.49</td>
      <td>78.48</td>
      <td>9.75</td>
      <td>69.96</td>
      <td>13.59</td>
      <td>77.04</td>
      <td>14.67</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">SVHN</th>
      <th>G-MCHAD</th>
      <td>96.13</td>
      <td>0.09</td>
      <td>98.76</td>
      <td>1.99</td>
      <td>99.44</td>
      <td>0.92</td>
      <td>96.63</td>
      <td>5.40</td>
      <td>6.38</td>
      <td>10.24</td>
    </tr>
    <tr>
      <th>MCHAD</th>
      <td>95.36</td>
      <td>0.41</td>
      <td>97.45</td>
      <td>1.20</td>
      <td>99.06</td>
      <td>0.49</td>
      <td>91.87</td>
      <td>4.13</td>
      <td>14.06</td>
      <td>7.90</td>
    </tr>
  </tbody>
</table>


<details>
<summary><b>SVHN</b></summary>

![mchad](img/auroc-SVHN.png)

</details>

<details>
<summary><b>CIFAR10</b></summary>

![mchad](img/auroc-CIFAR10.png)

</details>


<details>
<summary><b>CIFAR100</b></summary>

![mchad](img/auroc-CIFAR100.png)

</details>
