# informer_ChineseNotes

![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg?style=plastic)
![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4-green.svg?style=plastic)

[HiSTGNN](https://github.com/Irenegj/HiSTGNN) is the unified data-driven framework for multi-UAV risk assessment has been developed to incorporate interaction factors within swarm-defense systems

## Requirements

The model is implemented using Python3 with dependencies specified in requirements.txt

## Quick Configuration Environment

### Method 1：
Create a virtual environment in Anaconda Powershell Prompt (Anaconda3) :
```
conda create -n torch_lts python=3.7
```
Then enter the virtual environment:
```
activate torch_lts
```
Finally, install the environment dependency package using the requirements. TXT file:
```
pip install -r requirements.txt

```

### Method 2
Execute the following commands directly in your environment to install:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c conda-forge

pip install numpy == 1.19.4
pip install pyecharts
pip install xlrd
pip install openpyxl
pip install matplotlib
pip install pandas

```

## reference
* paper: [HiSTGNN: Hierarchical Spatio-Temporal Graph Neural Network for Multi-UAV Survivability Assessment](https://ieeexplore.ieee.org/document/11165742)

## To contact me
* e-mail：irenegj@mail.hfut.edu.cn




