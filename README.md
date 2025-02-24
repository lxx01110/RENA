# RENA
This repository is the official PyTorch implementation of "Dilution of Unreliable Information: Learning in Graph with Noisy
Structures and Absent Attributes". The codes are built on [GraphMAE]( https://github.com/THUDM/GraphMAE).

## Installation

You may use conda to install the environment. Please run the following script. 

```
conda  create  -n RENA
conda activate RENA
pip install -r requirements.txt
```


## Reproducing results in paper
Step 1. Multi-View Structure Dilution to obtain $\mathrm{\hat{A}}$
```
bash MSD.sh
```

Step 2&3 Attributes Reconstruct Dilution & Fine-tuning
```
bash ARD.sh
```
