#!/usr/bin/env bash


conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

conda install opencv 

cd /home/alexpartisan/Tool/pytorch_scatter 
python setup.py install

cd /home/alexpartisan/Tool/pytorch_sparse 
python setup.py install

cd /home/alexpartisan/Tool/pytorch_cluster 
python setup.py install

pip install torch_geometric
