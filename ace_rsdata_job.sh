#!/bin/bash
#SBATCH -N1
#SBATCH --partition=shared
#SBATCH --gres=gpu:2

cd /home/mgaskell/work/mqpproject/
source ENV/bin/activate
python ace_rsdata_nn.py
