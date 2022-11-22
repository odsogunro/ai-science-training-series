#!/bin/bash -l

#COBALT -n 1
#COBALT -t 60
#COBALT -A ALCFAITP
#COBALT -q full-node
#COBALT --attrs filesystems=home,grand

# qsub-gpu -I -n 8 -t 60 -A ALCFAITP -q full-node --attrs filesystems=home,grand

# Set up software deps:
module load conda/2022-07-01
conda activate

# You have to point this to YOUR local copy of ai-science-training-series

export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
mpirun -n 1 python tensorflow2_mnist_hvd.py
mpirun -n 2 python tensorflow2_mnist_hvd.py
mpirun -n 4 python tensorflow2_mnist_hvd.py
mpirun -n 8 python tensorflow2_mnist_hvd.py
# mpirun -n 16 python tensorflow2_mnist_hvd.py
