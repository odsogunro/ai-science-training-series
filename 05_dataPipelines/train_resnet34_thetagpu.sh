#!/bin/bash -l
#COBALT -t 60
#COBALT -q single-gpu

#COBALT -A ALCFAITP

#COBALT -n 1
#COBALT --attrs filesystems=home,grand

# Set up software deps:
module load conda/2022-07-01
conda activate

# You have to point this to YOUR local copy of ai-science-training-series
# cd /home/cadams/ThetaGPU/ai-science-training-series/05_dataPipelines
cd /home/osogunro/Projects/ai-science-training-series/05_dataPipelines

export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
python train_resnet34.py

# module load cobalt/cobalt-gpu

# $ qstat -u osogunro

# $ qsub -I -A ALCFAITP -n 1 -t 30 -q single-gpu

# $ qsub -I -A ALCFAITP -n 1 -t 30 -q full-node

# theta.alcf.gov:6006/#profile