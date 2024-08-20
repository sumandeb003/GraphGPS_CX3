#!/bin/bash
#PBS -l select=1:ncpus=2:mem=8gb:ngpus=1
#PBS -N check-awq-70b
#PBS -k oed
#PBS -j oe
#PBS -l walltime=00:31:00
eval "$(~/miniconda3/bin/conda shell.bash hook)"
cd ~/GraphGPS
conda activate graphgps
python main.py --cfg ~/GraphGPS/configs/GPS/zinc-GPS+RWSE.yaml  wandb.use False
