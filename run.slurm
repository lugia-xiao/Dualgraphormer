#!/bin/bash



#SBATCH --job-name=repeat_best

#SBATCH --output=1_normlr.out

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=10

#SBATCH --gpus=a100:1

#SBATCH --partition=gpu

#SBATCH --time=2-00:00:00

#SBATCH --mail-user=xiao.xiao.xx244@yale.edu

#SBATCH --mail-type=END



python -u train_normlr.py
