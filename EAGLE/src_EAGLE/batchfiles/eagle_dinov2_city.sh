#!/bin/bash -l
#SBATCH --job-name=EAGLEv2_City
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --time=0-05:59:59
#SBATCH --mail-user=benearnthof@hotmail.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/logs/EAGLEv2_City.out

source /dss/dsshome1/lxc01/ru25jan4/miniconda3/bin/activate
conda activate /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/miniconda/envs/whisper

export TORCH_HOME=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data/cache_dir
export HF_HOME=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data/cache_dir
export WANDB_CACHE_DIR=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data/cache_dir

NOW=$( date '+%F' )

python -W ignore::UserWarning /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/gitroot/EAGLE/EAGLE/src_EAGLE/train_segmentation_eigen.py
