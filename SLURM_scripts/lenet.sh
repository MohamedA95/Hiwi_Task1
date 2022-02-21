#!/bin/bash
#SBATCH -J C10_QLeNet2008Bit
#SBATCH -N 1
#SBATCH --chdir /home/mmoursi/Hiwi_Task1
#SBATCH -o res/C10_QLeNet2008Bit.log
#SBATCH -e res/C10_QLeNet2008Bit.log
#SBATCH --gres=gpu:K80:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --mail-type=None
#SBATCH --time=6:00:00

echo "Executing on $HOSTNAME"
date
module load nvidia/latest
module load cudnn/latest 
module load gcc/latest
python3 train_ddp.py -c config_json/lenet.json