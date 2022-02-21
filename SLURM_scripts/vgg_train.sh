#!/bin/bash
#SBATCH -J devDDP
#SBATCH -N 1
#SBATCH --chdir /home/mmoursi/Hiwi_Task1
#SBATCH -o res/devDDP_out.log
#SBATCH -e res/devDDP_err.log
#SBATCH --exclude=gpu[013,014,015]
#SBATCH --gres=gpu:V100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --mail-type=START,END
#SBATCH --time=4-00:00:00

echo "Executing on $HOSTNAME"
date
module load nvidia/latest
module load cudnn/latest 
module load gcc/latest
python3 train_ddp.py -c config_json/vgg_config.json