#!/bin/bash
#SBATCH -J alexnet
#SBATCH -N 1
#SBATCH -o /home/mmoursi/Hiwi_Task1/res/alexnet.out
#SBATCH -e /home/mmoursi/Hiwi_Task1/res/alexnet.err
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=4096
#SBATCH --mail-type=END
#SBATCH --time 20:00:00

echo "Executing on $HOSTNAME"
date
module load nvidia/latest
module load cudnn/latest 
python3 /home/mmoursi/Hiwi_Task1/train.py -c /home/mmoursi/Hiwi_Task1/config_json/alexnet_config.json