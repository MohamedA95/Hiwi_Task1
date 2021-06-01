#!/bin/bash
#SBATCH -J QVGGC100Pre
#SBATCH -N 1
#SBATCH -o /home/mmoursi/Hiwi_Task1/res/QVGGC100Pre.out
#SBATCH -e /home/mmoursi/Hiwi_Task1/res/QVGGC100Pre.err
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8192
#SBATCH --mail-type=END
#SBATCH --time 60:00:00

echo "Executing on $HOSTNAME"
date
module load nvidia/latest
module load cudnn/latest 
python3 /home/mmoursi/Hiwi_Task1/train.py -c /home/mmoursi/Hiwi_Task1/config_json/quant_vgg_pretrained_config.json