#!/bin/bash
#SBATCH -J QVGGImgPre4bit
#SBATCH -N 1
#SBATCH -o /home/mmoursi/Hiwi_Task1/res/QVGGImgPre4bit.out
#SBATCH -e /home/mmoursi/Hiwi_Task1/res/QVGGImgPre4bit.err
#SBATCH --gres=gpu:V100:1
#SBATCH --exclude=gpu[013,014,015]
#SBATCH --cpus-per-task=2
#SBATCH --mem=8192
#SBATCH --mail-type=START,END


echo "Executing on $HOSTNAME"
date
module load nvidia/latest
module load cudnn/latest 
python3 /home/mmoursi/Hiwi_Task1/train.py -c /home/mmoursi/Hiwi_Task1/config_json/quant_vgg_pretrained_config.json