#!/bin/bash
#SBATCH -J C100QVGG
#SBATCH -N 1
#SBATCH -o /home/mmoursi/Hiwi_Task1/res/C100QVGG.out
#SBATCH -e /home/mmoursi/Hiwi_Task1/res/C100QVGG.err
#SBATCH --gres=gpu:K80:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8096
#SBATCH --mail-type=END
#SBATCH --time 30:00:00

echo "Executing on $HOSTNAME"
date
module load nvidia/latest
module load cudnn/latest 
python3 /home/mmoursi/Hiwi_Task1/train.py -c /home/mmoursi/Hiwi_Task1/config_json/quant_vgg_config.json