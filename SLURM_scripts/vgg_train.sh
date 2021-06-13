#!/bin/bash
#SBATCH -J test
#SBATCH -N 1
#SBATCH -o /home/mmoursi/Hiwi_Task1/res/test.out
#SBATCH -e /home/mmoursi/Hiwi_Task1/res/test.err
#SBATCH --gres=gpu:K80:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8192
#SBATCH --mail-type=END
#SBATCH --time 2:00:00

echo "Executing on $HOSTNAME"
date
module load nvidia/latest
module load cudnn/latest 
python3 /home/mmoursi/Hiwi_Task1/train.py -c /home/mmoursi/Hiwi_Task1/config_json/vgg_config_copy.json