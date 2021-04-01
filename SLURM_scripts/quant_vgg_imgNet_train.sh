#!/bin/bash
#SBATCH -J QuanVGGTRimgNet 
#SBATCH -N 1
#SBATCH -o /home/mmoursi/Hiwi_Task1/res/QuanVGGTRimgNet.out
#SBATCH -e /home/mmoursi/Hiwi_Task1/res/QuanVGGTRimgNet.err
#SBATCH --gres=gpu:V100:2
#SBATCH --nodelist=gpu020
#SBATCH --mem=4096
#SBATCH --mail-type=BEGIN,END
#SBATCH --time 3-0

echo "Executing on $HOSTNAME"
date
module load nvidia/latest
module load cudnn/latest 
python3 /home/mmoursi/Hiwi_Task1/train.py -c /home/mmoursi/Hiwi_Task1/config_json/quant_vggImgNet_config.json

