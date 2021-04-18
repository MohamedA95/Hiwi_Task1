#!/bin/bash
#SBATCH -J OriginalQVGGTRC100
#SBATCH -N 1
#SBATCH -o /home/mmoursi/Hiwi_Task1/res/OriginalQVGGTRC100.out
#SBATCH -e /home/mmoursi/Hiwi_Task1/res/OriginalQVGGTRC100.err
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=4096
#SBATCH --mail-type=BEGIN,END
#SBATCH --time 60:00:00

echo "Executing on $HOSTNAME"
date
module load nvidia/latest
module load cudnn/latest 
python3 /home/mmoursi/Hiwi_Task1/train.py -c /home/mmoursi/Hiwi_Task1/config_json/original_quant_vgg_config.json
date