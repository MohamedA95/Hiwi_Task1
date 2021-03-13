#!/bin/bash
#SBATCH -J QuantVGGTRAIN 
#SBATCH -N 1
#SBATCH -o res/QuantvggTrain.out
#SBATCH -e res/QuantvggTrain.err
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=4096
#SBATCH --mail-type=BEGIN,END
#SBATCH --time 3-0

echo "Executing on $HOSTNAME"
module load nvidia/latest
module load cudnn/latest 
python3 /home/mmoursi/Hiwi_Task1/train.py -c quant_vgg_config.json

