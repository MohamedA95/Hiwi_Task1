#!/bin/bash
#SBATCH -J QuantVGGC10TRAIN 
#SBATCH -N 1
#SBATCH -o res/QuantvggC10Train.out
#SBATCH -e res/QuantvggC10Train.err
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=4096
#SBATCH --mail-type=BEGIN,END
#SBATCH --time 3-0

echo "Executing on $HOSTNAME"
module load nvidia/latest
module load cudnn/latest 
python3 /home/mmoursi/Hiwi_Task1/train.py -c quant_vggC10_config.json

