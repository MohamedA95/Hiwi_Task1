#!/bin/bash
#SBATCH -J VGGTRAIN 
#SBATCH -N 1
#SBATCH -o res/vggTrain.out
#SBATCH -e res/vggTrain.err
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=4096
#SBATCH --mail-type=END

echo "Executing on $HOSTNAME"
module load nvidia/latest
module load cudnn/latest 
python3 /home/mmoursi/Hiwi_Task1/train.py -c vgg_config.json

