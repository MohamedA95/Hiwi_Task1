#!/bin/bash
#SBATCH -J modelTest
#SBATCH -N 1
#SBATCH -o /home/mmoursi/Hiwi_Task1/res/modelTest.out
#SBATCH -e /home/mmoursi/Hiwi_Task1/res/modelTest.err
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=4096
#SBATCH --mail-type=END
date
echo "Executing on $HOSTNAME"
module load nvidia/latest
module load cudnn/latest 
python3 /home/mmoursi/Hiwi_Task1/test.py --resume /home/mmoursi/Hiwi_Task1/saved/models/C100QVGG/2905_014014/model_best.pth

