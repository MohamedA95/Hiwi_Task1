#!/bin/bash
#SBATCH -J QuantVGGTEST 
#SBATCH -N 1
#SBATCH -o /home/mmoursi/Hiwi_Task1/res/QuantvggTest.out
#SBATCH -e /home/mmoursi/Hiwi_Task1/res/QuantvggTest.err
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=4096
#SBATCH --mail-type=END

echo "Executing on $HOSTNAME"
module load nvidia/latest
module load cudnn/latest 
python3 /home/mmoursi/Hiwi_Task1/test.py -x --resume /home/mmoursi/Hiwi_Task1/saved/models/CIFAR10_Quant_VGG/2903_172019/model_best.pth

