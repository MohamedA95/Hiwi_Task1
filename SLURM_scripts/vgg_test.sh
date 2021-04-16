#!/bin/bash
#SBATCH -J vggTest
#SBATCH -N 1
#SBATCH -o /home/mmoursi/Hiwi_Task1/res/vggTest.out
#SBATCH -e /home/mmoursi/Hiwi_Task1/res/vggTest.err
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=4096
#SBATCH --mail-type=END
date
echo "Executing on $HOSTNAME"
module load nvidia/latest
module load cudnn/latest 
python3 /home/mmoursi/Hiwi_Task1/test.py --resume /home/mmoursi/Hiwi_Task1/saved/models/CIFAR100_VGG_StepLR/1104_194003/checkpoint-epoch100.pth

